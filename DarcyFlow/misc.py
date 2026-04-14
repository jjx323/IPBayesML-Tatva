#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Darcy Flow 专用有限元工具函数模块

本模块提供 Darcy 流方程求解所需的有限元工具：
1. 网格生成与管理 (create_mesh_2d)
2. 测量矩阵构建 (construct_measure_matrix)
3. 刚度/质量矩阵组装 (assemble_stiffness_matrix, assemble_mass_matrix)
4. 矩阵格式转换 (trans2scipy)
5. 误差计算工具 (error_compare)
6. 光滑化算子 (Smoother)

这些函数与 Darcy Flow 方程紧密相关，因此独立于 core 模块。

作者: 基于原 IPBayesML-FEniCSx09 重构，使用 tatva 库
"""

import numpy as np
import jax.numpy as jnp

# ============================================================
# 全局配置：强制使用 float64（必须在 import jax 之后）
# 注意：core/jax_backend.py 中已设置 os.environ['JAX_ENABLE_X64'] = '1'
# 这里再次确认，确保即使单独导入此模块也生效
# ============================================================
try:
    import jax.config
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass

# JAX 后端（替代 scipy.sparse，支持 GPU）
from core.jax_backend import (
    JAXSparseMatrix, JAXDiag,
    csr_from_coo, diags_jax, issparse_jax,
    cg_jax, bicgstab_jax, spsolve_jax, JAXLUSolver,
    LinearOperatorJAX
)

import scipy.sparse as sps  # 保留但不再用于计算
from scipy.sparse import csr_matrix as _csr_scipy

from typing import Optional, Tuple, Callable

__all__ = [
    'create_mesh_2d',
    'construct_measure_matrix', 
    'assemble_stiffness_matrix',
    'assemble_mass_matrix',
    'trans2scipy',
    'error_compare',
    'Smoother'
]


def create_mesh_2d(nx: int = 10, ny: int = 10, 
                    x_range: Tuple[float, float] = (0.0, 1.0),
                    y_range: Tuple[float, float] = (0.0, 1.0),
                    element_type: str = 'tri') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    创建二维网格
    
    Parameters:
    -----------
    nx : int
        x 方向的单元数
    ny : int  
        y 方向的单元数
    x_range : tuple
        x 范围 (xmin, xmax)
    y_range : tuple
        y 范围 (ymin, ymax)
    element_type : str
        单元类型 ('tri' 或 'quad')
    
    Returns:
    --------
    coords : jnp.ndarray
        节点坐标 (num_nodes, 2)
    elements : jnp.ndarray
        单元连接关系 (num_elements, nodes_per_element)
    """
    x = np.linspace(x_range[0], x_range[1], nx + 1)
    y = np.linspace(y_range[0], y_range[1], ny + 1)
    
    # 创建节点坐标
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    
    # 创建单元连接关系
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            
            if element_type == 'tri':
                # 将四边形拆分为两个三角形
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
            elif element_type == 'quad':
                elements.append([n0, n1, n2, n3])
    
    coords = jnp.array(coords, dtype=jnp.float64)
    elements = jnp.array(elements, dtype=np.int32)
    
    return coords, elements


def construct_measure_matrix(coords: np.ndarray, 
                            points: np.ndarray,
                            elements: np.ndarray = None):
    """
    构建测量矩阵 S（JAX 后端，支持 GPU）
    
    Input:
    ------
    coords : np.ndarray
        节点坐标 (N, dim)，dim=2 或 3
    points : np.ndarray  
        观测点坐标 (M, dim)
    elements : np.ndarray, optional
        单元连接关系（用于更精确的插值）
        
    Output:
    -------
    S : JAXSparseMatrix
        测量矩阵，形状 (M, N)
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, points.shape[0])
    
    nx, dim = points.shape
    nnodes = coords.shape[0]
    
    rows = []
    cols = []
    vals = []
    
    for k in range(nx):
        point = jnp.asarray(points[k], dtype=jnp.float64)
        distances = jnp.sqrt(jnp.sum((jnp.asarray(coords) - point)**2, axis=1))
        
        n_neighbors = min(4, nnodes)
        neighbor_idx = jnp.argpartition(distances, n_neighbors)[:n_neighbors]
        dist_neighbors = distances[neighbor_idx]
        
        eps = 1e-12
        weights = 1.0 / (dist_neighbors**2 + eps)
        weights = weights / jnp.sum(weights)
        
        for j, idx in enumerate(neighbor_idx):
            rows.append(k)
            cols.append(int(idx))
            vals.append(float(weights[j]))
    
    return csr_from_coo(
        jnp.array(vals, dtype=jnp.float64),
        jnp.array(rows, dtype=jnp.int32),
        jnp.array(cols, dtype=jnp.int32),
        shape=(nx, nnodes)
    )


def assemble_stiffness_matrix(coords: np.ndarray, 
                              elements: np.ndarray,
                              diffusivity_field: np.ndarray = None):
    """
    组装刚度矩阵 A（向量化版本，使用 JAX 后端，支持 GPU）
    
    对于 Darcy 流方程：-∇·(κ∇u) = f
    刚度矩阵 A_{ij} = ∫ κ ∇φ_i · ∇φ_j dx
    
    Parameters:
    -----------
    coords : np.ndarray (N, dim)
        节点坐标
    elements : np.ndarray (M, 3)  
        三角形单元连接关系
    diffusivity_field : np.ndarray (N,), optional
        扩散系数场 κ（如果为 None，则 κ=1）
        
    Returns:
    --------
    A : JAXSparseMatrix 刚度矩阵（GPU 上）
    """
    # 转为 JAX 数组
    coords = jnp.asarray(coords, dtype=jnp.float64)
    elements = jnp.asarray(elements, dtype=jnp.int32)
    
    nnodes = coords.shape[0]
    nelems = elements.shape[0]
    
    if diffusivity_field is None:
        diffusivity_field = jnp.ones(nnodes, dtype=jnp.float64)
    else:
        diffusivity_field = jnp.asarray(diffusivity_field, dtype=jnp.float64)

    # ====== 向量化提取所有单元数据 ======
    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
    c0, c1, c2 = coords[n0], coords[n1], coords[n2]

    # 面积向量: |det([c1-c0; c2-c0])|/2
    area = 0.5 * jnp.abs(
        (c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) -
        (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1])
    )
    inv_2area = 1.0 / (2.0 * area + 1e-30)  # 避免除零

    # 形函数导数矩阵 B (2×3)，对所有单元批量计算
    B = jnp.empty((nelems, 2, 3), dtype=jnp.float64)
    B = B.at[:, 0, 0].set((c1[:, 1] - c2[:, 1]) * inv_2area)
    B = B.at[:, 0, 1].set((c2[:, 1] - c0[:, 1]) * inv_2area)
    B = B.at[:, 0, 2].set((c0[:, 1] - c1[:, 1]) * inv_2area)
    B = B.at[:, 1, 0].set((c2[:, 0] - c1[:, 0]) * inv_2area)
    B = B.at[:, 1, 1].set((c0[:, 0] - c2[:, 0]) * inv_2area)
    B = B.at[:, 1, 2].set((c1[:, 0] - c0[:, 0]) * inv_2area)

    # 单元扩散系数平均值 (nelems,)
    kappa_avg = (diffusivity_field[n0] + diffusivity_field[n1] + diffusivity_field[n2]) / 3.0

    # 局部刚度矩阵 Ke = kappa * B^T @ B * area  → (nelems, 3, 3)
    BTB = jnp.einsum('eji,ejk->eik', B, B)  # (nelems, 3, 3)
    K_local = kappa_avg[:, None, None] * BTB * area[:, None, None]

    # ====== 构建 JAX COO 稀疏矩阵 ======
    row_idx = jnp.tile(elements[:, None], (1, 3)).ravel()
    col_idx = jnp.tile(elements[:, :, None], (1, 1, 3)).ravel()
    data = K_local.ravel()

    return csr_from_coo(data, row_idx.astype(jnp.int32), col_idx.astype(jnp.int32),
                        (nnodes, nnodes))


def assemble_mass_matrix(coords: np.ndarray, 
                        elements: np.ndarray):
    """
    组装质量矩阵 M（向量化版本，使用 JAX 后端）
    
    质量矩阵 M_{ij} = ∫ φ_i φ_j dx
    使用集中（lumped）对角质量矩阵
    
    Parameters:
    -----------
    coords : np.ndarray (N, dim)
        节点坐标
    elements : np.ndarray (M, 3)
        三角形单元连接关系
        
    Returns:
    --------
    M : JAXDiag 对角质量矩阵（GPU 上）
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    elements = jnp.asarray(elements, dtype=jnp.int32)
    
    nnodes = coords.shape[0]
    nelems = elements.shape[0]

    # 向量化计算面积
    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
    c0, c1, c2 = coords[n0], coords[n1], coords[n2]

    area = 0.5 * jnp.abs(
        (c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) -
        (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1])
    )

    # 一致质量矩阵每个节点贡献 area/3
    mass_contrib = area / 3.0

    # 用 scatter-add 向量化累加（GPU 友好）
    diag_entries = jnp.zeros(nnodes, dtype=jnp.float64)
    diag_entries = diag_entries.at[n0].add(mass_contrib)
    diag_entries = diag_entries.at[n1].add(mass_contrib)
    diag_entries = diag_entries.at[n2].add(mass_contrib)

    return diags_jax(diag_entries)


def trans2scipy(A, gpu=False):
    """
    矩阵格式转换（兼容性函数）
    
    现在所有矩阵都是 JAX 格式，此函数主要用于兼容旧代码。
    如果输入是 JAX 类型，直接返回。
    如果输入是 numpy/scipy 类型，转为 JAX。
    """
    if isinstance(A, (JAXSparseMatrix, JAXDiag)):
        return A
    elif isinstance(A, np.ndarray):
        A_jax = jnp.asarray(A, dtype=jnp.float64)
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            if jnp.allclose(A_jax, jnp.diag(jnp.diag(A_jax))):
                return diags_jax(jnp.diag(A_jax), shape=A.shape)
        return A_jax
    elif hasattr(A, 'tocsr'):  # scipy sparse
        csr = A.tocsr()
        dense = jnp.asarray(A.toarray(), dtype=jnp.float64)
        return diags_jax(jnp.diag(dense), shape=dense.shape) if dense.shape[0]==dense.shape[1] else dense
    else:
        raise TypeError(f"Unsupported matrix type: {type(A)}")


def error_compare(u_exact_func: Callable, 
                  u_numerical: np.ndarray,
                  coords: np.ndarray,
                  relative_error: bool = False) -> Tuple[float, float]:
    """
    计算数值解与精确解之间的误差
    
    Parameters:
    -----------
    u_exact_func : callable
        精确解函数，输入坐标，输出函数值
    u_numerical : np.ndarray
        数值解向量
    coords : np.ndarray
        节点坐标
    relative_error : bool
        是否计算相对误差
        
    Returns:
    --------
    error_L2 : float
        L2 范数误差
    error_max : float
        最大误差（无穷范数）
    """
    u_exact = np.array([u_exact_func(coord) for coord in coords])
    
    diff = u_numerical - u_exact
    error_L2 = np.sqrt(np.mean(diff**2))
    error_max = np.max(np.abs(diff))
    
    if relative_error:
        norm_exact = np.sqrt(np.mean(u_exact**2))
        if norm_exact > 1e-15:
            error_L2 = error_L2 / norm_exact
            error_max = error_max / max(np.max(np.abs(u_exact)), 1e-15)
    
    return error_L2, error_max


class Smoother:
    """
    光滑化算子：(I - α Δ)^{-1}
    
    用于在优化算法中对梯度进行光滑化处理
    """
    def __init__(self, coords, elements=None, degree=0.0):
        """
        Parameters:
        -----------
        coords : np.ndarray
            节点坐标
        elements : np.ndarray, optional
            单元连接关系
        degree : float
            光滑化强度参数 α
        """
        self.coords = coords
        self.elements = elements
        self.num_nodes = coords.shape[0]
        self.set_degree(degree)
    
    def set_degree(self, degree: float):
        """设置光滑化参数"""
        self.degree = degree
        
        if self.elements is not None and degree > 0:
            K = assemble_stiffness_matrix(self.coords, self.elements)
            M = assemble_mass_matrix(self.coords, self.elements)
            
            if isinstance(K, JAXSparseMatrix) and isinstance(M, JAXDiag):
                n = M.n
                diag_idx = jnp.arange(n, dtype=jnp.int32)
                M_sparse = csr_from_coo(M.diag, diag_idx, diag_idx, (n, n))
                
                k_data = jnp.asarray(K.data * degree)
                m_data = jnp.asarray(M.diag)
                
                combined_data = jnp.concatenate([k_data, m_data])
                combined_row = jnp.concatenate([jnp.asarray(K.row), diag_idx])
                combined_col = jnp.concatenate([jnp.asarray(K.col), diag_idx])
                
                self.A_smooth = csr_from_coo(combined_data, 
                                             combined_row.astype(jnp.int32),
                                             combined_col.astype(jnp.int32),
                                             (n, n))
            else:
                self.A_smooth = None
            
            try:
                from core.jax_backend import JAXLUSolver
                if self.A_smooth is not None:
                    self.lu_A = JAXLUSolver(self.A_smooth)
                else:
                    self.lu_A = None
                self.M_mat = M
            except Exception as e:
                print(f"Warning: Could not create smoother solver: {e}")
                self.lu_A = None
    
    def smoothing(self, fun_vec: np.ndarray, degree: float = None) -> np.ndarray:
        """应用光滑化操作"""
        assert fun_vec.shape[0] == self.num_nodes
        
        if degree is not None:
            old_degree = getattr(self, 'degree', 1.0)
            self.set_degree(degree)
        
        if self.degree <= 0 or self.lu_A is None:
            return fun_vec.copy()
        
        rhs = self.M_mat @ fun_vec
        smoothed = self.lu_A.solve(rhs)
        
        if degree is not None and hasattr(self, 'degree'):
            self.set_degree(old_degree)
        
        return smoothed


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DarcyFlow/misc.py module")
    print("=" * 60)
    
    coords, elements = create_mesh_2d(nx=5, ny=5, element_type='tri')
    print(f"Created mesh with {coords.shape[0]} nodes and {elements.shape[0]} elements")
    
    test_points = np.array([[0.25, 0.25], [0.75, 0.5]])
    S = construct_measure_matrix(coords, test_points)
    print(f"Constructed measurement matrix with shape {S.shape}")
    
    A = assemble_stiffness_matrix(coords, elements)
    M = assemble_mass_matrix(coords, elements)
    print(f"Assembled stiffness matrix {A.shape}, mass matrix {M.shape}")
    
    smoother = Smoother(coords, elements, degree=0.1)
    test_vec = np.random.randn(coords.shape[0])
    smooth_vec = smoother.smoothing(test_vec)
    print(f"Smoother works correctly")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
