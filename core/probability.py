#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概率论与先验分布模块

提供高斯先验、椭圆算子协方差等功能

作者: 基于 IPBayesML-FEniCSx09 重构，适配新架构
"""

import numpy as np
import jax.numpy as jnp

# JAX 后端替代 scipy.sparse
from core.jax_backend import (
    cg_jax, bicgstab_jax, spsolve_jax, JAXLUSolver,
    csr_from_coo, diags_jax, JAXSparseMatrix, JAXDiag
)
# 保留 scipy 导入名用于兼容性
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


def _assemble_stiffness_matrix_core(coords: np.ndarray, 
                                     elements: np.ndarray,
                                     diffusivity_field: np.ndarray = None):
    """
    [Core 内部] 组装刚度矩阵 A（用于椭圆先验算子）
    
    刚度矩阵 A_{ij} = ∫ κ ∇φ_i · ∇φ_j dx
    
    注意：此函数是 core 模块的内部实现，与具体 PDE 问题无关。
    DarcyFlow 问题有自己的实现在 DarcyFlow/misc.py 中。
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    elements = jnp.asarray(elements, dtype=jnp.int32)
    
    nnodes = coords.shape[0]
    nelems = elements.shape[0]
    
    if diffusivity_field is None:
        diffusivity_field = jnp.ones(nnodes, dtype=jnp.float64)
    else:
        diffusivity_field = jnp.asarray(diffusivity_field, dtype=jnp.float64)

    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
    c0, c1, c2 = coords[n0], coords[n1], coords[n2]

    area = 0.5 * jnp.abs(
        (c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) -
        (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1])
    )
    inv_2area = 1.0 / (2.0 * area + 1e-30)

    B = jnp.empty((nelems, 2, 3), dtype=jnp.float64)
    B = B.at[:, 0, 0].set((c1[:, 1] - c2[:, 1]) * inv_2area)
    B = B.at[:, 0, 1].set((c2[:, 1] - c0[:, 1]) * inv_2area)
    B = B.at[:, 0, 2].set((c0[:, 1] - c1[:, 1]) * inv_2area)
    B = B.at[:, 1, 0].set((c2[:, 0] - c1[:, 0]) * inv_2area)
    B = B.at[:, 1, 1].set((c0[:, 0] - c2[:, 0]) * inv_2area)
    B = B.at[:, 1, 2].set((c1[:, 0] - c0[:, 0]) * inv_2area)

    kappa_avg = (diffusivity_field[n0] + diffusivity_field[n1] + diffusivity_field[n2]) / 3.0

    BTB = jnp.einsum('eji,ejk->eik', B, B)
    K_local = kappa_avg[:, None, None] * BTB * area[:, None, None]

    row_idx = jnp.tile(elements[:, None], (1, 3)).ravel()
    col_idx = jnp.tile(elements[:, :, None], (1, 1, 3)).ravel()
    data = K_local.ravel()

    return csr_from_coo(data, row_idx.astype(jnp.int32), col_idx.astype(jnp.int32),
                        (nnodes, nnodes))


def _assemble_mass_matrix_core(coords: np.ndarray, elements: np.ndarray):
    """
    [Core 内部] 组装质量矩阵 M（用于椭圆先验算子）
    
    质量矩阵 M_{ij} = ∫ φ_i φ_j dx（集中对角质量矩阵）
    
    注意：此函数是 core 模块的内部实现，与具体 PDE 问题无关。
    """
    coords = jnp.asarray(coords, dtype=jnp.float64)
    elements = jnp.asarray(elements, dtype=jnp.int32)
    
    nnodes = coords.shape[0]

    n0, n1, n2 = elements[:, 0], elements[:, 1], elements[:, 2]
    c0, c1, c2 = coords[n0], coords[n1], coords[n2]

    area = 0.5 * jnp.abs(
        (c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) -
        (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1])
    )

    mass_contrib = area / 3.0

    diag_entries = jnp.zeros(nnodes, dtype=jnp.float64)
    diag_entries = diag_entries.at[n0].add(mass_contrib)
    diag_entries = diag_entries.at[n1].add(mass_contrib)
    diag_entries = diag_entries.at[n2].add(mass_contrib)

    return diags_jax(diag_entries)


def _create_simple_mesh_2d(nx: int = 10, ny: int = 10):
    """
    [Core 内部] 创建简单二维三角形网格（仅用于测试）
    
    注意：DarcyFlow 问题使用 DarcyFlow/misc.py 中的 create_mesh_2d。
    """
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    
    return coords, np.array(elements, dtype=np.int32)


class GaussianElliptic2:
    """
    高斯椭圆先验分布 N(mean, C)
    
    协方差算子的逆: C^{-1/2} = -∇(θ∇·) + a(x) Id
    
    这是一个常用的无限维高斯先验模型，
    通过有限元离散化后用于贝叶斯逆问题。
    
    参考文献:
    [1] Iglesias et al., "A computational framework for infinite-dimensional 
        Bayesian inverse problems part I", SIAM J. Sci. Comput., 2013
    """
    
    def __init__(self, coords: np.ndarray, elements: np.ndarray,
                 params: dict = None, boundary="Neumann", dtype=np.float64):
        """
        初始化椭圆高斯先验
        
        Parameters:
        -----------
        coords : np.ndarray
            节点坐标 (N, dim)
        elements : np.ndarray
            单元连接关系
        params : dict, optional
            参数字典，应包含:
            - 'theta': 函数 θ(x)，控制扩散系数
            - 'ax': 函数 a(x)，控制质量项
            - 'mean': 均值函数
            如果为None，使用默认值
        boundary : str
            边界条件类型 ('Neumann' 或 'Dirichlet')
        dtype : np.dtype
            数值类型
        """
        self.coords = coords
        self.elements = elements
        self._input_params = params
        self.boundary = boundary
        self.dtype = dtype
        self.num_dofs = coords.shape[0]
        
        # 设置默认参数
        if params is None:
            params = {
                "theta": lambda x: np.ones(len(x)) if len(x.shape) > 1 else 1.0,
                "ax": lambda x: np.ones(len(x)) if len(x.shape) > 1 else 1.0,
                "mean": lambda x: np.zeros(len(x)) if len(x.shape) > 1 else 0.0
            }
        assert len(params) == 3, "params must contain theta, ax, and mean"
        self.params = params
        
        # 初始化测量矩阵和算子
        self._init_measure()
        
        # 设置均值函数
        self.mean = self._evaluate_function(params["mean"])
        self.mean_vec = self.mean.copy()
        
        # 用于采样的临时变量
        self.sample_fun = np.zeros(self.num_dofs)
        
        # 辅助变量（用于高效计算梯度和 Hessian）
        # 注意：这些变量会被 JAX 后端操作覆盖为 JAX array（immutable）
        # 因此所有赋值必须用 = 替代 [:] =，或使用 jnp.array() 包装
        self.temp1 = np.zeros(self.num_dofs, dtype=np.float64)
        self.temp2 = np.zeros(self.num_dofs, dtype=np.float64)
        self.temp3 = np.zeros(self.num_dofs, dtype=np.float64)
        self.temp4 = np.zeros(self.num_dofs, dtype=np.float64)
        self._rdv = np.zeros(self.num_dofs)
        
        # 组装刚度矩阵 K 和质量矩阵 M
        self.K = self._assemble_K()
        self.M = _assemble_mass_matrix_core(coords, elements)
        
        # 标记是否已计算特征系统
        self.has_eigensystem = False
    
    def _evaluate_function(self, func) -> np.ndarray:
        """在节点上评估函数"""
        result = func(self.coords)
        if isinstance(result, (int, float)):
            result = np.full(self.num_dofs, result)
        return np.array(result, dtype=self.dtype)
    
    def _assemble_K(self) -> sps.csr_matrix:
        """
        组装刚度矩阵 K
        
        K_{ij} = ∫ θ(x) ∇φ_i · ∇φ_j dx + a(x) φ_i φ_j dx
        """
        theta_field = self._evaluate_function(self.params["theta"])
        ax_field = self._evaluate_function(self.params["ax"])
        
        # 扩散部分
        K_diffusion = _assemble_stiffness_matrix_core(self.coords, self.elements, theta_field)
        
        # 质量部分（使用 JAXDiag 的 scaled 方法）
        M_ax_base = _assemble_mass_matrix_core(self.coords, self.elements)
        
        # 对角矩阵缩放: M_ax[i,i] *= ax_field[i]
        if hasattr(M_ax_base, 'scaled'):
            M_ax_scaled = M_ax_base.scaled(ax_field)
        else:
            # fallback: 转 dense 处理
            M_dense = M_ax_base.toarray() if hasattr(M_ax_base, 'toarray') else np.array(M_ax_base)
            M_ax_scaled = M_dense * jnp.diag(ax_field)
        
        K = K_diffusion + M_ax_scaled
        return K
    
    def _init_measure(self):
        """初始化相关测量和求解器（使用 JAX 后端）"""
        try:
            self.solverK = JAXLUSolver(self._assemble_K())
        except Exception as e:
            print(f"Warning: Could not create K solver: {e}")
            self.solverK = None
        
        M = _assemble_mass_matrix_core(self.coords, self.elements)
        self._M = M  # 缓存质量矩阵
        try:
            self.solverM = JAXLUSolver(M)
        except Exception as e:
            print(f"Warning: Could not create M solver: {e}")
            self.solverM = None
        
        # 计算 M^{1/2}（简化为对角近似）
        if isinstance(M, JAXDiag):
            self.M_half_array = jnp.sqrt(jnp.abs(M.diag))
        else:
            self.M_half_array = jnp.sqrt(jnp.abs(M.diagonal()))
    
    def set_mean(self, point_values: np.ndarray):
        """
        设置均值场
        
        Parameters:
        -----------
        point_values : np.ndarray
            节点处的均值
        """
        assert len(point_values) == len(self.mean)
        self.mean[:] = point_values
        self.mean_vec = self.mean.copy()
    
    def generate_sample_zero_mean(self, num: int = 1) -> np.ndarray:
        """
        生成零均值样本
        
        从 N(0, C) 采样，其中 C = K^{-1} M
        
        通过 Cholesky 分解或平方根采样:
        z ~ N(0, I)
        sample = M^{1/2} K^{-1} z
        
        Parameters:
        -----------
        num : int
            采样数量
            
        Returns:
        --------
        samples : np.ndarray
            形状 (num, N) 或 (N,)
        """
        samples = np.zeros((num, self.num_dofs), dtype=self.dtype)
        
        for i in range(num):
            # 生成标准正态随机数
            self._rdv[:] = np.random.normal(0, 1, (self.num_dofs,))
            
            # 应用 M^{1/2}
            temp = self.M_half_array * self._rdv
            
            # 应用 K^{-1}（JAX CG）
            if self.solverK is not None:
                samples[i, :] = self.solverK.solve(temp)
            else:
                samples[i, :] = spsolve_jax(self.K, temp, method='cg')
        
        return samples.squeeze()
    
    def generate_sample(self, num: int = 1) -> np.ndarray:
        """
        生成带均值的高斯样本
        
        Returns:
        --------
        samples : np.ndarray
            从 N(mean, C) 采样的样本
        """
        rv = self.generate_sample_zero_mean(num=num)
        if num > 1:
            assert rv.shape[1] == self.mean.shape[0]
        else:
            assert rv.shape[0] == self.mean.shape[0]
        
        val = self.mean + rv
        return val.squeeze()
    
    def log_density(self, u: np.ndarray) -> float:
        """
        计算对数密度 log π(u)
        
        对于高斯分布: log π(u) = -0.5 (u-mean)^T C^{-1} (u-mean) + const
        
        Parameters:
        -----------
        u : np.ndarray
            
        Returns:
        --------
        log_pdf : float
        """
        u = np.asarray(u).flatten()
        diff = u - self.mean_vec
        
        # 计算 (u-mean)^T K M^{-1} K (u-mean)
        temp = self.K @ diff
        if self.solverM is not None:
            temp = self.solverM.solve(temp)
        else:
            temp = spsolve_jax(self.M,  temp, method="cg")
        temp = self.K @ temp
        
        return -0.5 * float(diff @ temp)
    
    def grad_log_density(self, u: np.ndarray) -> np.ndarray:
        """
        计算对数密度的梯度 ∇_u log π(u)
        
        Returns:
        --------
        grad : np.ndarray
        """
        # 对于先验：∇ log π_prior = -C^{-1}(u - mean)
        return -self.eval_grad(u)
    
    def eval_grad(self, u: np.ndarray) -> np.ndarray:
        """
        计算先验损失的梯度
        
        ∇ L_prior(u) = M^{-1} K M^{-1} K (u - mean)
        
        Parameters:
        -----------
        u : np.ndarray
            参数向量
            
        Returns:
        --------
        grad : np.ndarray
            梯度向量
        """
        assert u.ndim == 1
        assert self.num_dofs == u.shape[0]
        
        # temp1 = u - mean
        temp1 = u - self.mean_vec
        
        # temp2 = K @ temp1
        temp2 = self.K @ temp1
        
        # temp1 = M^{-1} @ temp2
        if self.solverM is not None:
            temp1 = self.solverM.solve(temp2)
        else:
            temp1 = spsolve_jax(self.M,  temp2, method="cg")
        
        # temp2 = K^T @ temp1 (= K @ temp1 因为K对称)
        temp2 = self.K @ temp1
        
        # temp1 = M^{-1} @ temp2
        if self.solverM is not None:
            temp1 = self.solverM.solve(temp2)
        else:
            temp1 = spsolve_jax(self.M,  temp2, method="cg")
        
        return temp1
    
    def eval_hessian(self, u: np.ndarray) -> np.ndarray:
        """
        计算先验损失的 Hessian 向量乘法
        
        H @ u = M^{-1} K M^{-1} K u
        
        注意：对于高斯先验，Hessian 与 u 无关（常数矩阵）
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            Hessian 应用于 u 的结果
        """
        assert u.ndim == 1
        assert self.num_dofs == u.shape[0]
        
        # 复用 eval_grad 的逻辑（因为 Hessian 是常数）
        return self.eval_grad(u)
    
    def eval_CM_inner(self, u, v=None):
        """
        计算 (C^{-1/2}u, C^{-1/2}v)_M = (u-mean)^T K M^{-1} K (v-mean)
        
        这是协方差内积的加权版本。
        
        Parameters:
        -----------
        u : np.ndarray
        v : np.ndarray, optional
            如果为None则使用u
            
        Returns:
        --------
        val : float
            内积值
        """
        if v is None:
            v = np.array(u)
        assert u.ndim == 1 and v.ndim == 1
        assert self.num_dofs == u.shape[0] == v.shape[0]
        
        # temp3 = K @ (v - mean)
        self.temp3[:] = v - self.mean_vec
        temp3 = self.K @ self.temp3
        
        # temp4 = M^{-1} @ temp3
        if self.solverM is not None:
            self.temp4 = self.solverM.solve(temp3)
        else:
            self.temp4 = spsolve_jax(self.M,  temp3, method="cg")
        
        # temp3 = K^T @ temp4
        temp3 = self.K @ self.temp4
        
        # 内积: (u-mean) . temp3
        val = np.dot(u - self.mean_vec, temp3)
        
        return val
    
    def eval_Cinv(self, u: np.ndarray) -> np.ndarray:
        """
        计算 C^{-1} u = M^{-1} K M^{-1} K u
        
        这是精度矩阵（协方差逆）作用于向量
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量，可以是1D或2D
            
        Returns:
        --------
        result : np.ndarray
            结果向量
        """
        u = u.squeeze()
        assert u.ndim in [1, 2]
        assert u.shape[0] == self.num_dofs
        
        if u.ndim == 2:
            num_ = u.shape[1]
        else:
            num_ = 1
        
        if u.ndim == 2:
            val = np.zeros_like(u)
        else:
            val = np.zeros(self.num_dofs)
        
        for idx in range(num_):
            if num_ > 1:
                self.temp1 = np.asarray(u[:, idx], dtype=np.float64)
            else:
                self.temp1 = np.asarray(u, dtype=np.float64)
            
            # K @ temp1
            self.temp2 = self.K @ self.temp1
            
            # M^{-1} @ temp2
            if self.solverM is not None:
                self.temp1 = self.solverM.solve(self.temp2)
            else:
                self.temp1 = spsolve_jax(self.M,  self.temp2, method="cg")
            
            # K @ temp1
            self.temp2 = self.K @ self.temp1
            
            # M^{-1} @ temp2
            if self.solverM is not None:
                if num_ > 1:
                    val[:, idx] = self.solverM.solve(self.temp2)
                else:
                    val = self.solverM.solve(self.temp2)
            else:
                if num_ > 1:
                    val[:, idx] = spsolve_jax(self.M,  self.temp2, method="cg")
                else:
                    val = spsolve_jax(self.M,  self.temp2, method="cg")
        
        return val.squeeze()
    
    def eval_C(self, u: np.ndarray) -> np.ndarray:
        """
        计算 C u = K^{-1} M K^{-1} M u
        
        这是协方差算子作用于向量
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            结果向量
        """
        u = u.squeeze()
        assert u.ndim in [1, 2]
        assert u.shape[0] == self.num_dofs
        
        if u.ndim == 2:
            num_ = u.shape[1]
        else:
            num_ = 1
            
        if u.ndim == 2:
            val = np.zeros_like(u)
        else:
            val = np.zeros(self.num_dofs)
        
        for idx in range(num_):
            if num_ > 1:
                self.temp1 = np.asarray(u[:, idx], dtype=np.float64)
            else:
                self.temp1 = np.asarray(u, dtype=np.float64)
            
            # M @ temp1
            self.temp2 = self.M @ self.temp1
            
            # K^{-1} @ temp2
            if self.solverK is not None:
                self.temp1 = self.solverK.solve(self.temp2)
            else:
                self.temp1 = spsolve_jax(self.K,  self.temp2, method="cg")
            
            # M @ temp1
            self.temp2 = self.M @ self.temp1
            
            # K^{-1} @ temp2
            if self.solverK is not None:
                if num_ > 1:
                    val[:, idx] = self.solverK.solve(self.temp2)
                else:
                    val = self.solverK.solve(self.temp2)
            else:
                if num_ > 1:
                    val[:, idx] = spsolve_jax(self.K,  self.temp2, method="cg")
                else:
                    val = spsolve_jax(self.K,  self.temp2, method="cg")
        
        return val.squeeze()
    
    def eval_sqrtC(self, u: np.ndarray) -> np.ndarray:
        """
        计算 C^{1/2} u = K^{-1} M u
        
        协方差平方根算子
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            结果向量
        """
        u = u.squeeze()
        assert u.ndim in [1, 2]
        assert u.shape[0] == self.num_dofs
        
        if u.ndim == 2:
            num_ = u.shape[1]
        else:
            num_ = 1
            
        if u.ndim == 2:
            val = np.zeros_like(u)
        else:
            val = np.zeros(self.num_dofs)
        
        for idx in range(num_):
            if num_ > 1:
                self.temp1 = np.asarray(u[:, idx], dtype=np.float64)
            else:
                self.temp1 = np.asarray(u, dtype=np.float64)
            
            # M @ temp1
            self.temp2 = self.M @ self.temp1
            
            # K^{-1} @ temp2
            if self.solverK is not None:
                if num_ > 1:
                    val[:, idx] = self.solverK.solve(self.temp2)
                else:
                    val = self.solverK.solve(self.temp2)
            else:
                if num_ > 1:
                    val[:, idx] = spsolve_jax(self.K,  self.temp2, method="cg")
                else:
                    val = spsolve_jax(self.K,  self.temp2, method="cg")
        
        return val.squeeze()
    
    def eval_sqrtCinv(self, u: np.ndarray) -> np.ndarray:
        """
        计算 C^{-1/2} u = M^{-1} K u
        
        协方差逆平方根算子
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            结果向量
        """
        u = u.squeeze()
        assert u.ndim in [1, 2]
        assert u.shape[0] == self.num_dofs
        
        if u.ndim == 2:
            num_ = u.shape[1]
        else:
            num_ = 1
            
        if u.ndim == 2:
            val = np.zeros_like(u)
        else:
            val = np.zeros(self.num_dofs)
        
        for idx in range(num_):
            if num_ > 1:
                self.temp1 = np.asarray(u[:, idx], dtype=np.float64)
            else:
                self.temp1 = np.asarray(u, dtype=np.float64)
            
            # K @ temp1
            self.temp2 = self.K @ self.temp1
            
            # M^{-1} @ temp2
            if self.solverM is not None:
                if num_ > 1:
                    val[:, idx] = self.solverM.solve(self.temp2)
                else:
                    val = self.solverM.solve(self.temp2)
            else:
                if num_ > 1:
                    val[:, idx] = spsolve_jax(self.M,  self.temp2, method="cg")
                else:
                    val = spsolve_jax(self.M,  self.temp2, method="cg")
        
        return val.squeeze()
    
    def precondition(self, u: np.ndarray) -> np.ndarray:
        """
        预条件子操作: K^{-1} M u
        
        这通常是近似协方差算子的应用
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            预条件后的向量
        """
        assert u.ndim == 1
        assert u.shape[0] == self.num_dofs
        
        self.temp1 = np.asarray(u, dtype=np.float64)  # 替代 self.temp1[:] = u，兼容 JAX immutable array
        
        # K^{-1} @ temp1
        if self.solverK is not None:
            self.temp2 = self.solverK.solve(self.temp1)
        else:
            self.temp2 = spsolve_jax(self.K,  self.temp1, method="cg")
        
        # M @ temp2
        result = self.M @ self.temp2
        
        # K^{-1} @ result
        if self.solverK is not None:
            self.temp1 = self.solverK.solve(result)
        else:
            self.temp1 = spsolve_jax(self.K,  result, method="cg")
        
        return np.array(self.temp1)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gaussian Elliptic Prior")
    print("=" * 60)
    
    # 创建测试网格
    nx, ny = 8, 8
    # 使用简单的网格生成（不依赖 DarcyFlow 模块）
    coords, elements = _create_simple_mesh_2d(nx=nx, ny=ny)
    print(f"✓ Created mesh with {coords.shape[0]} nodes")
    
    # 创建先验分布
    prior = GaussianElliptic2(coords, elements)
    print(f"✓ Initialized GaussianElliptic2 prior")
    
    # 测试采样
    samples = prior.generate_sample(num=3)
    print(f"✓ Generated {samples.shape[0]} samples, shape: {samples.shape}")
    
    # 测试梯度计算
    u_test = np.random.randn(coords.shape[0])
    grad = prior.eval_grad(u_test)
    print(f"✓ Computed gradient, shape: {grad.shape}, norm: {np.linalg.norm(grad):.6f}")
    
    # 测试 Hessian 计算
    hess_u = prior.eval_hessian(u_test)
    print(f"✓ Computed Hessian-vector product, shape: {hess_u.shape}")
    
    # 测试内积
    inner_val = prior.eval_CM_inner(u_test)
    print(f"✓ Computed CM-inner product: {inner_val:.6f}")
    
    # 测试协方差操作
    C_inv_u = prior.eval_Cinv(u_test)
    C_u = prior.eval_C(u_test)
    print(f"✓ Computed C^{-1}@u and C@u operations")
    
    print("\n" + "=" * 60)
    print("All probability tests passed!")
    print("=" * 60)
