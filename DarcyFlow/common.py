#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 tatva (JAX-FEM) 的 Darcy 流方程求解器

稳态 Darcy 流方程:
    -∇·(e^{u}∇w) = f,  in Ω,
    w = wD,          on ∂Ω

其中:
    - u: 对数渗透率场（待反演的参数）
    - w: 压力/水头（状态变量）
    - f: 源项
    - wD: 边界条件

本模块提供：
1. EquSolverDarcyFlow: Darcy 流方程求解器
2. ModelDarcyFlow: Darcy 流的贝叶斯逆问题模型

参考文献:
[1] Jia, Li, Meng (2022). "Stein variational gradient descent on infinite-dimensional 
    space and applications to statistical inverse problems", SIAM J. Numer. Anal.
[2] Ghattas & Willcox (2021). "Learning physics-based models from data", Acta Numerica.

作者: 基于原 IPBayesML-FEniCSx09 重构，使用 tatva 库和 scipy sparse
"""

import numpy as np
import jax.numpy as jnp

# ====== JAX 后端替代 scipy.sparse ======
from core.jax_backend import (
    JAXSparseMatrix, JAXDiag,
    cg_jax, bicgstab_jax, spsolve_jax, JAXLUSolver
)

# 保留 scipy 导入名用于类型标注兼容（不再用于计算）
import scipy.sparse as sps  # 兼容性
import scipy.sparse.linalg as spsl  # 兼容性

from typing import Optional, Tuple, Dict, Any, Callable
import sys
import os

# 添加父目录到路径
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from core.model import EquSolverBase, ModelBase
from DarcyFlow.misc import (
    construct_measure_matrix, Smoother, trans2scipy,
    assemble_stiffness_matrix, assemble_mass_matrix, create_mesh_2d
)


__all__ = ['EquSolverDarcyFlow', 'ModelDarcyFlow']


class EquSolverDarcyFlow(EquSolverBase):
    """
    Darcy 流方程求解器
    
    使用有限元方法求解稳态 Darcy 方程，并提供正问题、伴随问题和增量问题的求解。
    
    支持的功能：
    - 正问题求解: A(u)w = b
    - 伴随问题求解: A(u)^T λ = S^T r
    - 增量前向问题: A(u) δw = f_inc(δu)
    - 增量伴随问题: A(u)^T δλ = g_inc(δu)
    """
    
    def __init__(self, coords: np.ndarray, 
                 elements: np.ndarray,
                 u: np.ndarray = None,
                 f: np.ndarray = None, 
                 wD: np.ndarray = None,
                 degree: int = 1):
        """
        初始化 Darcy 流求解器
        
        Parameters:
        -----------
        coords : np.ndarray
            节点坐标 (N, dim)，dim=2 或 3
        elements : np.ndarray
            单元连接关系 (M, nodes_per_element)
        u : np.ndarray, optional
            对数渗透率场的初始值 (N,)
        f : np.ndarray, optional
            源项向量 (N,)，默认为均匀源项 1
        wD : np.ndarray, optional
            Dirichlet 边界条件值 (N,)，默认为 0
        degree : int
            有限元多项式阶数（保留参数，当前仅支持线性单元）
        """
        super().__init__()
        
        self.coords = np.array(coords, dtype=np.float64)  # 确保float64避免JAX截断
        self.elements = np.array(elements, dtype=np.int32)
        self.degree = degree
        
        self.num_dofs = coords.shape[0]
        
        # ====== 性能优化：预计算和缓存不变量（必须在 set_params 之前）======
        # 边界节点集合仅依赖几何
        self._boundary_nodes_cached = self._identify_boundary_nodes()
        self._boundary_nodes_array = np.array(self._boundary_nodes_cached)
        
        # 设置默认参数并初始化
        self.set_params(u, f, wD)
    
    def set_params(self, u=None, f=None, wD=None):
        """
        设置或更新模型参数
        
        Parameters:
        -----------
        u : np.ndarray, optional
            对数渗透率场
        f : np.ndarray, optional  
            源项
        wD : np.ndarray, optional
            边界条件值
        """
        f_changed = (f is not None) if hasattr(self, 'f') else True
        wD_changed = (wD is not None) if hasattr(self, 'wD') else True

        # 默认参数
        if f is None:
            f = getattr(self, 'f', np.ones(self.num_dofs))
        if u is None:
            u = getattr(self, 'u', np.ones(self.num_dofs))
        if wD is None:
            wD = getattr(self, 'wD', np.zeros(self.num_dofs))

        # 存储参数
        self.f = np.array(f, dtype=np.float64)
        self.u = np.array(u, dtype=np.float64)
        self.wD = np.array(wD, dtype=np.float64)

        # 分配解向量内存（使用 JAX 数组以支持 GPU 加速）
        self.sol_forward = jnp.zeros(self.num_dofs, dtype=jnp.float64)
        self.sol_adjoint = jnp.zeros(self.num_dofs, dtype=jnp.float64)
        self.sol_inc_forward = jnp.zeros(self.num_dofs, dtype=jnp.float64)
        self.sol_inc_adjoint = jnp.zeros(self.num_dofs, dtype=jnp.float64)

        # ====== 性能优化：仅首次或 f/wD 变化时组装质量矩阵 M ======
        if not hasattr(self, '_M') or f_changed:
            from DarcyFlow.misc import assemble_mass_matrix
            self._M = assemble_mass_matrix(self.coords, self.elements)

        # 初始化方程系统（A 依赖 u，M 已缓存）
        self._init_equation()
        self._init_solver()
    
    def _init_equation(self):
        """初始化方程系统（组装矩阵）— 使用缓存的 M"""
        from DarcyFlow.misc import assemble_stiffness_matrix

        # 计算扩散系数 κ = exp(u)，带数值截断防止溢出
        u_clipped = np.clip(self.u, -20, 20)
        kappa = np.exp(u_clipped)

        # 组装刚度矩阵 A = ∫ exp(u) ∇φ_i · ∇φ_j dx（仅此项依赖 u）
        self.A = assemble_stiffness_matrix(self.coords, self.elements, kappa)

        # 应用边界条件（使用缓存的边界节点）
        self._apply_boundary_conditions_to_A()

        # 使用缓存的质量矩阵构建右端项 b
        self.b = self._M @ self.f

        # 应用边界条件到右端项
        self._apply_boundary_conditions_to_b()
    
    def _apply_boundary_conditions_to_A(self):
        """对刚度矩阵应用 Dirichlet 边界条件（JAX 惩罚法）"""
        bn = self._boundary_nodes_array
        if len(bn) == 0:
            return
        # 使用 JAX 稀疏矩阵的边界条件方法（返回新矩阵）
        self.A = self.A.apply_penalty_boundary(bn, penalty=1e20)
    
    def _apply_boundary_conditions_to_b(self):
        """对右端项应用 Dirichlet 边界条件"""
        bn = self._boundary_nodes_array
        if len(bn) == 0:
            return
        b_arr = jnp.asarray(self.b)
        wD_arr = jnp.asarray(self.wD)
        b_arr = b_arr.at[bn].set(wD_arr[bn] * 1e20)
        self.b = b_arr  # 保持 JAX 数组，不拷贝到 CPU
    
    def _identify_boundary_nodes(self) -> list:
        """
        识别边界节点（简化实现）
        
        Returns:
        --------
        boundary_nodes : list
            边界节点索引列表
        """
        # 简化：找到坐标范围边界的节点
        tol = 1e-10
        xmin, xmax = self.coords[:, 0].min(), self.coords[:, 0].max()
        ymin, ymax = self.coords[:, 1].min(), self.coords[:, 1].max()
        
        boundary_nodes = []
        for i, coord in enumerate(self.coords):
            if (abs(coord[0] - xmin) < tol or abs(coord[0] - xmax) < tol or
                abs(coord[1] - ymin) < tol or abs(coord[1] - ymax) < tol):
                boundary_nodes.append(i)
        
        return boundary_nodes
    
    def _init_solver(self):
        """初始化线性求解器（使用 JAX CG 求解器，支持 GPU）"""
        try:
            self.lu_solver = JAXLUSolver(self.A)
        except Exception as e:
            print(f"Warning: Could not create JAX solver: {e}")
            self.lu_solver = None
    
    def _update_u(self, u: np.ndarray):
        """
        更新渗透率场 u 并重新组装刚度矩阵
        
        Parameters:
        -----------
        u : np.ndarray
            新的对数渗透率场
        """
        assert self.u.shape[0] == u.shape[0]
        # 数值截断防止 exp(u) 溢出
        self.u = np.clip(np.array(u), -20, 20)
        self._init_equation()
        self._init_solver()
    
    def _init_measurement_matrix(self, coordinates: np.ndarray):
        """
        初始化测量矩阵
        
        Parameters:
        -----------
        coordinates : np.ndarray
            观测点坐标 (M, dim)
        """
        assert coordinates.shape[1] == self.coords.shape[1], \
            f"Observation points dimension {coordinates.shape[1]} != mesh dimension {self.coords.shape[1]}"
        self.S = construct_measure_matrix(self.coords, coordinates)
    
    def get_data(self, w) -> np.ndarray:
        """
        从状态变量提取观测数据

        d = S @ w
        """
        assert w.shape[0] == self.S.shape[1], \
            f"Shape mismatch: w {w.shape[0]}, S columns {self.S.shape[1]}"
        result = self.S @ w
        # 仅在最终输出到用户时转为 numpy（如需绘图等）
        return result  # 保持 JAX 数组，调用方按需转 numpy
    
    def forward_solve(self, u=None):
        """
        求解正问题: A(u) w = b

        Parameters:
        -----------
        u : np.ndarray, optional
            如果提供，则更新渗透率场
            
        Returns:
        --------
        w : jnp.ndarray
            正问题的解（压力/水头）— 保持 JAX 数组以支持 GPU
        """
        if u is not None:
            self._update_u(u)
        
        # 求解线性系统（JAX CG 求解器，GPU 加速）
        if self.lu_solver is not None:
            self.sol_forward = self.lu_solver.solve(self.b)
        else:
            self.sol_forward = spsolve_jax(self.A, self.b, method='cg')
        
        return self.sol_forward  # 返回 JAX 数组，不拷贝到 CPU
    
    def adjoint_solve(self, vec: np.ndarray, u=None):
        """
        求解伴随问题: A(u)^T λ = S^T vec

        Parameters:
        -----------
        vec : np.ndarray
            残差向量 (d_obs - G(u))
        u : np.ndarray, optional
            如果提供，则更新渗透率场
            
        Returns:
        --------
        lam : jnp.ndarray
            伴随变量 — 保持 JAX 数组以支持 GPU
        """
        if u is not None:
            self._update_u(u)
        
        assert self.S.shape[0] == vec.shape[0], \
            f"Measurement matrix rows {self.S.shape[0]} != vec length {vec.shape[0]}"
        
        # 右端项: F_s = -S^T @ vec
        F_s = -(self.S.T @ vec).squeeze()
        
        # 求解伴随问题（使用 JAX BiCGSTAB 或转置 CG）
        if self.lu_solver is not None:
            self.sol_adjoint = self.lu_solver.solve(F_s, trans='T')
        else:
            # A.T 可能非 SPD，使用 BiCGSTAB
            self.sol_adjoint = bicgstab_jax(self.A.T, F_s, tol=1e-8)[0]
        
        return self.sol_adjoint  # 返回 JAX 数组，不拷贝到 CPU
    
    def inc_forward_solve(self, u_hat: np.ndarray, sol_forward=None):
        """
        求解增量正问题: A(u) δw = -δA · w

        其中 δA 是由 δu 引起的刚度矩阵变化
        
        Parameters:
        -----------
        u_hat : np.ndarray
            渗透率场的增量 δu
        sol_forward : np.ndarray, optional
            正问题的解（如果已知可避免重复计算）
            
        Returns:
        --------
        dw : jnp.ndarray
            状态变量的增量 — 保持 JAX 数组以支持 GPU
        """
        if sol_forward is not None:
            assert sol_forward.shape[0] == self.num_dofs
            self.sol_forward = jnp.asarray(sol_forward, dtype=jnp.float64)
        
        # 计算增量刚度矩阵 δA
        # δA_{ij} = ∫ exp(u) δu (∇φ_i · ∇φ_j) dx
        u_safe = np.clip(self.u, -20, 20)
        delta_kappa = np.exp(u_safe) * np.clip(u_hat, -10, 10)
        delta_A = assemble_stiffness_matrix(self.coords, self.elements, delta_kappa)
        
        # 右端项: b_inc = -δA @ w
        b_inc = -(delta_A @ self.sol_forward)
        
        # 求解增量方程（JAX CG）
        if self.lu_solver is not None:
            self.sol_inc_forward = self.lu_solver.solve(b_inc)
        else:
            self.sol_inc_forward = spsolve_jax(self.A, b_inc, method='cg')
        
        return self.sol_inc_forward  # 返回 JAX 数组，不拷贝到 CPU
    
    def inc_adjoint_solve(self, vec: np.ndarray,
                          u_hat: np.ndarray,
                          sol_adjoint=None,
                          simple=False):
        """
        求解增量伴随问题

        Parameters:
        -----------
        vec : np.ndarray
            残差向量
        u_hat : np.ndarray
            渗透率的增量
        sol_adjoint : np.ndarray, optional
            伴随问题的解
        simple : bool
            是否使用简化模式（忽略梯度项）
            
        Returns:
        --------
        dlam : jnp.ndarray
            伴随变量的增量 — 保持 JAX 数组以支持 GPU
        """
        if sol_adjoint is not None:
            assert sol_adjoint.shape[0] == self.num_dofs
            self.sol_adjoint = jnp.asarray(sol_adjoint, dtype=jnp.float64)
        
        # 基础右端项
        Fs = -(self.S.T @ vec).squeeze()
        
        if simple == False:
            # 完整形式：包含梯度相关项
            # 计算 δA^T @ λ
            delta_kappa = np.exp(self.u) * u_hat
            delta_A = assemble_stiffness_matrix(self.coords, self.elements, delta_kappa)
            
            rhs = -delta_A.T @ self.sol_adjoint + Fs
        elif simple == True:
            # 简化形式：忽略梯度修正
            rhs = Fs
        
        # 求解增量伴随问题（JAX BiCGSTAB）
        if self.lu_solver is not None:
            self.sol_inc_adjoint = self.lu_solver.solve(rhs, trans='T')
        else:
            self.sol_inc_adjoint = bicgstab_jax(self.A.T, rhs, tol=1e-8)[0]
        
        return self.sol_inc_adjoint  # 返回 JAX 数组，不拷贝到 CPU


class ModelDarcyFlow(ModelBase):
    """
    Darcy 流贝叶斯逆问题模型
    
    整合先验分布、Darcy 流求解器和噪声模型，提供：
    - 完整的损失函数及其梯度和 Hessian
    - 通过伴随方法高效计算梯度
    - 支持光滑化操作
    """
    
    def __init__(self, prior, equ_solver: EquSolverDarcyFlow, 
                 noise, data: Dict[str, Any],
                 smoother: Smoother = None):
        """
        初始化 Darcy 流逆问题模型
        
        Parameters:
        -----------
        prior : object
            先验分布对象（如 GaussianElliptic）
        equ_solver : EquSolverDarcyFlow
            Darcy 流方程求解器
        noise : object
            观测噪声模型（如 GaussianIID）
        data : dict
            观测数据字典 {'coordinates': ..., 'data': ...}
        smoother : Smoother, optional
            光滑化算子
        """
        super().__init__(prior, equ_solver, noise, data)
        
        # 验证接口
        assert hasattr(self.noise, 'eval_CM_inner'), \
            "noise must have eval_CM_inner method"
        assert hasattr(self.prior, 'eval_grad'), \
            "prior must have eval_grad method"
        assert hasattr(self.prior, 'eval_hessian'), \
            "prior must have eval_hessian method"
        assert hasattr(self.prior, 'precondition'), \
            "prior must have precondition method"
        assert hasattr(self.equ_solver, 'forward_solve'), \
            "equ_solver must have forward_solve method"
        assert hasattr(self.equ_solver, 'adjoint_solve'), \
            "equ_solver must have adjoint_solve method"
        assert hasattr(self.equ_solver, 'inc_forward_solve'), \
            "equ_solver must have inc_forward_solve method"
        assert hasattr(self.equ_solver, 'inc_adjoint_solve'), \
            "equ_solver must have inc_adjoint_solve method"
        assert hasattr(self.equ_solver, 'S'), \
            "equ_solver must have measurement matrix S"
        
        # 存储辅助变量
        self.p = np.zeros(self.num_dofs)      # 用于存储正向解
        self.q = np.zeros(self.num_dofs)      # 用于存储伴随解
        self.pp = np.zeros(self.num_dofs)     # 用于存储增量正向解
        self.qq = np.zeros(self.num_dofs)     # 用于存储增量伴随解
        
        self.function_space = None  # 保留兼容性
        self.name = "ModelDarcyFlow"
        
        # 初始化质量矩阵求解器
        self._init_solver_M()
        
        # 设置光滑化算子
        if smoother is None:
            smoother = Smoother(
                self.equ_solver.coords if hasattr(self.equ_solver, 'coords') else None,
                self.equ_solver.elements if hasattr(self.equ_solver, 'elements') else None,
                degree=0.0
            )
        assert hasattr(smoother, "smoothing"), "smoother must have smoothing method"
        self.smoother = smoother
    
    def _init_measurement_matrix(self, coordinates: np.ndarray):
        """初始化测量矩阵"""
        assert coordinates.shape[1] >= 2
        self.equ_solver._init_measurement_matrix(coordinates)
        self.S = self.equ_solver.S
    
    def update_param(self, u: np.ndarray, update_sol: bool = True):
        """
        更新参数并可选地重新求解正问题
        
        Parameters:
        -----------
        u : np.ndarray
            新的参数场
        update_sol : bool
            是否同时更新正向解
        """
        self.equ_solver._update_u(u)
        if update_sol:
            self.equ_solver.forward_solve()
    
    def _init_solver_M(self):
        """初始化质量矩阵的 JAX 求解器"""
        try:
            self.solverM = JAXLUSolver(self.M)
        except Exception as e:
            print(f"Warning: Could not create mass matrix solver: {e}")
            self.solverM = None
    
    def loss_res(self, u=None) -> float:
        """
        计算数据残差损失
        
        L_res(u) = 0.5 * ||G(u) - d_obs||^2_{Γ^{-1}}
               = 0.5 * (G(u) - d_obs)^T Γ^{-1} (G(u) - d_obs)
        """
        if u is not None:
            self.equ_solver.forward_solve(u)
        
        d_pred = self.get_data(self.equ_solver.sol_forward)
        assert d_pred.shape[0] == self.data.shape[0], \
            f"Predicted data shape {d_pred.shape} != observed data shape {self.data.shape}"
        
        res = d_pred - self.data
        val = 0.5 * self.noise.eval_CM_inner(res)
        return val
    
    def loss_prior(self, u=None) -> float:
        """计算先验损失"""
        if u is not None:
            assert u.shape[0] == self.num_dofs
            val = 0.5 * self.prior.eval_CM_inner(u)
        else:
            val = 0.5 * self.prior.eval_CM_inner(self.equ_solver.u)
        return val
    
    def eval_grad_res(self, u: np.ndarray) -> np.ndarray:
        """
        使用伴随方法计算数据残差损失的梯度
        
        ∇_u L_res = ∫ q exp(u) ∇w · ∇(·) dx
        
        其中:
        - w 是正向解
        - q 是伴随解
        """
        self.equ_solver.forward_solve(u)
        
        # 计算残差和归一化
        vec = self.get_data(self.equ_solver.sol_forward) - self.noise.mean - self.data
        vec = vec / (self.noise.std_dev**2)
        
        # 求解伴随问题得到 q
        self.equ_solver.adjoint_solve(vec.squeeze())
        
        # 存储中间结果用于 Hessian 计算
        self.p = self.equ_solver.sol_forward.copy()
        self.q = self.equ_solver.sol_adjoint.copy()
        
        # 计算梯度: grad_i = ∫ exp(u) ∇q · ∇w φ_i dx
        # 简化计算: 使用数值近似
        integrand = np.exp(self.equ_solver.u) * self._compute_gradient_product(self.q, self.p)
        
        # 应用质量矩阵的逆
        if self.solverM is not None:
            L = integrand
            val = self.solverM.solve(L)
        else:
            val = integrand
        
        return jnp.array(val) if not isinstance(val, np.ndarray) else val
    
    def eval_grad_total(self, u: np.ndarray) -> np.ndarray:
        """
        计算总损失的梯度（先验 + 似然）
        
        ∇_u J = C_0^{-1}(u - mean) + ∇_u L_res(u)
        """
        # 先验梯度
        grad_prior = self.prior.eval_grad(u)
        
        # 似然梯度
        grad_likelihood = self.eval_grad_res(u)
        
        return grad_prior + grad_likelihood
    
    def compute_log_joint(self, u: np.ndarray) -> float:
        """
        计算对数联合概率 log π(u, d)
        
        log π(u, d) = log π_prior(u) + log L(d|u)
        """
        log_prior = self.prior.log_density(u)
        loss_res = self.loss_res(u)
        log_likelihood = -loss_res
        
        return log_prior + log_likelihood
    
    def _compute_gradient_product(self, v1, v2):
        """
        计算两个场变量的梯度点积: ∇v1 · ∇v2 在每个节点处的近似值
        
        Parameters:
        -----------
        v1, v2 : array-like
            节点值（可以是 numpy 或 JAX 数组）
            
        Returns:
        --------
        result : jnp.ndarray
            梯度点积在每个节点的值
        """
        coords = self.equ_solver.coords
        elements = self.equ_solver.elements
        result = np.zeros(self.num_dofs)

        for e in range(elements.shape[0]):
            n0, n1, n2 = elements[e]
            c0, c1, c2 = coords[n0], coords[n1], coords[n2]
            area = 0.5 * abs((c1[0]-c0[0])*(c2[1]-c0[1]) - (c2[0]-c0[0])*(c1[1]-c0[1]))
            if area < 1e-15:
                continue
            inv2A = 1.0 / (2.0 * area)
            # 正确的FEM梯度（线性三角形，常数梯度）
            dN = inv2A * np.array([
                [c1[1]-c2[1], c2[1]-c0[1], c0[1]-c1[1]],
                [c2[0]-c1[0], c0[0]-c2[0], c1[0]-c0[0]]
            ])
            gv1 = dN @ np.array([v1[n0], v1[n1], v1[n2]])  # (2,)
            gv2 = dN @ np.array([v2[n0], v2[n1], v2[n2]])  # (2,)
            dot = float(gv1 @ gv2) * area  # ∫ ∇v1·∇v2 dx on this element
            # 均匀分配到三个节点（因为 ∫φ_k dx = area/3）
            for nk in [n0, n1, n2]:
                result[nk] += dot / 3.0
        return result
    
    # def _compute_gradient_product(self, v1, v2):
    #     """
    #     计算两个场变量的梯度点积: ∇v1 · ∇v2 在每个节点处的近似值
        
    #     Parameters:
    #     -----------
    #     v1, v2 : array-like
    #         节点值（可以是 numpy 或 JAX 数组）
            
    #     Returns:
    #     --------
    #     result : jnp.ndarray
    #         梯度点积在每个节点的值
    #     """
    #     # 使用 JAX 向量化运算替代 Python 循环，支持 GPU 加速
    #     coords = self.equ_solver.coords  # numpy 数组（几何数据不变）
    #     nelems = self.equ_solver.elements.shape[0]
        
    #     v1 = jnp.asarray(v1, dtype=jnp.float64)
    #     v2 = jnp.asarray(v2, dtype=jnp.float64)
    #     result = jnp.zeros(self.num_dofs, dtype=jnp.float64)
        
    #     # 对每个单元计算梯度贡献
    #     for e in range(nelems):
    #         elem_nodes = self.equ_solver.elements[e]
            
    #         if len(elem_nodes) == 3:
    #             elem_coords = coords[elem_nodes]
    #             area = abs(0.5 * (
    #                 (elem_coords[1][0] - elem_coords[0][0]) * (elem_coords[2][1] - elem_coords[0][1]) -
    #                 (elem_coords[2][0] - elem_coords[0][0]) * (elem_coords[1][1] - elem_coords[0][1])
    #             ))
    #             area = max(area, 1e-15)  # 防止除零
                
    #             # 近似梯度点积
    #             dv1_01 = v1[elem_nodes[1]] - v1[elem_nodes[0]]
    #             dv1_02 = v1[elem_nodes[2]] - v1[elem_nodes[0]]
    #             dv2_01 = v2[elem_nodes[1]] - v2[elem_nodes[0]]
    #             dv2_02 = v2[elem_nodes[2]] - v2[elem_nodes[0]]
                
    #             contrib_0 = (dv1_01 * dv2_01 + dv1_02 * dv2_02) / area
    #             result = result.at[elem_nodes[0]].add(contrib_0)
    #             result = result.at[elem_nodes[1]].add(contrib_0)
    #             result = result.at[elem_nodes[2]].add(contrib_0)
        
    #     return result / max(nelems, 1)
    
    def eval_hessian_res(self, u_hat: np.ndarray) -> np.ndarray:
        """
        计算数据残差损失的 Hessian 向量乘法
        
        H_res @ û 包含三部分：
        1. 一阶项: ∫ û exp(u) ∇q · ∇w
        2. 二阶项 A: ∫ exp(u) ∇δw · ∇q
        3. 二阶项 B: ∫ exp(u) ∇w · ∇δq
        
        Parameters:
        -----------
        u_hat : np.ndarray
            方向向量
            
        Returns:
        --------
        result : np.ndarray
            Hessian 向量乘积结果
        """
        assert u_hat.shape[0] == self.num_dofs
        
        # 将 u_hat 包装成函数形式
        u_hat_fun = u_hat.copy()
        
        # 求解增量正向方程得到 δw
        self.equ_solver.inc_forward_solve(u_hat_fun)
        pp = self.equ_solver.sol_inc_forward
        
        # 计算残差方向
        vec = np.array(self.S @ self.equ_solver.sol_inc_forward).squeeze()
        vec = vec / (self.noise.std_dev**2)
        
        # 求解增量伴随方程得到 δq
        self.equ_solver.inc_adjoint_solve(vec, u_hat_fun)
        qq = self.equ_solver.sol_inc_adjoint
        
        # 存储
        self.pp = pp.copy()
        self.qq = qq.copy()
        
        # 计算三项贡献
        # 第一项: û * exp(u) * ∇q·∇w
        u_safe = np.clip(self.equ_solver.u, -20, 20)
        term1 = u_hat * np.exp(u_safe) * self._compute_gradient_product(self.q, self.p)
        
        # 第二、三项: 更复杂的积分（简化处理）
        term2_3 = np.exp(u_safe) * (
            self._compute_gradient_product(pp, self.q) +
            self._compute_gradient_product(self.p, qq)
        )
        
        # 合并
        A_total = term1 + term2_3
        
        # 应用质量矩阵的逆
        if self.solverM is not None:
            val = self.solverM.solve(A_total)
        else:
            val = A_total
        
        return jnp.array(val) if not isinstance(val, np.ndarray) else val
    
    def linearized_forward_solve(self, u_hat: np.ndarray, **kwargs):
        """线性化前向求解器（即增量正向求解）"""
        val = self.equ_solver.inc_forward_solve(u_hat, **kwargs)
        return val  # 保持 JAX 数组

    def linearized_adjoint_solve(self, vec: np.ndarray,
                                  u_hat: np.ndarray,
                                  **kwargs):
        """线性化伴随求解器（即增量伴随求解）"""
        val = self.equ_solver.inc_adjoint_solve(vec, u_hat, **kwargs)
        return val  # 保持 JAX 数组


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Darcy Flow Solver")
    print("=" * 60)
    
    # 创建测试网格
    nx, ny = 10, 10
    coords, elements = create_mesh_2d(nx=nx, ny=ny, element_type='tri')
    print(f"✓ Created mesh with {coords.shape[0]} nodes")
    
    # 测试 Darcy 流求解器
    solver = EquSolverDarcyFlow(coords, elements)
    print(f"✓ Initialized EquSolverDarcyFlow")
    
    # 正问题求解
    w = solver.forward_solve()
    print(f"✓ Forward solve completed, solution norm: {np.linalg.norm(w):.6f}")
    
    # 伴随问题测试
    test_vec = np.random.rand(5)
    lam = solver.adjoint_solve(test_vec)
    print(f"✓ Adjoint solve completed")
    
    # 增量方程测试
    u_hat = np.random.randn(coords.shape[0]) * 0.01
    dw = solver.inc_forward_solve(u_hat)
    print(f"✓ Incremental forward solve completed")
    
    dlam = solver.inc_adjoint_solve(test_vec, u_hat)
    print(f"✓ Incremental adjoint solve completed")
    
    print("\n" + "=" * 60)
    print("All Darcy Flow tests passed!")
    print("=" * 60)
