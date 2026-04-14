#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 tatva (JAX-FEM) 的贝叶斯逆问题模型基类

本模块提供：
1. 方程求解器基类 (EquSolverBase)
2. 贝叶斯模型基类 (ModelBase)

作者: 基于原 IPBayesML-FEniCSx09 重构，使用 tatva 库
"""

import numpy as np
import jax.numpy as jnp

# JAX 后端替代 scipy.sparse
from core.jax_backend import LinearOperatorJAX, cg_jax, bicgstab_jax
# 保留 scipy 导入名用于兼容性
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from typing import Optional, Tuple, Callable, Dict, Any


def _assemble_mass_matrix_for_model(coords: np.ndarray, elements: np.ndarray):
    """
    [Core 内部] 组装质量矩阵 M（用于 ModelBase）
    
    质量矩阵 M_{ij} = ∫ φ_i φ_j dx（集中对角质量矩阵）
    
    注意：此函数是 core 模块的内部实现，与具体 PDE 问题无关。
    DarcyFlow 问题有自己的实现在 DarcyFlow/misc.py 中。
    """
    coords_jax = jnp.asarray(coords, dtype=jnp.float64)
    elements_jax = jnp.asarray(elements, dtype=jnp.int32)
    
    nnodes = coords_jax.shape[0]

    n0, n1, n2 = elements_jax[:, 0], elements_jax[:, 1], elements_jax[:, 2]
    c0, c1, c2 = coords_jax[n0], coords_jax[n1], coords_jax[n2]

    area = 0.5 * jnp.abs(
        (c1[:, 0] - c0[:, 0]) * (c2[:, 1] - c0[:, 1]) -
        (c2[:, 0] - c0[:, 0]) * (c1[:, 1] - c0[:, 1])
    )

    mass_contrib = area / 3.0

    from core.jax_backend import diags_jax
    diag_entries = jnp.zeros(nnodes, dtype=jnp.float64)
    diag_entries = diag_entries.at[n0].add(mass_contrib)
    diag_entries = diag_entries.at[n1].add(mass_contrib)
    diag_entries = diag_entries.at[n2].add(mass_contrib)

    return diags_jax(diag_entries)


__all__ = ['EquSolverBase', 'ModelBase']


class EquSolverBase:
    """
    PDE 方程求解器基类
    
    定义了标准接口，用于求解正问题、伴随问题和增量方程。
    子类需要实现具体的 PDE 求解逻辑。
    """
    
    def __init__(self):
        """初始化求解器"""
        # 以下属性应在子类中设置：
        self.Vh = None          # 函数空间（节点数）
        self.sol_forward = None  # 正问题解向量
        self.sol_adjoint = None  # 伴随问题解向量
        self.sol_inc_forward = None   # 增量正问题解
        self.sol_inc_adjoint = None   # 增量伴随问题解
        self.num_dofs = None     # 自由度数量
    
    def forward_solve(self, u=None) -> np.ndarray:
        """
        求解正问题
        
        Parameters:
        -----------
        u : np.ndarray, optional
            参数场（如渗透率场的对数）
            
        Returns:
        --------
        w : np.ndarray
            正问题的解（状态变量）
        """
        raise NotImplementedError("Subclass must implement forward_solve()")
    
    def adjoint_solve(self, vec: np.ndarray, u=None) -> np.ndarray:
        """
        求解伴随问题
        
        Parameters:
        -----------
        vec : np.ndarray
            伴随源项
        u : np.ndarray, optional  
            参数场
            
        Returns:
        --------
        lam : np.ndarray
            伴随问题的解
        """
        raise NotImplementedError("Subclass must implement adjoint_solve()")
    
    def inc_forward_solve(self, u_hat: np.ndarray, sol_forward=None) -> np.ndarray:
        """
        求解增量正问题（线性化后的前向方程）
        
        Parameters:
        -----------
        u_hat : np.ndarray
            参数的增量
        sol_forward : np.ndarray, optional
            正问题的解（如果已知，可避免重复计算）
            
        Returns:
        --------
        dw : np.ndarray
            状态变量的增量
        """
        raise NotImplementedError("Subclass must implement inc_forward_solve()")
    
    def inc_adjoint_solve(self, vec: np.ndarray, 
                          u_hat: np.ndarray,
                          sol_adjoint=None, 
                          simple=False) -> np.ndarray:
        """
        求解增量伴随问题
        
        Parameters:
        -----------
        vec : np.ndarray
            增量伴随源项
        u_hat : np.ndarray
            参数的增量
        sol_adjoint : np.ndarray, optional
            伴随问题的解
        simple : bool
            是否使用简化模式
            
        Returns:
        --------
        dlam : np.ndarray
            伴随变量的增量
        """
        raise NotImplementedError("Subclass must implement inc_adjoint_solve()")


class ModelBase:
    """
    贝叶斯逆问题模型的基类
    
    整合先验分布、PDE 求解器和噪声模型，提供：
    - 损失函数计算
    - 梯度计算（通过伴随方法）
    - Hessian 计算
    - 预条件子
    """
    
    def __init__(self, prior, equ_solver, noise, data: Dict[str, Any]):
        """
        初始化模型
        
        Parameters:
        -----------
        prior : object
            先验分布对象，需具有 generate_sample(), eval_grad(), eval_hessian()
        equ_solver : EquSolverBase
            PDE 方程求解器对象，需具有 forward_solve(), Vh, num_dofs
        noise : object  
            噪声模型对象，需具有 eval_CM_inner()
        data : dict
            观测数据字典，应包含 'coordinates' 和 'data' 键
                - coordinates: 观测点坐标 (M, dim)
                - data: 观测数据 (M,)
        """
        assert hasattr(equ_solver, 'forward_solve')
        assert hasattr(prior, 'generate_sample')
        
        self.prior = prior
        self.equ_solver = equ_solver
        self.noise = noise
        
        self.coordinates = data["coordinates"]
        self.data = np.array(data["data"])
        
        self._init_measurement_matrix(self.coordinates)
        
        assert hasattr(self.equ_solver, "num_dofs"), "equ_solver must have num_dofs"
        self.num_dofs = self.equ_solver.num_dofs
        
        # 组装质量矩阵 M
        self.M = _assemble_mass_matrix_for_model(
            self.equ_solver.coords if hasattr(self.equ_solver, 'coords') else None,
            self.equ_solver.elements if hasattr(self.equ_solver, 'elements') else None
        )
        
    def _init_measurement_matrix(self, coordinates: np.ndarray):
        """
        初始化测量矩阵 S
        
        S 用于从 FEM 解中提取观测点的值: d = S @ w
        """
        raise NotImplementedError("Subclass must implement _init_measurement_matrix()")
    
    def get_data(self, w: np.ndarray) -> np.ndarray:
        """
        从状态变量 w 提取观测数据
        
        Parameters:
        -----------
        w : np.ndarray
            FEM 解向量
            
        Returns:
        --------
        d : np.ndarray
            在观测点处的值
        """
        assert w.shape[0] == self.S.shape[1], \
            f"Measure matrix shape {self.S.shape} not compatible with input {w.shape}"
        return np.array(self.S @ w)
    
    def loss_res(self, u=None) -> float:
        """
        计算数据残差损失（负对数似然的一部分）
        
        L_res(u) = 0.5 * ||d_obs - G(u)||^2_Γ^{-1}
        
        Parameters:
        -----------
        u : np.ndarray, optional
            参数场
            
        Returns:
        --------
        loss : float
            数据残差损失值
        """
        raise NotImplementedError("Subclass must implement loss_res()")
    
    def loss_prior(self, u=None) -> float:
        """
        计算先验损失（负对数先验的一部分）
        
        L_prior(u) = 0.5 * ||u||^2_C^{-1}
        
        Parameters:
        -----------
        u : np.ndarray, optional
            参数场
            
        Returns:
        --------
        loss : float
            先验损失值
        """
        if u is not None:
            assert u.shape[0] == self.num_dofs
            val = 0.5 * self.prior.eval_CM_inner(u)
        else:
            val = 0.5 * self.prior.eval_CM_inner(
                getattr(self.equ_solver, 'u', np.zeros(self.num_dofs))
            )
        return val
    
    def loss(self, u=None) -> Tuple[float, float, float]:
        """
        计算总损失函数
        
        L(u) = L_res(u) + L_prior(u)
        
        Returns:
        --------
        total_loss : float
        res_loss : float
        prior_loss : float
        """
        loss_prior = self.loss_prior(u)
        loss_res = self.loss_res(u)
        loss_total = loss_res + loss_prior
        return loss_total, loss_res, loss_prior
    
    def eval_grad_res(self, u: np.ndarray) -> np.ndarray:
        """
        计算数据残差损失的梯度 ∇_u L_res(u)
        
        使用伴随方法计算梯度
        
        Parameters:
        -----------
        u : np.ndarray
            参数场
            
        Returns:
        --------
        grad : np.ndarray
            梯度向量
        """
        raise NotImplementedError("Subclass must implement eval_grad_res()")
    
    def eval_grad_prior(self, u: np.ndarray) -> np.ndarray:
        """
        计算先验损失的梯度 ∇_u L_prior(u)
        
        Parameters:
        -----------
        u : np.ndarray
            参数场
            
        Returns:
        --------
        grad : np.ndarray
            先验梯度向量
        """
        return self.prior.eval_grad(u)
    
    def eval_grad(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算总损失梯度
        
        Returns:
        --------
        grad_total : np.ndarray
        grad_res : np.ndarray
        grad_prior : np.ndarray
        """
        grad_res = self.eval_grad_res(u)
        grad_prior = self.eval_grad_prior(u)
        return grad_res + grad_prior, grad_res, grad_prior
    
    def eval_hessian_res(self, u_hat: np.ndarray) -> np.ndarray:
        """
        计算数据残差损失的 Hessian 向量乘积 H_res @ u_hat
        
        Parameters:
        -----------
        u_hat : np.ndarray
            方向向量
            
        Returns:
        --------
        result : np.ndarray
            Hessian 向量乘积结果
        """
        raise NotImplementedError("Subclass must implement eval_hessian_res()")
    
    def eval_hessian_prior(self, u: np.ndarray) -> np.ndarray:
        """
        计算先验损失的 Hessian 向量乘法 H_prior @ u
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            Hessian 向量乘积结果
        """
        return self.prior.eval_hessian(u)
    
    def eval_hessian(self, u: np.ndarray) -> np.ndarray:
        """
        计算 Hessian 向量乘法 (H_res + H_prior) @ u
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            总 Hessian 向量乘积结果
        """
        hessian_res = self.eval_hessian_res(u)
        hessian_prior = self.eval_hessian_prior(u)
        return hessian_res + hessian_prior
    
    def hessian_linear_operator(self):
        """
        将 Hessian 构建为 LinearOperator 对象（JAX 版本）
        
        Returns:
        --------
        op : LinearOperatorJAX
            Hessian 算子
        """
        linear_ope = LinearOperatorJAX(
            (self.num_dofs, self.num_dofs), 
            matvec=self.eval_hessian
        )
        return linear_ope
    
    def MxHessian(self, u: np.ndarray) -> np.ndarray:
        """
        计算 M @ Hessian @ u（使矩阵对称化）
        
        通常算法需要对称矩阵，这里通过左乘质量矩阵实现对称化
        
        Parameters:
        -----------
        u : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            M @ H @ u
        """
        return self.M @ self.eval_hessian(u)
    
    def MxHessian_linear_operator(self):
        """构建 M @ Hessian 作为 LinearOperator（JAX 版本）"""
        linear_op = LinearOperatorJAX(
            (self.num_dofs, self.num_dofs), 
            matvec=self.MxHessian
        )
        return linear_op
    
    def precondition(self, u: np.ndarray) -> np.ndarray:
        """应用预条件子"""
        return self.prior.precondition(u)
    
    def precondition_linear_operator(self):
        """构建预条件子作为 LinearOperator（JAX 版本）"""
        linear_ope = LinearOperatorJAX(
            (self.num_dofs, self.num_dofs), 
            matvec=self.precondition
        )
        return linear_ope


if __name__ == "__main__":
    print("=" * 60)
    print("Testing base classes")
    print("=" * 60)
    
    # 测试 EquSolverBase
    solver = EquSolverBase()
    print(f"✓ Created EquSolverBase instance")
    
    try:
        solver.forward_solve()
    except NotImplementedError:
        print("✓ forward_solve correctly raises NotImplementedError")
    
    # 测试 ModelBase
    class DummyPrior:
        def generate_sample(self): return np.zeros(10)
        def eval_grad(self, u): return u
        def eval_hessian(self, u): return u
        def eval_CM_inner(self, u): return np.sum(u**2)
        def precondition(self, u): return u
    
    class DummyNoise:
        def eval_CM_inner(self, u): return np.sum(u**2)
        mean = np.zeros(5)
        std_dev = 1.0
    
    class DummySolver(EquSolverBase):
        def __init__(self):
            super().__init__()
            self.num_dofs = 10
    
    solver_dummy = DummySolver()
    prior_dummy = DummyPrior()
    noise_dummy = DummyNoise()
    data_dict = {
        'coordinates': np.random.rand(5, 3),
        'data': np.random.rand(5)
    }
    
    try:
        model = ModelBase(prior_dummy, solver_dummy, noise_dummy, data_dict)
        print("✓ ModelBase initialization works (base class)")
    except Exception as e:
        print(f"⚠ ModelBase requires subclass implementation of _init_measurement_matrix")
    
    print("\n" + "=" * 60)
    print("Base class tests completed!")
    print("=" * 60)
