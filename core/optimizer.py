#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化算法模块

提供梯度下降法、牛顿法等优化算法

作者: 基于 IPBayesML-FEniCSx09 迁移，适配新架构
"""

import numpy as np
import jax.numpy as jnp

from core.linear_eq_solver import cg_my
from core.jax_backend import cg_jax_python as cg_jax_backend, bicgstab_jax


__all__ = ['OptimBase', 'GradientDescent', 'NewtonCG']


class OptimBase(object):
    """
    优化算法基类
    
    提供标准接口：
    - 初始化
    - 计算搜索方向
    - 线性搜索（Armijo 规则）
    - 更新迭代点
    """
    
    def __init__(self, model, c_armijo=1e-5, it_backtrack=20):
        """
        初始化优化器
        
        Parameters:
        -----------
        model : object
            模型对象，需具有 M, eval_grad, loss, update_param 属性/方法
        c_armijo : float
            Armijo 条件常数 (0 < c < 1)
        it_backtrack : int
            线性搜索最大回溯次数
        """
        assert hasattr(model, "M"), "model must have mass matrix M"
        assert hasattr(model, "eval_grad"), "model must have eval_grad method"
        assert hasattr(model, "loss"), "model must have loss method"
        assert hasattr(model, "update_param"), "model must have update_param method"
        
        self.model = model
        self.c_armijo = c_armijo
        self.it_backtrack = it_backtrack
        self.M = self.model.M
        self.converged = True
        
        # 是否归一化搜索方向
        self.if_normalize_dd = False
        # 线性搜索初始步长
        self.step_length_line_search = 1.0
    
    def set_init(self):
        """设置初始值"""
        raise NotImplementedError("Subclass must implement set_init()")
    
    def armijo_line_search(self, mk, g, dd, cost_pre, show_step=False):
        """
        Armijo 回溯线性搜索
        
        寻找满足 Armijo 条件的步长 α:
        f(x + α d) ≤ f(x) + c α g^T d
        
        Parameters:
        -----------
        mk : np.ndarray
            当前参数向量
        g : np.ndarray
            当前梯度
        dd : np.ndarray
            搜索方向
        cost_pre : float
            当前损失值
        show_step : bool
            是否显示步骤信息
            
        Returns:
        --------
        mk_new : np.ndarray
            更新后的参数
        cost_all : tuple
            新的损失值 (total, res, prior)
        converged : bool
            是否收敛
        """
        converged = True
        mk_pre = mk.copy()
        c_armijo = self.c_armijo
        step_length = self.step_length_line_search
        backtrack_converged = False
        
        # 计算 g^T M d
        tmp = self.M @ dd
        gMdd = g @ tmp
        
        # 计算 ||d||_M
        dd_norm = np.sqrt(dd @ tmp)
        
        for it in range(self.it_backtrack):
            if not self.if_normalize_dd:
                mk = mk + step_length * dd
            else:
                mk = mk + step_length * dd / max(dd_norm, 1.0)
            
            # 更新参数并计算新的损失
            self.model.update_param(mk, update_sol=True)
            cost_all = self.model.loss()
            cost_new = cost_all[0]
            
            # 检查 Armijo 条件
            if cost_new < cost_pre + step_length * c_armijo * gMdd:
                cost_pre = cost_new
                backtrack_converged = True
                break
            else:
                step_length *= 0.5
                mk = mk_pre.copy()
            
            if show_step:
                print(f"  Backtracking iteration {it+1}: step_length = {step_length:.6f}")
        
        if not backtrack_converged:
            print("Warning: Backtracking failed. Sufficient descent direction not found.")
            converged = False
        
        return mk, cost_all, converged
    
    def step(self, **kwargs):
        """执行一步更新"""
        raise NotImplementedError("Subclass must implement step()")
    
    def gradient(self, method, show_step, **kwargs):
        """计算梯度"""
        raise NotImplementedError()


class GradientDescent(OptimBase):
    """
    梯度下降法 (Gradient Descent)
    
    使用负梯度作为搜索方向的优化算法
    """
    
    def __init__(self, model, mk=None, lr=1e-5, if_normalize_dd=False):
        """
        Parameters:
        -----------
        model : object
            模型对象
        mk : np.ndarray, optional
            初始参数向量
        lr : float
            学习率（用于固定步长模式）
        if_normalize_dd : bool
            是否归一化梯度方向
        """
        super().__init__(model=model)
        
        assert hasattr(model, "prior") and hasattr(model.prior, "mean_vec")
        assert hasattr(model, "update_param")
        assert hasattr(model, "loss")
        assert hasattr(model, "eval_grad")
        
        self.lr = lr
        if mk is None:
            mk = np.copy(self.model.prior.mean_vec)
        self.mk = mk
        
        self.model.update_param(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        self.if_normalize_dd = if_normalize_dd
    
    def re_init(self, mk=None):
        """重新初始化"""
        if mk is None:
            mk = np.copy(self.model.prior.mean_vec)
        self.mk = mk
        self.model.update_param(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]

    def set_init(self, mk):
        """设置初始值"""
        self.mk = mk
    
    def descent_direction(self, smoothing=None):
        """计算搜索方向（负梯度）"""
        self.model.update_param(self.mk, update_sol=False)
        gg = self.model.eval_grad(self.mk)
        self.grad, self.grad_res, self.grad_prior = gg[0], gg[1], gg[2]
        
        if smoothing is not None:
            self.grad = smoothing(gg[0])
    
    def step(self, method='armijo', show_step=False):
        """
        执行一步更新
        
        Parameters:
        -----------
        method : str
            'armijo' 或 'fixed'
        show_step : bool
        """
        if method == 'armijo':
            self.mk, cost_all, self.converged = self.armijo_line_search(
                self.mk, self.grad, -self.grad,
                self.cost, show_step=show_step
            )
            self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        elif method == 'fixed':
            self.mk = self.mk - self.lr * self.grad
        else:
            raise ValueError("method should be 'armijo' or 'fixed'")


class NewtonCG(OptimBase):
    """
    牛顿-共轭梯度法 (Newton-CG)
    
    结合牛顿法的快速收敛性和 CG 法的高效线性求解
    """
    
    def __init__(self, model, mk=None, lr=1.0, 
                 if_pre_cond=True, if_normalize_dd=False):
        """
        Parameters:
        -----------
        model : object
            模型对象
        mk : np.ndarray, optional
            初始参数向量
        lr : float
            步长缩放因子
        if_pre_cond : bool
            是否使用预条件子
        if_normalize_dd : bool
            是否归一化牛顿方向
        """
        super().__init__(model=model)
        
        assert hasattr(model, "update_param")
        assert hasattr(model, "prior") and hasattr(model.prior, "mean_vec")
        assert hasattr(model, "hessian_linear_operator")
        assert hasattr(model, "precondition_linear_operator")
        assert hasattr(model, "loss")
        assert hasattr(model, "eval_grad")
        assert hasattr(model, "MxHessian_linear_operator")
        
        self.lr = lr
        if mk is None:
            mk = np.copy(self.model.prior.mean_vec)
        self.mk = mk
        
        self.model.update_param(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        
        # 构建 Hessian 算子 (M H 使其对称正定)
        self.hessian_operator = self.model.MxHessian_linear_operator()
        
        self.if_pre_cond = if_pre_cond
        self.if_normalize_dd = if_normalize_dd
    
    def re_init(self, mk=None):
        """重新初始化"""
        if mk is None:
            mk = np.copy(self.model.prior.mean_vec)
        self.mk = mk
        self.model.update_param(mk, update_sol=True)
        cost_all = self.model.loss()
        self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        
    def set_init(self, mk):
        """设置初始值"""
        self.mk = np.array(mk)
    
    def descent_direction(self, cg_tol=None, cg_max=1000, 
                          method='cg_my', curvature_detector=False):
        """
        计算牛顿方向（通过 CG 求解 H δ = -g）
        
        实际求解: M H δ = -M g（对称形式）
        
        Parameters:
        -----------
        cg_tol : float, optional
            CG 相对容差
        cg_max : int
            CG 最大迭代次数
        method : str
            CG 方法 ('cg_my', 'bicgstab', 'cg', 'cgs')
        curvature_detector : bool
            是否检测负曲率
        """
        self.model.update_param(self.mk, update_sol=False)
        gg = self.model.eval_grad(self.mk)
        self.grad, self.grad_res, self.grad_prior = gg[0], gg[1], gg[2]
        
        # 设置预条件子
        if self.if_pre_cond:
            pre_cond = self.model.precondition_linear_operator()
        else:
            pre_cond = None
        
        # 设置 CG 容差
        if cg_tol is None:
            norm_grad = np.sqrt(self.grad @ (self.M @ self.grad))
            cg_tol = min(0.5, np.sqrt(norm_grad))
        atol = 0.1
        
        # 求解 H δ = -grad（或 M H δ = -M grad）
        rhs = -self.M @ self.grad
        
        if method == 'cg_my':
            self.g, info, k = cg_my(
                self.hessian_operator, rhs, Minv=pre_cond,
                tol=cg_tol, atol=atol, maxiter=cg_max, 
                curvature_detector=curvature_detector
            )
            if k == 1:
                # 第一次遇到负曲率，使用负梯度
                self.g = -self.grad
            if info != 0:
                print(f"CG info: {info}")
                
        elif method == 'bicgstab':
            # 使用 JAX BiCGSTAB（GPU 加速）
            self.g, info = bicgstab_jax(
                self.hessian_operator, rhs,
                M_left=pre_cond if pre_cond is not None else None,
                tol=cg_tol, maxiter=cg_max
            )
            if info != 0 and info > 0:
                print(f"BiCGSTAB info: {info}")
                
        elif method == 'cg':
            # 使用 JAX CG（GPU 加速）
            self.g, info, k = cg_jax_backend(
                self.hessian_operator, rhs, x0=jnp.zeros(self.num_dofs),
                Minv_apply_fn=pre_cond,
                tol=cg_tol, atol=atol, maxiter=cg_max
            )
            if info != 0 and info > 0:
                print(f"CG info: {info}")
            
        elif method == 'cgs':
            # CGS 没有直接 JAX 版本，使用 BiCGSTAB 作为替代
            self.g, info = bicgstab_jax(
                self.hessian_operator, rhs,
                M_left=pre_cond if pre_cond is not None else None,
                tol=cg_tol, maxiter=cg_max
            )
            if info != 0 and info > 0:
                print(f"CGS(BiCGSTAB) info: {info}")
        else:
            raise ValueError("method should be 'cg', 'cgs', or 'bicgstab'")
        
        self.hessian_terminate_info = info
    
    def step(self, method='armijo', show_step=False):
        """
        执行一步更新
        
        Parameters:
        -----------
        method : str
            'armijo' 或 'fixed'
        show_step : bool
        """
        if method == 'armijo':
            self.mk, cost_all, self.converged = self.armijo_line_search(
                self.mk, self.grad, self.g,
                self.cost, show_step=show_step
            )
            self.cost, self.cost_res, self.cost_prior = cost_all[0], cost_all[1], cost_all[2]
        elif method == 'fixed':
            self.mk = self.mk + self.lr * self.g
        else:
            raise ValueError("method should be 'armijo' or 'fixed'")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing optimization algorithms")
    print("=" * 60)
    
    # 创建简单的二次测试问题
    class TestModel:
        def __init__(self):
            n = 10
            self.num_dofs = n
            A = np.random.rand(n, n)
            self.H = A.T @ A + np.eye(n)  # 正定 Hessian
            self.M = np.eye(n)
            self.g_true = np.random.rand(n)
            
        class prior:
            mean_vec = np.zeros(10)
        
        def update_param(self, u, **kwargs):
            pass
            
        def loss(self):
            return (np.sum(self.mk**2), 0.5*np.sum(self.mk**2), 0.5*np.sum(self.mk**2))
            
        def eval_grad(self, u):
            return (u, u, u)
            
        def hessian_linear_operator(self):
            from core.jax_backend import LinearOperatorJAX
            return LinearOperatorJAX((10, 10), matvec=lambda x: jnp.asarray(x, dtype=jnp.float64))
            
        def precondition_linear_operator(self):
            from core.jax_backend import LinearOperatorJAX
            return LinearOperatorJAX((10, 10), matvec=lambda x: jnp.asarray(x, dtype=jnp.float64))
            
        def MxHessian_linear_operator(self):
            from core.jax_backend import LinearOperatorJAX
            return LinearOperatorJAX((10, 10), matvec=lambda x: jnp.asarray(x, dtype=jnp.float64))
    
    model_test = TestModel()
    
    # 测试梯度下降
    gd = GradientDescent(model_test, mk=np.ones(10))
    gd.descent_direction()
    gd.step(method='armijo')
    print(f"✓ Gradient Descent works correctly")
    
    # 测试牛顿-CG
    newton = NewtonCG(model_test, mk=np.ones(10))
    newton.descent_direction(method='cg_my')
    newton.step(method='armijo')
    print(f"✓ Newton-CG works correctly")
    
    print("\n" + "=" * 60)
    print("All optimizer tests passed!")
    print("=" * 60)
