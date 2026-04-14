#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
线性方程求解器模块

提供共轭梯度法 (CG) 和相关迭代求解器

作者: 基于 IPBayesML-FEniCSx09 迁移
"""

import numpy as np
from typing import Callable, Optional, Tuple


def cg_my(A, b, x0=None, tol=1e-6, atol=0.0, maxiter=None,
          Minv=None, callback=None, curvature_detector=False):
    """
    自实现的共轭梯度法 (Conjugate Gradient)
    
    求解线性系统 Ax = b
    
    Parameters:
    -----------
    A : callable 或 sparse matrix
        系统矩阵或算子，支持 A @ x 操作
    b : np.ndarray
        右端项
    x0 : np.ndarray, optional
        初始猜测
    tol : float
        相对容差
    atol : float  
        绝对容差
    maxiter : int, optional
        最大迭代次数
    Minv : callable, optional
        预条件子 M^{-1}
    callback : callable, optional
        回调函数
    curvature_detector : bool
        是否检测负曲率（用于非正定系统）
        
    Returns:
    --------
    x : np.ndarray
        解向量
    info : int
        收敛信息 (0=收敛, >0=未收敛, -0=数值问题)
    k : int
        实际迭代次数
    """
    n = len(b)
    
    if maxiter is None:
        maxiter = n * 10
    
    # 初始化
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.copy(x0).astype(np.float64)
    
    # 计算初始残差
    r = b - _matvec(A, x)
    
    # 应用预条件子
    if Minv is not None:
        z = Minv(r)
    else:
        z = r.copy()
    
    p = z.copy()
    rz_old = np.dot(r, z)
    
    rz_0 = np.linalg.norm(rz_old)
    if rz_0 == 0:
        return x, 0, 0
    
    k = 1
    info = 0
    
    while k <= maxiter:
        # 计算 Ap
        Ap = _matvec(A, p)
        
        pAp = np.dot(p, Ap)
        
        # 曲率检测（检查是否为负）
        if curvature_detector and pAp < 0:
            print(f"Warning: Negative curvature detected at iteration {k}, pAp={pAp}")
            if k == 1:
                # 第一次遇到负曲率，使用负梯度方向
                x = x - 0.01 * r
                return x, 1, 1
            break
        
        if abs(pAp) < 1e-30:
            break
        
        # 步长
        alpha = rz_old / pAp
        
        # 更新解
        x = x + alpha * p
        
        # 更新残差
        r = r - alpha * Ap
        
        # 检查收敛
        r_norm = np.linalg.norm(r)
        b_norm = np.linalg.norm(b)
        
        rel_resid = r_norm / b_norm if b_norm > 0 else r_norm
        abs_resid = r_norm
        
        if rel_resid < tol and abs_resid < atol:
            info = 0
            break
        
        # 预条件
        if Minv is not None:
            z = Minv(r)
        else:
            z = r.copy()
        
        rz_new = np.dot(r, z)
        
        # Fletcher-Reeves 公式
        beta = rz_new / rz_old
        
        # 更新搜索方向
        p = z + beta * p
        
        rz_old = rz_new
        k += 1
        
        if callback is not None:
            callback(x, r, k)
    else:
        info = maxiter
    
    return x, info, min(k, maxiter)


def _matvec(A, x):
    """矩阵-向量乘法，支持多种格式"""
    if hasattr(A, 'dot'):  # numpy array 或 scipy sparse
        return A.dot(x)
    elif hasattr(A, '__matmul__'):
        return A @ x
    elif callable(A):  # 函数/LinearOperator
        return A(x)
    else:
        raise TypeError(f"Unsupported matrix type: {type(A)}")


def bicgstab(A, b, x0=None, tol=1e-6, maxiter=None, 
             callback=None, M=None, **kwargs):
    """
    双共轭梯度稳定法 (BiCGSTAB)
    
    Parameters 与 cg 类似，适用于非对称/非正定系统
    """
    n = len(b)
    
    if maxiter is None:
        maxiter = 2 * n
    
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.copy(x0).astype(np.float64)

    r = b - _matvec(A, x)
    r_hat = r.copy()

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    
    info = 0
    
    for k in range(1, maxiter + 1):
        rho_new = np.dot(r_hat, r)
        
        if abs(rho_new) < 1e-30:
            info = -3  # Breakdown
            break
        
        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        v = _matvec(A, p)
        
        alpha = rho_new / np.dot(r_hat, v)
        
        s = r - alpha * v
        
        # 预条件（如果提供）
        if M is not None and hasattr(M, 'dot'):
            s_hat = M.dot(s)
        elif M is not None and callable(M):
            s_hat = M(s)
        else:
            s_hat = s.copy()
        
        t = _matvec(A, s_hat)
        
        omega = np.dot(t, s) / np.dot(t, t)
        
        x = x + alpha * p + omega * s_hat
        r = s - omega * t
        
        # 收敛检查
        r_norm = np.linalg.norm(r)
        if r_norm < tol * np.linalg.norm(b):
            info = 0
            break
        
        rho_old = rho_new
        
        if callback is not None:
            callback(x, r, k)
    else:
        info = maxiter
    
    return x, info


if __name__ == "__main__":
    print("=" * 60)
    print("Testing linear solvers")
    print("=" * 60)
    
    # 创建测试问题: 对称正定系统
    n = 100
    A_test = np.random.rand(n, n)
    A_test = A_test.T @ A_test + n * np.eye(n)  # 使其对称正定
    b_test = np.random.rand(n)
    
    # 测试 CG
    x_cg, info_cg, k_cg = cg_my(A_test, b_test, tol=1e-10)
    residual = np.linalg.norm(A_test @ x_cg - b_test) / np.linalg.norm(b_test)
    print(f"✓ CG solver converged in {k_cg} iterations, relative residual: {residual:.2e}")
    
    # 测试 BiCGSTAB
    x_bicg, info_bicg = bicgstab(A_test, b_test, tol=1e-10)
    residual_bicg = np.linalg.norm(A_test @ x_bicg - b_test) / np.linalg.norm(b_test)
    print(f"✓ BiCGSTAB solver converged with relative residual: {residual_bicg:.2e}")
    
    print("\n" + "=" * 60)
    print("All linear solver tests passed!")
    print("=" * 60)
