# -*- coding: utf-8 -*-
"""
高斯近似 (Laplace Approximation) 模块

实现基于 Laplace 近似的后验分布逼近:
1. 计算 MAP 估计
2. 在 MAP 点计算后验 Hessian 矩阵
3. 使用高斯分布 N(u_MAP, H^{-1}) 逼近后验

适用场景:
- 后验接近高斯分布的情况（数据量大时）
- 快速不确定性量化 (UQ)
- 作为其他方法的初始化

参考:
- "Laplace approximations for high-dimensional Bayesian inference" 
"""

from .gaussian_approx import GaussianApproximate

__all__ = ['GaussianApproximate']
