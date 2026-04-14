# -*- coding: utf-8 -*-
"""
序贯蒙特卡洛 (Sequential Monte Carlo, SMC) 模块

SMC 是一种粒子滤波方法，通过一系列中间分布
逐步从先验过渡到目标后验分布。

核心思想:
1. 定义一系列"温度"或噪声水平: π_t(u) ∝ π_prior(u)^{1-t} * L(y|u)^t
2. 在每个层级 t，对粒子进行重采样 + MCMC 移动步骤
3. 最终得到近似后验的加权粒子集合

优势:
- 适合多峰后验分布
- 天然支持并行计算
- 可扩展到高维问题

本模块包含:
- SMC: 基础 SMC 采样器（配合 pCN）
- smc_newton_pcnl: 结合 Newton 优化的高级 SMC（配合 Newton-pCNL）

参考:
- Del Moral et al. (2006) "Sequential Monte Carlo samplers"
"""

from .SMC_sampler import SMC, SMCResult

__all__ = ['SMC', 'SMCResult']
