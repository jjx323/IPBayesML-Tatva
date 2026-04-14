# -*- coding: utf-8 -*-
"""
随机化 MAP (randomized Maximum A Posteriori, rMAP) 模块

rMAP 是一种通过随机化数据扰动来生成后验样本的方法：

1. 对观测数据添加随机噪声：d_j = d + ε_j, ε_j ~ N(0, Γ)
2. 对每个扰动的数据求解 MAP 估计
3. 得到的 MAP 估计集合近似服从后验分布

优势:
- 天然并行化（每个样本独立计算）
- 不需要 MCMC 的混合时间
- 适用于中等规模问题
- 保留了优化方法的高效性

参考:
- Bui-Thanh et al. (2017) "Randomized MAP for large-scale linear inverse problems"
"""

from .rMAP import rMAP

__all__ = ['rMAP']
