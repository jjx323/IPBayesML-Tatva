# -*- coding: utf-8 -*-
"""
优化方法模块

提供 Darcy 流贝叶斯逆问题的优化求解器:
- optim_methods: 完整的优化工作流程（Newton-CG, GD, 混合方法）
- optim_dim_independent: 维度独立优化的变体

使用场景:
- MAP 估计计算
- 高斯近似前的优化步骤
- rMAP 方法的初始化
"""

from .optim_methods import run_optimization, OptimizationResult
from .optim_dim_independent import run_dimension_independent_optimization

__all__ = [
    'run_optimization',
    'OptimizationResult', 
    'run_dimension_independent_optimization'
]
