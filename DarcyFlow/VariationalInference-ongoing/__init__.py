# -*- coding: utf-8 -*-
"""
变分推断 (Variational Inference, VI) 模块

变分推断通过优化近似分布来逼近后验：
q*(u) = argmin_q KL(q(u) || π(u|d))

本模块包含:
- Mean-Field VI: 均场近似（假设参数独立）
- SVGD: Stein Variational Gradient Descent

注意：此目录标记为 "ongoing"，
表示这些方法仍在开发和完善中。

参考:
- Blei et al. (2017) "Variational Inference: A Review for Statisticians"
- Liu & Wang (2016) "Stein Variational Gradient Descent"
"""

from .mean_field_VI import MeanFieldVI
from .svgd import SVGD

__all__ = ['MeanFieldVI', 'SVGD']
