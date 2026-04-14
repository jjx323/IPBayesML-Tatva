#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化绘图模块

提供网格绘制、函数可视化等功能

作者: 基于 IPBayesML-FEniCSx09 迁移
"""

import numpy as np
from pathlib import Path
from typing import Optional


def plot_mesh(coords, elements, show=True, path=None, title="Mesh"):
    """
    绘制有限元网格
    
    Parameters:
    -----------
    coords : np.ndarray
        节点坐标 (N, dim)
    elements : np.ndarray
        单元连接关系
    show : bool
        是否显示图形
    path : str or Path, optional
        保存路径
    title : str
        图形标题
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if elements.shape[1] == 3:
            # 三角形网格
            triang = tri.Triangulation(coords[:, 0], coords[:, 1], elements)
            ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.5)
        else:
            # 四边形或其他
            for elem in elements:
                pts = coords[elem]
                pts_closed = np.vstack([pts, pts[0]])
                ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'k-', linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.set_title(title)
        
        if path is not None:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
            
    except ImportError:
        print("Warning: matplotlib not available for plotting")


def plot_fun2d(fun_vec, coords=None, elements=None, nx=200, 
               show=True, path=None, grid_on=False, package="matplotlib",
               title="Function", vmin=None, vmax=None, cmap='viridis'):
    """
    绘制二维标量函数
    
    Parameters:
    -----------
    fun_vec : np.ndarray
        函数值向量 (N,) 或可索引对象
    coords : np.ndarray, optional
        节点坐标 (N, 2)。如果为None，需要从外部获取
    elements : np.ndarray, optional
        单元连接关系
    nx : int
        插值分辨率
    show : bool
        是否显示图形
    path : str or Path, optional
        保存路径
    grid_on : bool
        是否显示网格
    package : str
        绘图包名称
    title : str
        图形标题
    vmin, vmax : float, optional
        颜色范围
    cmap : str
        颜色映射名称
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        
        # 处理输入
        if hasattr(fun_vec, 'x') and hasattr(fun_vec, 'array'):
            # dolfinx Function 类型
            vals = np.array(fun_vec.x.array)
        else:
            vals = np.array(fun_vec).flatten()
        
        if coords is None:
            print("Error: coordinates required for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        if elements is not None and elements.shape[1] == 3:
            # 三角形插值
            triang = tri.Triangulation(coords[:, 0], coords[:, 1], elements)
            
            tcf = ax.tricontourf(triang, vals, levels=20, cmap=cmap, 
                                  vmin=vmin, vmax=vmax)
            plt.colorbar(tcf, ax=ax, label='Value')
            
            if grid_on:
                ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
        else:
            # 散点图或规则网格插值
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=vals, 
                                cmap=cmap, s=10, vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, ax=ax, label='Value')
        
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        if path is not None:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()
            
    except ImportError:
        print("Warning: matplotlib not available for plotting")
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")


def project(source_vals, target_coords=None, source_coords=None,
             elements=None):
    """
    将函数从一个空间（源网格）投影到另一个空间（目标网格）

    Parameters
    ----------
    source_vals : np.ndarray
        源空间上的函数值，shape = (N_source,)
    target_coords : np.ndarray, optional (positional or keyword)
        目标网格节点坐标，shape = (N_target, dim)
    source_coords : np.ndarray, optional
        源网格节点坐标，shape = (N_source, dim)。
        若提供则在两个不同网格间进行插值投影。
    elements : np.ndarray, optional
        源网格单元连接关系

    Returns
    -------
    projected : np.ndarray, shape = (N_target,)
        投影到目标网格上的函数值
    """
    source_vals = np.asarray(source_vals)

    # 兼容旧调用方式: project(vals, coords) — 此时第二个参数是 target_coords
    # 以及新调用方式: project(vals, target_coords=coords, source_coords=src_coords)

    if source_coords is None:
        # 无源网格信息，无法做跨网格插值
        if target_coords is not None and len(source_vals) == target_coords.shape[0]:
            # 维度一致且无跨网格需求 → 直接返回
            return np.array(source_vals, dtype=np.float64)
        else:
            # 维度不一致但无法插值 → 截断/填充兜底
            n_target = target_coords.shape[0] if target_coords is not None else len(source_vals)
            n_copy = min(len(source_vals), n_target)
            result = np.zeros(n_target, dtype=np.float64)
            result[:n_copy] = source_vals[:n_copy]
            print(f"[project] WARNING: No source_coords, using zero-padding "
                  f"({len(source_vals)} -> {n_target})")
            return result

    # 有源网格坐标 → 使用 scipy.interpolate.griddata 做跨网格线性插值
    from scipy.interpolate import griddata
    source_coords = np.asarray(source_coords, dtype=np.float64)

    if target_coords is None:
        raise ValueError("target_coords must be provided when source_coords is given")

    result = griddata(
        source_coords, source_vals, target_coords, method='linear'
    )
    return np.nan_to_num(result, nan=0.0)
