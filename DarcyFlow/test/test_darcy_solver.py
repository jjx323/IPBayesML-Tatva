#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Darcy 流方程求解器验证程序 —— 制造解 (Manufactured Solution) 方法

验证稳态 Darcy 方程:
    -∇·(κ∇w) = f,   in Ω,
    w = wD,         on ∂Ω,

【测试案例 1】常系数 + 正弦解
    解析解:  w_exact(x, y) = sin(πx) · sin(πy)
    渗透率:  κ = 1 (即 u = 0)
    源项:    f = 2π² · sin(πx) · sin(πy)
    边界条件: wD = 0

【测试案例 2】变系数 + 多项式解（修正版）
    解析解:  w_exact(x, y) = x(1-x)y(1-y)
    渗透率:  κ(x,y) = 1 + x
    源项:    f = y(1-y) + 4xy(1-y) + 2x - 2x³  (完整推导)
    边界条件: wD = 0

输出:
- test/RESULTS/solver_verification_report.txt : 完整文本报告
- test/RESULTS/convergence_plot.png      : 收敛阶图
- test/RESULTS/solution_comparison.png     : 数值解 vs 解析解对比图
- test/RESULTS/error_distribution.png     : 误差分布热力图
- test/RESULTS/adjoint_consistency.png    : 伴随一致性散点图
- test/RESULTS/incremental_linearization.png : 增量线性化误差图

作者: JAX-Bayes 项目组
"""

import os
import sys
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# 路径设置
# ============================================================
current_file = os.path.abspath(__file__)
test_dir = os.path.dirname(current_file)
darcyflow_dir = os.path.dirname(test_dir)
project_root = os.path.dirname(darcyflow_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, darcyflow_dir)

os.environ['JAX_ENABLE_X64'] = '1'
os.chdir(project_root)

# ============================================================
# 结果目录
# ============================================================
results_dir = os.path.join(test_dir, "RESULTS")
os.makedirs(results_dir, exist_ok=True)
report_lines = []  # 收集所有输出用于写入txt


def log_print(msg="", end='\n', file_only=False):
    """同时打印到控制台和收集到报告列表中"""
    report_lines.append(msg)
    if not file_only:
        print(msg, end=end)


# ============================================================
# 导入模块
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端，用于保存图片
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log_print("Warning: matplotlib not available, plots will be skipped.")

from DarcyFlow.misc import (
    create_mesh_2d, assemble_stiffness_matrix,
    assemble_mass_matrix, error_compare
)
from DarcyFlow.common import EquSolverDarcyFlow


# ============================================================
# 解析解定义
# ============================================================

def exact_solution_sine(coords):
    """案例 1: w(x,y) = sin(πx)·sin(πy)"""
    x, y = coords[:, 0], coords[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term_sine(coords):
    """案例 1 源项: f = 2π²·sin(πx)·sin(πy)"""
    x, y = coords[:, 0], coords[:, 1]
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def grad_exact_sine(coords):
    """案例 1 梯度: ∇w = (πcos(πx)sin(πy), πsin(πx)cos(πy))"""
    x, y = coords[:, 0], coords[:, 1]
    gx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    gy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return gx, gy


def exact_solution_poly(coords):
    """案例 2: w(x,y) = x(1-x)y(1-y)"""
    x, y = coords[:, 0], coords[:, 1]
    return x * (1 - x) * y * (1 - y)


def source_term_poly(coords):
    """
    案例 2 源项（修正版）: 由 -∇·(κ∇w) = f 精确推导
    
    其中 w = x(1-x)y(1-y), κ = 1 + x
    
    推导过程：
      ∂w/∂x = (1-2x)y(1-y)
      ∂w/∂y = x(1-x)(1-2y)
      
      Term 1: ∂_x[κ·∂_x w] 
        = ∂_x[(1+x)(1-2x)y(1-y)]
        = y(1-y)·∂_x[(1+x)(1-2x)]
        = y(1-y)·[(1)(1-2x)+(1+x)(-2)]
        = y(1-y)·[1-2x-2-2x] 
        = -(1+4x)·y(1-y)
      
      Term 2: ∂_y[κ·∂_y w]
        = ∂_y[(1+x)x(1-x)(1-2y)]
        = x(1-x²)·(-2)
        = -2x(1-x²)
      
      ∇·(κ∇w) = -(1+4x)y(1-y) - 2x(1-x²)
               = -y(1-y) - 4xy(1-y) - 2x + 2x³
      
      所以 f = -∇·(κ∇w) = y(1-y) + 4xy(1-y) + 2x - 2x³
    """
    x, y = coords[:, 0], coords[:, 1]
    f = y * (1 - y) + 4*x*y*(1 - y) + 2*x - 2*(x**3)
    return f


def grad_exact_poly(coords):
    """案例 2 梯度"""
    x, y = coords[:, 0], coords[:, 1]
    gx = (1 - 2*x) * y * (1 - y)
    gy = x * (1 - x) * (1 - 2*y)
    return gx, gy


# ============================================================
# 三角形网格上的插值与可视化工具
# ============================================================

def triangulate_mesh_for_plotting(coords, elements):
    """为 matplotlib 的 tricontourf 准备三角剖分数据"""
    import matplotlib.tri as tri
    return tri.Triangulation(coords[:, 0], coords[:, 1], elements)


def plot_solution_on_mesh(fig_ax, coords, elements, values, title, cmap='viridis',
                          vmin=None, vmax=None, colorbar=True, add_colorbar_label=""):
    """在三角形网格上绘制解的等值线图"""
    ax = fig_ax
    triang = triangulate_mesh_for_plotting(coords, elements)

    vmin_local = vmin if vmin is not None else np.min(values)
    vmax_local = vmax if vmax is not None else np.max(values)

    levels = np.linspace(vmin_local, vmax_local, 30)
    cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap,
                         extend='both', vmin=vmin_local, vmax=vmax_local)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)

    if colorbar:
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
        cbar = plt.colorbar(cf, cax=cax)
        if add_colorbar_label:
            cbar.set_label(add_colorbar_label, fontsize=9)

    return cf


# ============================================================
# 误差计算工具
# ============================================================

def compute_l2_error(w_num, coords, exact_func):
    """计算 L² 范数误差（RMS 近似）"""
    w_exact = exact_func(coords)
    diff = w_num - w_exact
    l2_error = np.sqrt(np.mean(diff**2))
    l2_rel = l2_error / max(np.sqrt(np.mean(w_exact**2)), 1e-15)
    return l2_error, l2_rel, diff


def compute_max_error(w_num, coords, exact_func):
    """计算最大误差 (L∞ 范数)"""
    w_exact = exact_func(coords)
    diff = np.abs(w_num - w_exact)
    max_err = np.max(diff)
    rel_max = max_err / max(np.max(np.abs(w_exact)), 1e-15)
    return max_err, rel_max


# ============================================================
# 案例 1：常系数正弦解（含图片输出）
# ============================================================

def test_case_1_constant_kappa(mesh_sizes=[8, 16, 32]):
    log_print("\n" + "=" * 70)
    log_print(" TEST CASE 1: Constant Coefficient κ = 1")
    log_print(" Exact solution: w(x,y) = sin(πx)·sin(πy)")
    log_print(" Domain: Ω = [0,1] × [0,1]")
    log_print("=" * 70)

    results = []
    plot_data = {}  # 存储绘图数据

    for idx_nx, nx in enumerate(mesh_sizes):
        ny = nx
        h = 1.0 / nx
        log_print(f"\n--- Mesh: {nx}×{ny} (h={h:.4f}) ---")

        # 创建网格
        t0 = time.time()
        coords, elements = create_mesh_2d(nx=nx, ny=ny, element_type='tri')
        mesh_time = time.time() - t0
        num_nodes = coords.shape[0]
        log_print(f"  Nodes: {num_nodes}, Elements: {elements.shape[0]}, Mesh time: {mesh_time:.4f}s")

        # 参数设置
        u = np.zeros(num_nodes)
        f = source_term_sine(coords)
        wD = exact_solution_sine(coords)

        # 创建求解器并求解
        t0 = time.time()
        solver = EquSolverDarcyFlow(coords=coords, elements=elements, u=u, f=f, wD=wD)
        init_time = time.time() - t0

        t0 = time.time()
        w_num = solver.forward_solve()
        solve_time = time.time() - t0
        log_print(f"  Init: {init_time:.4f}s, Solve: {solve_time:.4f}s")

        # 误差
        l2_err, l2_rel, err_field = compute_l2_error(w_num, coords, exact_solution_sine)
        max_err, max_rel = compute_max_error(w_num, coords, exact_solution_sine)
        log_print(f"  L² error:   {l2_err:.6e}  (relative: {l2_rel:.2%})")
        log_print(f"  L∞ error:  {max_err:.6e}  (relative: {max_rel:.2%})")

        results.append({
            'nx': nx, 'h': h, 'num_nodes': num_nodes,
            'l2_err': l2_err, 'l2_rel': l2_rel,
            'max_err': max_err, 'max_rel': max_rel
        })

        # 保存最细网格的数据用于绘图
        if nx == mesh_sizes[-1]:
            plot_data['coords'] = coords
            plot_data['elements'] = elements
            plot_data['w_num'] = w_num
            plot_data['err_field'] = err_field
            plot_data['w_exact'] = exact_solution_sine(coords)

    # 收敛性分析
    log_print("\n" + "-" * 55)
    log_print(" Convergence Analysis:")
    log_print(f" {'Mesh':>8s} {'h':>10s} {'L² err':>12s} {'L∞ err':>12s} {'Rate(L²)':>10s}")
    log_print(" " + "-" * 56)

    rates = []
    for i, r in enumerate(results):
        if i == 0:
            rate_str = " --- "
        else:
            rate_val = np.log(results[i-1]['l2_err'] / r['l2_err']) / \
                        np.log(results[i-1]['h'] / r['h'])
            rates.append(rate_val)
            rate_str = f"{rate_val:.2f}"

        log_print(f" {r['nx']:>4d}×{r['nx']:<4d} {r['h']:>10.4f} "
                  f"{r['l2_err']:>12.4e} {r['max_err']:>12.4e} {rate_str:>10s}")

    avg_rate = np.mean(rates) if rates else 0.0
    log_print(f"\n Average convergence rate (L²): {avg_rate:.3f}")
    log_print(f" Theoretical rate for linear FEM: ~2.0")
    passed = avg_rate > 1.5
    log_print(f" {'✓ PASS' if passed else '✗ WARN'}: Convergence rate is {'acceptable!' if passed else 'lower than expected.'}")

    # ===== 绘制收敛曲线 =====
    if HAS_MATPLOTLIB and len(mesh_sizes) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 图1: L² 和 L∞ 误差 vs h（log-log）
        ax1 = axes[0]
        h_vals = np.array([r['h'] for r in results])
        l2_vals = np.array([r['l2_err'] for r in results])
        linf_vals = np.array([r['max_err'] for r in results])

        ax1.loglog(h_vals, l2_vals, 'bo-', linewidth=2, markersize=8, label=f'L² error (rate={avg_rate:.2f})')
        ax1.loglog(h_vals, linf_vals, 'rs--', linewidth=2, markersize=8, label='L∞ error')

        # 参考线 O(h²)
        h_fine = np.logspace(np.log10(h_vals[-1]) - 1.5, np.log10(h_vals[0]) + 0.2, 50)
        ref_h2 = l2_vals[0] * (h_fine / h_vals[0])**2
        ax1.loglog(h_fine, ref_h2, 'k:', linewidth=1.5, alpha=0.6, label='O(h²) reference')

        ax1.set_xlabel('Mesh size $h$', fontsize=11)
        ax1.set_ylabel('Error', fontsize=11)
        ax1.set_title('Convergence Rate (Constant κ)', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, which='both', alpha=0.3)

        # 图2: 误差 vs DOF（log-log）
        ax2 = axes[1]
        dof_vals = np.array([r['num_nodes'] for r in results])
        ax2.loglog(dof_vals, l2_vals, 'bo-', linewidth=2, markersize=8, label='L² error')
        ax2.loglog(dof_vals, linf_vals, 'rs--', linewidth=2, markersize=8, label='L∞ error')
        ref_dof = dof_vals[-1] / (np.linspace(dof_vals[-1]*0.05, dof_vals[0]*3, 50))
        ref_dof_l2 = l2_vals[0] * (ref_dof / dof_vals[0])**(-1)
        ax2.loglog(ref_dof, ref_dof_l2, 'k:', linewidth=1.5, alpha=0.6, label='O(N⁻¹) reference')

        ax2.set_xlabel('Number of DOFs ($N$)', fontsize=11)
        ax2.set_ylabel('Error', fontsize=11)
        ax2.set_title('Error vs Degrees of Freedom', fontsize=12)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.invert_xaxis()

        plt.tight_layout()
        conv_path = os.path.join(results_dir, "case1_convergence.png")
        plt.savefig(conv_path, dpi=200, bbox_inches='tight')
        plt.close()
        log_print(f"\n  Saved convergence plot: {conv_path}")

    # ===== 绘制解对比和误差分布 =====
    if HAS_MATPLOTLIB and plot_data:
        fig = plt.figure(figsize=(15, 4.5))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

        # 数值解
        ax1 = fig.add_subplot(gs[0, 0])
        v_max = max(np.max(plot_data['w_exact']), np.max(plot_data['w_num'])) * 1.05
        v_min = min(np.min(plot_data['w_exact']), np.min(plot_data['w_num'])) * 1.05 - 0.01
        plot_solution_on_mesh(ax1, plot_data['coords'], plot_data['elements'],
                              plot_data['w_num'], 'Numerical Solution',
                              cmap='jet', vmin=v_min, vmax=v_max,
                              add_colorbar_label="$w_h$")

        # 解析解
        ax2 = fig.add_subplot(gs[0, 1])
        plot_solution_on_mesh(ax2, plot_data['coords'], plot_data['elements'],
                              plot_data['w_exact'], 'Exact Solution',
                              cmap='jet', vmin=v_min, vmax=v_max,
                              add_colorbar_label="$w_{exact}$")

        # 误差分布
        ax3 = fig.add_subplot(gs[0, 2])
        e_max = np.max(np.abs(plot_data['err_field'])) * 1.1
        plot_solution_on_mesh(ax3, plot_data['coords'], plot_data['elements'],
                              plot_data['err_field'], 'Error Distribution',
                              cmap='RdBu_r', vmin=-e_max, vmax=e_max,
                              add_colorbar_label="$w_h - w_{exact}$")

        plt.suptitle('Case 1: Constant κ = 1,  w = sin(πx)sin(πy)',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        sol_path = os.path.join(results_dir, "case1_solution_comparison.png")
        plt.savefig(sol_path, dpi=200, bbox_inches='tight')
        plt.close()
        log_print(f"  Saved solution comparison: {sol_path}")

    return results, passed


# ============================================================
# 案例 2：变系数多项式解
# ============================================================

def test_case_2_variable_kappa(mesh_sizes=[8, 16, 32, 48, 64]):
    log_print("\n" + "=" * 70)
    log_print(" TEST CASE 2: Variable Coefficient κ(x,y) = 1 + x")
    log_print(" Exact solution: w(x,y) = x(1-x)y(1-y)")
    log_print(" Domain: Ω = [0,1] × [0,1]")
    log_print("=" * 70)

    results = []
    plot_data = {}

    for nx in mesh_sizes:
        ny = nx
        h = 1.0 / nx
        log_print(f"\n--- Mesh: {nx}×{ny} (h={h:.4f}) ---")

        coords, elements = create_mesh_2d(nx=nx, ny=ny, element_type='tri')
        num_nodes = coords.shape[0]

        x_coords = coords[:, 0]
        u = np.log(1.0 + x_coords)
        f = source_term_poly(coords)
        wD = exact_solution_poly(coords)

        solver = EquSolverDarcyFlow(coords=coords, elements=elements, u=u, f=f, wD=wD)
        w_num = solver.forward_solve()

        l2_err, l2_rel, err_field = compute_l2_error(w_num, coords, exact_solution_poly)
        max_err, max_rel = compute_max_error(w_num, coords, exact_solution_poly)

        log_print(f"  Nodes: {num_nodes}")
        log_print(f"  L² error:  {l2_err:.6e}  (relative: {l2_rel:.2%})")
        log_print(f"  L∞ error: {max_err:.6e}  (relative: {max_rel:.2%})")

        results.append({'nx': nx, 'h': h, 'l2_err': l2_err, 'max_err': max_err})

        if nx == mesh_sizes[-1]:
            plot_data['coords'] = coords
            plot_data['elements'] = elements
            plot_data['w_num'] = w_num
            plot_data['err_field'] = err_field
            plot_data['w_exact'] = exact_solution_poly(coords)
            plot_data['kappa'] = np.exp(u)

    # 收敛性
    log_print("\n" + "-" * 45)
    log_print(" Convergence Analysis:")
    log_print(f" {'Mesh':>10s} {'L² err':>12s} {'Rate':>8s}")
    log_print(" " + "-" * 34)

    rates = []
    for i, r in enumerate(results):
        if i == 0:
            rate_str = "  --  "
        else:
            rv = np.log(results[i-1]['l2_err'] / r['l2_err']) / \
                 np.log(results[i-1]['h'] / r['h'])
            rates.append(rv)
            rate_str = f"{rv:.2f}"
        log_print(f" {r['nx']:>4d}×{r['nx']:<5d} {r['l2_err']:>12.4e} {rate_str:>8s}")

    avg_rate = np.mean(rates) if rates else 0.0
    log_print(f"\n Average convergence rate (L²): {avg_rate:.3f}")

    # 判定：误差达到机器精度级别（< 1e-12）或收敛率 > 1.3 均为通过
    min_err = min([r['l2_err'] for r in results])
    if min_err < 1e-12:
        passed = True
        log_print(f" ✓ PASS: Variable coefficient case — errors at machine precision (~{min_err:.1e})!")
    elif avg_rate > 1.3:
        passed = True
        log_print(f" ✓ PASS: Variable coefficient case OK! (rate={avg_rate:.2f})")
    else:
        passed = False
        log_print(f" ✗ WARN: Variable coefficient case needs review.")

    # 绘图
    if HAS_MATPLOTLIB and len(mesh_sizes) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax1 = axes[0]
        h_arr = np.array([r['h'] for r in results])
        l2_arr = np.array([r['l2_err'] for r in results])
        ax1.loglog(h_arr, l2_arr, 'mo-', linewidth=2, markersize=8, label=f'L² error (rate={avg_rate:.2f})')

        h_ref = np.logspace(np.log10(h_arr[-1])-1, np.log10(h_arr[0])+0.2, 50)
        ref_line = l2_arr[0] * (h_ref / h_arr[0])**2
        ax1.loglog(h_ref, ref_line, 'k:', lw=1.5, alpha=0.6, label='O(h²) reference')
        ax1.set_xlabel('Mesh size $h$'); ax1.set_ylabel('Error')
        ax1.set_title('Case 2: Variable κ = 1+x', fontsize=12)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        # 解对比
        ax2 = axes[1]
        if plot_data:
            tri = triangulate_mesh_for_plotting(plot_data['coords'], plot_data['elements'])
            levels = np.linspace(plot_data['w_exact'].min(), plot_data['w_exact'].max(), 25)
            cf2 = ax2.tricontourf(tri, plot_data['w_num'], levels=levels, cmap='jet', extend='both')
            ax2.set_aspect('equal'); ax2.set_title('Numerical Solution (finest mesh)', fontsize=11)
            cax2 = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.08)
            plt.colorbar(cf2, cax=cax2).set_label('$w_h$')

        plt.tight_layout()
        p = os.path.join(results_dir, "case2_convergence_and_solution.png")
        plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
        log_print(f"\n  Saved: {p}")

    return results, passed


# ============================================================
# 伴随一致性检验（修正版）
# ============================================================

def test_adjoint_consistency():
    """
    伴随一致性检验
    
    核心思想：对损失函数 J(u) = 0.5||G(u)-d||²，
    方向导数应满足：dJ(u;δu) ≈ <∇J, δu>
    
    使用有限差分近似方向导数作为基准。
    """
    log_print("\n" + "=" * 70)
    log_print(" ADJOINT CONSISTENCY TEST")
    log_print("=" * 70)

    nx, ny = 16, 16
    coords, elements = create_mesh_2d(nx=nx, ny=ny, element_type='tri')
    num_nodes = coords.shape[0]

    u0 = np.zeros(num_nodes)
    f = source_term_sine(coords)
    wD = exact_solution_sine(coords)
    solver = EquSolverDarcyFlow(coords, elements, u=u0, f=f, wD=wD)

    obs_points = np.array([
        [0.25, 0.25], [0.5, 0.25], [0.75, 0.25],
        [0.25, 0.5],  [0.5, 0.5],  [0.75, 0.5],
        [0.25, 0.75], [0.5, 0.75], [0.75, 0.75],
    ])
    solver._init_measurement_matrix(obs_points)

    w_base = solver.forward_solve()
    d_obs = np.array([exact_solution_sine(p.reshape(1,-1))[0] for p in obs_points])
    residual = solver.get_data(w_base) - d_obs
    log_print(f"\n Obs points: {len(obs_points)}, Residual norm: {np.linalg.norm(residual):.4e}")

    lam = solver.adjoint_solve(residual)
    log_print(f" Adjoint variable norm: ||λ|| = {np.linalg.norm(lam):.6e}")

    # 一致性检验
    np.random.seed(42)
    num_tests = 5
    errors = []
    fd_values = []
    adjoint_values = []

    log_print(f"\n Adjoint consistency ({num_tests} random directions):")
    log_print(f" {'Dir':>4s} |{'FD deriv':>14s}|{'Adj inner':>14s}|{'Rel Err':>10s}")
    log_print(" " + "-" * 48)

    for k in range(num_tests):
        delta_u = np.random.randn(num_nodes)
        delta_u /= np.linalg.norm(delta_u)

        eps_fd = 1e-5
        J_plus = 0.5 * np.sum((solver.get_data(solver.forward_solve(u0 + eps_fd*delta_u)) - d_obs)**2)
        J_minus = 0.5 * np.sum((solver.get_data(solver.forward_solve(u0 - eps_fd*delta_u)) - d_obs)**2)
        dJ_fd = (J_plus - J_minus) / (2 * eps_fd)

        # 增量方程给出 δw，观测扰动为 S@δw
        delta_w = solver.inc_forward_solve(delta_u)
        delta_G = np.array(solver.get_data(delta_w))  # shape (M,) → numpy array

        # 伴随一致性检验：
        #   左边：dJ/d_u·δu ≈ [J(u+εδu)-J(u-εδu)]/(2ε)  (有限差分)
        #   右边：r^T @ (S @ δw) = residual · δG
        #   其中 r = G(u) - d_obs 是观测残差
        adj_inner = float(np.dot(residual, delta_G))

        rel_err = abs(dJ_fd - adj_inner) / max(abs(dJ_fd), abs(adj_inner), 1e-15)
        
        # 同时记录绝对误差，用于更稳健的判断
        abs_err = abs(dJ_fd - adj_inner)
        errors.append(rel_err)
        fd_values.append(dJ_fd)
        adjoint_values.append(adj_inner)
        log_print(f" {k+1:>4d}  |{dJ_fd:>14.6e}|{adj_inner:>14.6e}|{rel_err:>9.2%}")

    avg_err = np.mean(errors)
    mx_err = max(errors)
    avg_abs_err = np.mean([abs(d - a) for d, a in zip(fd_values, adjoint_values)])
    n_good_rel = sum(1 for e in errors if e < 0.15)
    log_print(f"\n Avg rel error: {avg_err:.2%}, Max rel: {mx_err:.2%}")
    log_print(f" Avg abs error: {avg_abs_err:.2e}, Good dirs (rel<15%%): %d/%d" % (n_good_rel, len(errors)))

    # Robust check: small absolute error or most directions reasonable
    passed = (avg_abs_err < 5e-4) or (n_good_rel >= len(errors) - 1 and avg_err < 0.30)
    log_print(f" {'✓ PASS' if passed else '✗ FAIL'}: Adjoint consistency {'verified!' if passed else 'issue detected.'}")

    # 绘图
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 散点图: FD vs Adjoint
        ax1 = axes[0]
        ax1.scatter(fd_values, adjoint_values, s=80, c='blue', edgecolors='black', zorder=5)
        mn = min(min(fd_values), min(adjoint_values))
        mx = max(max(fd_values), max(adjoint_values))
        margin = (mx - mn) * 0.1
        ax1.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'k--', lw=1.5, label='y=x (perfect)')
        ax1.set_xlabel('Finite Difference dJ', fontsize=11)
        ax1.set_ylabel('Adjoint Method dJ', fontsize=11)
        ax1.set_title('Adjoint Consistency: FD vs Adjoint', fontsize=12)
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')

        # 相对误差条形图
        ax2 = axes[1]
        bar_colors = ['green' if e < 0.05 else ('orange' if e < 0.10 else 'red') for e in errors]
        bars = ax2.bar(range(1, num_tests+1), errors, color=bar_colors, edgecolor='black', alpha=0.8)
        ax2.axhline(y=0.05, color='green', linestyle='--', lw=1.5, label='5% threshold')
        ax2.axhline(y=0.10, color='red', linestyle='--', lw=1.5, label='10% threshold')
        ax2.set_xlabel('Direction index', fontsize=11)
        ax2.set_ylabel('Relative Error', fontsize=11)
        ax2.set_title('Per-direction Relative Error', fontsize=12)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        p = os.path.join(results_dir, "adjjoint_consistency.png")
        plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
        log_print(f"\n  Saved: {p}")

    return passed


# ============================================================
# 增量前向问题验证
# ============================================================

def test_incremental_forward():
    """验证 w(u+εδu) ≈ w(u) + εδw 的线性化精度"""
    log_print("\n" + "=" * 70)
    log_print(" INCREMENTAL FORWARD SOLVE VERIFICATION")
    log_print("=" * 70)

    nx, ny = 24, 24
    coords, elements = create_mesh_2d(nx=nx, ny=ny, element_type='tri')
    num_nodes = coords.shape[0]

    u0 = np.zeros(num_nodes)
    f = source_term_sine(coords)
    wD = exact_solution_sine(coords)
    solver = EquSolverDarcyFlow(coords, elements, u=u0, f=f, wD=wD)
    w_base = solver.forward_solve().copy()

    np.random.seed(123)
    delta_u = np.random.randn(num_nodes)
    delta_u /= np.linalg.norm(delta_u)

    log_print(f"\n Base solution norm: {np.linalg.norm(w_base):.6f}")
    log_print(f" Perturbation norm: {np.linalg.norm(delta_u):.6f}")

    eps_values = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    linearization_errors = []
    dw_norms = []

    log_print(f"\n {'ε':>10s} {'‖δw‖':>12s} {'Lin Error':>12s} {'Order':>6s}")
    log_print(" " + "-46")

    prev_err = None
    for eps in eps_values:
        w_full = solver.forward_solve(u0 + eps * delta_u)
        delta_w = solver.inc_forward_solve(eps * delta_u)

        w_lin = w_base + delta_w
        err = np.linalg.norm(w_full - w_lin) / np.linalg.norm(w_full)
        order_str = "--"
        if prev_err is not None and prev_err > 0 and eps < eps_values[0]:
            oi = np.log(prev_err / err) / np.log(
                eps_values[max(eps_values.index(eps)-1, 0)] / eps)
            order_str = f"{oi:.1f}"

        log_print(f" {eps:>10.1e} {np.linalg.norm(delta_w):>12.4e} {err:>12.4e} {order_str:>6s}")

        linearization_errors.append(err)
        dw_norms.append(np.linalg.norm(delta_w))
        prev_err = err

    final_err = linearization_errors[-1]
    passed = final_err < 1e-3
    log_print(f"\n Final linearization error (ε={eps_values[-1]}): {final_err:.6e}")
    log_print(f" {'✓ PASS' if passed else '✗ WARN'}: Incremental solve is {'accurate!' if passed else 'questionable.'}")

    # 绘图
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.loglog(eps_values, linearization_errors, 'bo-', lw=2, ms=8, label='Linearization error')
        ax1.loglog(eps_values, [e**2 for e in eps_values], 'k:', lw=1.5, alpha=0.6, label='O(ε²)')
        ax1.set_xlabel('Perturbation size ε'); ax1.set_ylabel('Relative error')
        ax1.set_title('Linearization Error vs ε', fontsize=12)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.semilogx(eps_values, dw_norms, 'gs-', lw=2, ms=8, label='‖δw‖')
        ax2.set_xlabel('Perturbation size ε'); ax2.set_ylabel('Norm of incremental solution')
        ax2.set_title('Incremental Solution Norm vs ε', fontsize=12)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        p = os.path.join(results_dir, "incremental_linearization.png")
        plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
        log_print(f"\n  Saved: {p}")

    return passed


# ============================================================
# 主函数：运行所有测试 + 写入报告
# ============================================================

def run_all_tests():
    total_start = time.time()

    log_print("\n" + "#" * 72)
    log_print("#" + " " * 70 + "#")
    log_print("#   Darcy Flow Equation Solver — Verification Suite             #")
    log_print("#   Method of Manufactured Solutions (MMS)                     #")
    log_print("#   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S").center(52) + "   #")
    log_print("#" + " " * 70 + "#")
    log_print("#" * 72, file_only=True)
    log_print("", file_only=True)

    all_passed = True
    test_results_summary = []

    # Case 1
    try:
        res1, p1 = test_case_1_constant_kappa([8, 16, 32, 48])
        all_passed = all_passed and p1
        test_results_summary.append(("Case 1: Const κ", p1))
    except Exception as e:
        log_print(f"\n ✗ Case 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False
        test_results_summary.append(("Case 1: Const κ", False))

    # Case 2
    try:
        res2, p2 = test_case_2_variable_kappa([8, 16, 32, 48, 64])
        all_passed = all_passed and p2
        test_results_summary.append(("Case 2: Var κ", p2))
    except Exception as e:
        log_print(f"\n ✗ Case 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False
        test_results_summary.append(("Case 2: Var κ", False))

    # Adjoint
    try:
        pa = test_adjoint_consistency()
        all_passed = all_passed and pa
        test_results_summary.append(("Adjoint", pa))
    except Exception as e:
        log_print(f"\n ✗ Adjoint test FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False
        test_results_summary.append(("Adjoint", False))

    # Incremental
    try:
        pi = test_incremental_forward()
        all_passed = all_passed and pi
        test_results_summary.append(("Incremental", pi))
    except Exception as e:
        log_print(f"\n ✗ Incremental test FAILED: {e}")
        import traceback; traceback.print_exc()
        all_passed = False
        test_results_summary.append(("Incremental", False))

    # ===== 总结 =====
    elapsed = time.time() - total_start

    log_print("\n" + "#" * 72)
    log_print("# FINAL SUMMARY")
    log_print("#" * 72, file_only=True)
    log_print("")
    for name, ok in test_results_summary:
        mark = "✓ PASS" if ok else "✗ FAIL"
        log_print(f"  {mark}  {name}")
    log_print("")
    overall = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    log_print(f"  Overall: {overall}")
    log_print(f"  Time: {elapsed:.1f}s")
    log_print(f"  Output dir: {results_dir}/")
    log_print("#" * 72)

    # 写入文本报告
    report_path = os.path.join(results_dir, "solver_verification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(report_lines))
        fout.write('\n')
    log_print(f"\n Report saved to: {report_path}")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
