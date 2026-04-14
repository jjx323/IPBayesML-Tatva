# -*- coding: utf-8 -*-
"""
Darcy 流逆问题数据生成器

本模块用于:
1. 创建真实参数场（解析函数或随机采样）
2. 求解正问题得到"干净"观测数据
3. 添加高斯噪声生成带噪声的观测数据
4. 保存所有数据供后续使用
5. 生成可视化图像（真实参数场、干净/带噪数据、误差分布、总览图）

数据格式:
- fun_data.npy: 真实参数场的 DOF 值
- measure_coordinates.npy: 观测点坐标
- clean_data.npy: 干净观测数据
- noisy_data_XX.npy: 不同噪声水平的带噪数据

注意：tatva 版本使用 numpy 数组格式，而非 FEniCSx 格式。
数据输出目录: DarcyFlow/DATA/
图像输出目录: DarcyFlow/DATA/figures/
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict

import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合脚本运行
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec

# ============================================================
#  路径设置 (与所有 DarcyFlow 脚本一致)
# ============================================================
import sys, os

## The following codes control the number of threads the numpy and scipy can employed.
os.environ.setdefault('JAX_ENABLE_X64', '1')  # 强制 float64 双精度
nthreads = 2
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)

current_file = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file)
darcyflow_dir = script_dir   # generate_data.py 就在 DarcyFlow/ 目录下
project_root = os.path.dirname(darcyflow_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, darcyflow_dir)
os.chdir(project_root)


# ============================================================
#  数据生成函数
# ============================================================

def generate_true_parameter(
    coords: np.ndarray,
    param_type: str = 'analytic',
    **kwargs
) -> np.ndarray:
    """
    生成真实参数场 u(x)
    
    Parameters:
    -----------
    coords : np.ndarray, shape=(num_points, dim)
        节点坐标
    param_type : str
        参数类型:
        - 'analytic': 解析函数
        - 'random': 从先验分布随机采样
        - 'smooth': 光滑随机函数
        
    Returns:
    --------
    u_true : np.ndarray, shape=(num_points,)
        真实参数值
    """
    x = coords[:, 0]
    y = coords[:, 1] if coords.shape[1] > 1 else x
    
    if param_type == 'analytic':
        # 使用解析表达式（默认示例）
        func_form = kwargs.get('func_form', 'sinusoidal')
        
        if func_form == 'sinusoidal':
            # 正弦波叠加
            u_true = (
                1.0 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y) +
                0.5 * np.sin(4*np.pi*x) * np.sin(3*np.pi*y) +
                0.3
            )
            
        elif func_form == 'polynomial':
            # 多项式
            u_true = (
                1.0 + 
                0.5*x + 0.3*y +
                0.8*x**2 + 0.6*y**2 +
                0.4*x*y
            )
            
        elif func_form == 'gaussian_bump':
            # 高斯峰
            cx, cy = kwargs.get('center', (0.5, 0.5))
            sigma = kwargs.get('sigma', 0.15)
            amplitude = kwargs.get('amplitude', 1.0)
            
            r2 = (x - cx)**2 + (y - cy)**2
            u_true = amplitude * np.exp(-r2 / (2*sigma**2))
            
        elif func_form == 'layered':
            # 分层结构（模拟地质层）
            u_true = np.zeros_like(x)
            u_true[y < 0.33] = 0.5
            u_true[(y >= 0.33) & (y < 0.66)] = 1.5
            u_true[y >= 0.66] = 2.5
            
            # 平滑过渡
            transition_width = 0.05
            for i in range(len(y)):
                if abs(y[i] - 0.33) < transition_width:
                    blend = (y[i] - 0.33 + transition_width) / (2*transition_width)
                    u_true[i] = 0.5 + (1.5 - 0.5) * blend
                elif abs(y[i] - 0.66) < transition_width:
                    blend = (y[i] - 0.66 + transition_width) / (2*transition_width)
                    u_true[i] = 1.5 + (2.5 - 1.5) * blend
                    
        else:
            raise ValueError(f"Unknown analytic function form: {func_form}")
            
    elif param_type == 'random':
        # 从先验采样
        prior = kwargs.get('prior')
        if prior is None:
            raise ValueError("Must provide prior distribution for random sampling")
        u_true = prior.generate_sample()
        
    elif param_type == 'smooth_random':
        # 光滑随机场（低频成分为主）
        n_modes = kwargs.get('n_modes', 10)
        u_true = np.zeros_like(x)
        
        for kx in range(1, n_modes+1):
            for ky in range(1, n_modes+1):
                amp = 1.0 / ((kx**2 + ky**2)**0.75)  # 衰减谱
                phase = np.random.uniform(0, 2*np.pi)
                u_true += amp * np.sin(2*np.pi*kx*x + phase) * \
                         np.cos(2*np.pi*ky*y + phase/2)
        
        # 归一化到合理范围
        u_true = (u_true - u_true.min()) / (u_true.max() - u_true.min() + 1e-12)
        u_true = 0.5 + 2.0 * u_true  # 缩放到 [0.5, 2.5]
        
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
        
    return u_true


def generate_observation_coords(
    domain_bounds: tuple = (0, 0, 1, 1),
    num_obs: int = 25,
    pattern: str = 'grid'
) -> np.ndarray:
    """
    生成观测点坐标
    
    Parameters:
    -----------
    domain_bounds : tuple
        定义域边界 [xmin, ymin, xmax, ymax]
    num_obs : int
        观测点数量
    pattern : str
        点分布模式:
        - 'grid': 规则网格
        - 'random': 随机均匀分布
        - 'halton': 准随机 Halton 序列
        - 'boundary': 边界点
        - 'mixed': 混合模式
        
    Returns:
    --------
    obs_coords : np.ndarray, shape=(num_obs, 2)
        观测点坐标
    """
    xmin, ymin, xmax, ymax = domain_bounds
    
    if pattern == 'grid':
        # 规则网格
        nx = ny = int(np.sqrt(num_obs))
        num_actual = nx * ny
        
        x_grid = np.linspace(xmin, xmax, nx)
        y_grid = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        obs_coords = np.column_stack([xx.ravel(), yy.ravel()])
        
    elif pattern == 'random':
        # 完全随机
        obs_x = np.random.uniform(xmin, xmax, num_obs)
        obs_y = np.random.uniform(ymin, ymax, num_obs)
        obs_coords = np.column_stack([obs_x, obs_y])
        
    elif pattern == 'halton':
        # Halton 低差异序列（更好的空间覆盖）
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=2, scramble=True)
            sample = sampler.random(num_obs)
            obs_coords = sample * np.array([xmax-xmin, ymax-ymin]) + np.array([xmin, ymin])
        except ImportError:
            print("scipy.stats.qmc not available, falling back to random")
            return generate_observation_coords(domain_bounds, num_obs, pattern='random')
            
    elif pattern == 'boundary':
        # 仅在边界上
        n_per_side = num_obs // 4
        remainder = num_obs % 4
        
        points = []
        
        # 底边
        bottom = np.column_stack([
            np.linspace(xmin, xmax, n_per_side),
            np.full(n_per_side, ymin)
        ])
        points.append(bottom)
        
        # 右边
        right = np.column_stack([
            np.full(n_per_side, xmax),
            np.linspace(ymin, ymax, n_per_side)
        ])
        points.append(right)
        
        # 顶边
        top = np.column_stack([
            np.linspace(xmax, xmin, n_per_side),
            np.full(n_per_side, ymax)
        ])
        points.append(top)
        
        # 左边
        left = np.column_stack([
            np.full(n_per_side, xmin),
            np.linspace(ymax, ymin, n_per_side)
        ])
        points.append(left)
        
        obs_coords = np.vstack(points)[:num_obs]
        
    elif pattern == 'mixed':
        # 混合：内部网格 + 边界点
        n_internal = int(num_obs * 0.7)
        n_boundary = num_obs - n_internal
        
        internal = generate_observation_coords(
            domain_bounds, n_internal, pattern='random'
        )
        
        boundary = generate_observation_coords(
            domain_bounds, n_boundary, pattern='boundary'
        )
        
        obs_coords = np.vstack([internal, boundary])
        
    else:
        raise ValueError(f"Unknown observation pattern: {pattern}")
        
    # 截断到正确数量
    return obs_coords[:num_obs]


def add_noise_to_data(
    clean_data: np.ndarray,
    noise_level: float = 0.05,
    noise_type: str = 'gaussian',
    seed: Optional[int] = None
) -> np.ndarray:
    """添加噪声到观测数据"""
    if seed is not None:
        np.random.seed(seed)
        
    data_range = max(abs(clean_data.min()), abs(clean_data.max()), 1e-10)
    
    if noise_type == 'gaussian':
        std_dev = noise_level * data_range
        noise = np.random.randn(len(clean_data)) * std_dev
        noisy_data = clean_data + noise
        
    elif noise_type == 'uniform':
        amplitude = noise_level * data_range
        noise = np.random.uniform(-amplitude, amplitude, len(clean_data))
        noisy_data = clean_data + noise
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
        
    return noisy_data


def save_dataset(
    data_folder,
    coords: np.ndarray,
    elements: np.ndarray,
    u_true: np.ndarray,
    obs_coords: np.ndarray,
    clean_data: np.ndarray,
    noisy_data_dict: Dict[float, np.ndarray],
    fun_info: dict
):
    """保存完整数据集到文件"""
    data_path = Path(data_folder)
    data_path.mkdir(exist_ok=True, parents=True)
    
    # 保存节点信息
    np.save(data_path / "coords", coords)
    np.save(data_path / "elements", elements)
    
    # 保存真实解
    np.save(data_path / "fun_data", u_true)
    
    with open(data_path / "fun_type_info.txt", 'w') as f:
        f.write(fun_info.get('fun_type', 'Lagrange'))
    with open(data_path / "fun_degree_info.txt", 'w') as f:
        f.write(str(fun_info.get('fun_degree', 1)))
        
    # 保存观测信息
    np.save(data_path / "measure_coordinates", obs_coords)
    np.save(data_path / "clean_data", clean_data)
    
    # 保存不同噪声水平的数据
    for noise_level, noisy in noisy_data_dict.items():
        filename = f"noisy_data_{noise_level}.npy"
        np.save(data_path / filename, noisy)
        
    # 保存元信息
    meta_info = {
        'num_dofs': len(u_true),
        'num_observations': len(obs_coords),
        'noise_levels': list(noisy_data_dict.keys()),
        'domain': fun_info.get('domain', [0, 0, 1, 1])
    }
    
    np.save(data_path / "meta_info", meta_info)
    
    print(f"Dataset saved to: {data_path}")
    print(f"  DOFs: {meta_info['num_dofs']}")
    print(f"  Observations: {meta_info['num_observations']}")
    print(f"  Noise levels: {meta_info['noise_levels']}")


# ============================================================
#  可视化函数
# ============================================================

def plot_dataset(
    data_folder,
    coords: np.ndarray,
    elements: np.ndarray,
    u_true: np.ndarray,
    obs_coords: np.ndarray,
    clean_data: np.ndarray,
    noisy_data_dict: Dict[float, np.ndarray],
):
    """
    生成数据集可视化图像并保存到 data_folder/figures/

    生成图像:
    - true_parameter.png      : 真实参数场二维色图（带观测点标注）
    - observation_curves.png  : 干净数据与各噪声水平曲线对比（按观测点索引）
    - noise_distribution.png  : 各噪声水平误差直方图对比
    - overview.png            : 总览图（真实参数场 + 曲线对比 + 误差分布）
    """
    data_path = Path(data_folder)
    fig_path = data_path / "figures"
    fig_path.mkdir(exist_ok=True, parents=True)

    # 构建三角剖分（用于 tripcolor）
    triang = tri.Triangulation(coords[:, 0], coords[:, 1], elements)

    noise_levels = list(noisy_data_dict.keys())
    n_noisy = len(noise_levels)
    vmin = u_true.min()
    vmax = u_true.max()
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, n_noisy))

    # 观测点索引（x 轴），按与原点距离排序，让曲线更平滑有规律
    obs_index = np.arange(len(clean_data))
    sort_order = np.argsort(obs_coords[:, 0] * 10 + obs_coords[:, 1])
    idx_sorted    = obs_index[sort_order]
    clean_sorted  = clean_data[sort_order]
    noisy_sorted  = {lv: nd[sort_order] for lv, nd in noisy_data_dict.items()}

    # ------------------------------------------------------------------ #
    # 图1：真实参数场（二维色图）
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    tcf = ax.tripcolor(triang, u_true, shading='gouraud', cmap='RdYlBu_r',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(tcf, ax=ax, label='u(x)')
    ax.scatter(
        obs_coords[:, 0], obs_coords[:, 1],
        c='k', marker='x', s=40, linewidths=1.2,
        label=f'Obs. points (n={len(obs_coords)})'
    )
    ax.set_title('True Parameter Field', fontsize=13, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path / "true_parameter.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path / 'true_parameter.png'}")

    # ------------------------------------------------------------------ #
    # 图2：一维散点对比图
    #   均匀抽取 n_sample 个点，每个噪声水平一个子图，共享 x 轴
    #   蓝色实心圆 = clean，彩色 × = noisy，灰色竖线连接两者体现偏移
    #   点数自适应：最多取全部观测点，上限 200
    # ------------------------------------------------------------------ #
    n_sample   = min(200, len(clean_data))
    sample_pos = np.linspace(0, len(idx_sorted) - 1, n_sample, dtype=int)

    s_pos   = sample_pos                       # x 轴坐标（等间距）
    s_clean = clean_sorted[sample_pos]
    s_noisy = {lv: nd[sample_pos] for lv, nd in noisy_sorted.items()}

    # 点数较多时缩小 marker，竖线也更细
    ms      = max(8,  int(300 / n_sample))     # marker size
    lw_stem = max(0.4, 1.5 - n_sample / 200)  # stem linewidth

    fig, axes_sc = plt.subplots(
        n_noisy, 1,
        figsize=(min(18, 6 + n_sample * 0.06), 3.5 * n_noisy),
        sharex=True
    )
    if n_noisy == 1:
        axes_sc = [axes_sc]
    fig.subplots_adjust(hspace=0.28)

    for i, ((level, s_nd), color) in enumerate(zip(s_noisy.items(), palette)):
        ax = axes_sc[i]

        # 竖线（stem）：连接 clean 与 noisy
        for xp, cv, nv in zip(s_pos, s_clean, s_nd):
            ax.plot([xp, xp], [cv, nv],
                    color='gray', linewidth=lw_stem, alpha=0.55, zorder=1)

        # clean 散点
        ax.scatter(s_pos, s_clean,
                   color='steelblue', s=ms, zorder=3,
                   label='Clean', edgecolors='white', linewidths=0.4)
        # noisy 散点
        ax.scatter(s_pos, s_nd,
                   color=color, marker='x', s=ms * 1.2,
                   linewidths=max(0.8, lw_stem * 1.8),
                   zorder=4, label=f'Noisy  {level*100:.0f}%')

        ax.set_ylabel('Value', fontsize=9)
        ax.set_title(
            f'noise = {level*100:.0f}%   '
            f'max|err| = {np.max(np.abs(s_nd - s_clean)):.4f}   '
            f'std = {(s_nd - s_clean).std():.4f}',
            fontsize=10, fontweight='bold', loc='left'
        )
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlim(-1, n_sample)

    axes_sc[-1].set_xlabel(
        f'Sample index  ({n_sample} points uniformly drawn, sorted by x then y)',
        fontsize=10
    )
    fig.suptitle(f'1-D Scatter: Clean vs Noisy  —  {n_sample} Points',
                 fontsize=13, fontweight='bold', y=1.005)
    fig.savefig(fig_path / "scatter_1d_noise.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fig_path / 'scatter_1d_noise.png'}")

    print(f"\n✓ All figures saved to: {fig_path}")


# ============================================================
#  主入口函数
# ============================================================

def create_synthetic_dataset(
    output_dir = "data",
    nx: int = 50,
    ny: int = 50,
    param_type: str = 'analytic',
    func_form: str = 'sinusoidal',
    num_observations: int = 49,
    obs_pattern: str = 'grid',
    noise_levels: list = [0.01, 0.03, 0.05, 0.10],
    seed: int = 42
):
    """
    创建完整的合成数据集

    这是主要的数据生成入口函数。

    Parameters:
    -----------
    output_dir : str or Path
        输出目录
    nx, ny : int
        网格分辨率
    param_type : str
        参数类型 ('analytic', 'random', 'smooth_random')
    func_form : str
        解析函数形式
    num_observations : int
        观测点数量
    obs_pattern : str
        观测点分布模式
    noise_levels : list
        噪声水平列表
    seed : int
        随机种子
    """
    np.random.seed(seed)
    
    print("="*60)
    print("Creating Synthetic Dataset for Darcy Flow Inverse Problem")
    print("="*60)
    
    # 1. 生成网格
    print("\n[1/5] Generating mesh...")
    from DarcyFlow.misc import create_mesh_2d
    coords, elements = create_mesh_2d(nx=nx, ny=ny)
    print(f"  Nodes: {len(coords)}, Elements: {len(elements)}")
    
    # 2. 生成真实参数场
    print("\n[2/5] Generating true parameter field...")
    
    gen_kwargs = {'func_form': func_form}
    if param_type == 'random':
        from core.probability import GaussianElliptic2
        prior_params = {
            "theta": lambda x: 0.1 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.1,
            "ax": lambda x: 0.5 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.5,
            "mean": lambda x: np.zeros(len(x)) if len(np.array(x).shape) > 1 else 0.0,
        }
        prior = GaussianElliptic2(coords, elements, prior_params)
        gen_kwargs['prior'] = prior
        print(f"  Built GaussianElliptic2 prior for random sampling")
    
    u_true = generate_true_parameter(
        coords,
        param_type=param_type,
        **gen_kwargs
    )
    print(f"  Parameter range: [{u_true.min():.4f}, {u_true.max():.4f}]")
    
    # 3. 生成观测点并计算干净数据
    print("\n[3/5] Computing observations...")
    obs_coords = generate_observation_coords(
        num_obs=num_observations,
        pattern=obs_pattern
    )
    
    from DarcyFlow.misc import construct_measure_matrix
    from DarcyFlow.common import EquSolverDarcyFlow
    print("\n[3/5] Solving forward PDE with true parameter...")
    equ_solver = EquSolverDarcyFlow(coords, elements, u=u_true)
    S = construct_measure_matrix(coords, obs_coords, elements=elements)
    equ_solver._init_measurement_matrix(obs_coords)

    w_true = equ_solver.forward_solve()
    clean_data = np.array(equ_solver.S @ w_true)   # d = S·w_true
    
    print(f"  Observation points: {num_observations}")
    print(f"  Clean data range: [{clean_data.min():.4f}, {clean_data.max():.4f}]")
    
    # 4. 添加不同噪声
    print("\n[4/5] Adding noise...")
    noisy_data_dict = {}
    for level in noise_levels:
        noisy = add_noise_to_data(clean_data, noise_level=level)
        noisy_data_dict[level] = noisy
        print(f"  Noise level {level}: "
              f"|data-noise| max = {np.max(np.abs(noisy-clean_data)):.4f}")
    
    # 5. 保存数据
    print("\n[5/5] Saving dataset and generating figures...")
    save_dataset(
        data_folder=output_dir,
        coords=coords,
        elements=elements,
        u_true=u_true,
        obs_coords=obs_coords,
        clean_data=clean_data,
        noisy_data_dict=noisy_data_dict,
        fun_info={
            'func_form': func_form,
            'fun_type': 'Lagrange',
            'fun_degree': 1,
            'domain': [0, 0, 1, 1]
        }
    )

    # 生成可视化图像
    plot_dataset(
        data_folder=output_dir,
        coords=coords,
        elements=elements,
        u_true=u_true,
        obs_coords=obs_coords,
        clean_data=clean_data,
        noisy_data_dict=noisy_data_dict,
    )
    
    print("\n✓ Dataset creation complete!")
    print("="*60)
    
    return {
        'coords': coords,
        'elements': elements,
        'u_true': u_true,
        'obs_coords': obs_coords,
        'clean_data': clean_data,
        'noisy_data': noisy_data_dict
    }


if __name__ == "__main__":
    # 数据输出目录: DarcyFlow/DATA/
    data_output_dir = Path(darcyflow_dir) / "DATA"
    data_output_dir.mkdir(exist_ok=True, parents=True)

    dataset = create_synthetic_dataset(
        output_dir=str(data_output_dir),
        nx=200,
        ny=200,
        param_type='random',
        func_form='sinusoidal',
        num_observations=200,
        obs_pattern='random',
        noise_levels=[0.01, 0.05, 0.10],
        seed=42
    )