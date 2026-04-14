## The following codes control the number of threads the numpy and scipy can employed.
## Sometimes, using more cores will lead to worser performance.
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')  # 强制 JAX 使用 float64 双精度
nthreads = 2
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
## The above must be added before all of the codes.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import time

# ============================================================
#  论文风格设置 (Computational Mathematics Paper Style)
# ============================================================
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'stix',          # 数学字体与正文一致
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 0.8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,                # 设为True需系统安装LaTeX
})

import sys
# 设置正确的路径
current_file = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file)
darcyflow_dir = os.path.dirname(script_dir)  # DarcyFlow/
project_root = os.path.dirname(darcyflow_dir)   # IPBayesML-Tatva/

sys.path.insert(0, project_root)
sys.path.insert(0, darcyflow_dir)
os.chdir(project_root)

from DarcyFlow.common import EquSolverDarcyFlow, ModelDarcyFlow
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.optimizer import NewtonCG, GradientDescent
from DarcyFlow.misc import error_compare, create_mesh_2d, construct_measure_matrix
from core.plot import project, plot_fun2d, plot_mesh


# ============================================================
#  方法选择
# ============================================================
# method = "NewtonCG"
method = "GradientDescent"
# method = "NewtonCG+GradDescent"

start_time = time.time()
# 数据文件夹位于 DarcyFlow/data
data_folder = Path(darcyflow_dir) / "DATA"
# 结果文件夹位于当前方法目录下
results_folder = Path(script_dir) / "RESULTS" / method
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

## Load true function if available (注意：可能来自不同分辨率网格)
true_fun_vals = None
source_coords = None  # 源网格坐标（用于跨网格投影）
if (data_folder/"fun_data.npy").exists():
    true_fun_vals = np.load(data_folder/"fun_data.npy")
    print(f"  true_fun_vals loaded: shape={true_fun_vals.shape}")
    # 尝试加载生成数据时使用的源网格坐标（用于正确投影到当前反演网格）
    if (data_folder/"coords.npy").exists():
        source_coords = np.load(data_folder/"coords.npy")
        if source_coords.ndim == 2:
            source_coords = np.array(source_coords, dtype=np.float64)
            print(f"  Source coords loaded: shape={source_coords.shape} "
                  f"(mesh for data generation)")
            if len(true_fun_vals) != source_coords.shape[0]:
                print(f"  WARNING: fun_data length ({len(true_fun_vals)}) != "
                      f"source coords nodes ({source_coords.shape[0]})")

## Set noise level and load data
noise_level = 0.05
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy") if (data_folder/"clean_data.npy").exists() else data["data"]

## Create mesh using tatva/scipy
nx = 20
coords, elements = create_mesh_2d(nx=nx, ny=nx, element_type='tri')
coords = np.array(coords, dtype=np.float64)
elements = np.array(elements, dtype=np.int32)

print(f"Mesh created: coords shape = {coords.shape}, elements shape = {elements.shape}")
print(f"Coords range: x=[{coords[:,0].min():.3f}, {coords[:,0].max():.3f}], y=[{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")

equ_solver = EquSolverDarcyFlow(coords, elements, degree=1)
print(f"EquSolver initialized: num_dofs = {equ_solver.num_dofs}")

## Generate prior distribution
params = {
    "theta": lambda x: 0.1 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.1,
    "ax": lambda x: 0.5 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.5,
    "mean": lambda x: np.zeros(len(x)) if len(np.array(x).shape) > 1 else 0.0
}
prior = GaussianElliptic2(coords, elements, params)

noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)
print(f"Model initialized: num_dofs = {model.num_dofs}, num_obs = {len(data['data'])}")

# ============================================================
#  将真实参数投影到反演网格空间（统一参考系）
#  核心思想：数据生成网格(如200×200)与反演网格(如20×20)不同，
#           在加载时一次性投影到反演网格，之后所有比较都在同一网格上进行。
# ============================================================
true_fun_projected = None   # 投影后的真解（在反演网格上，shape=(num_dofs_inv,)）
if true_fun_vals is not None:
    if source_coords is not None and len(true_fun_vals) == source_coords.shape[0]:
        # 情况A: 有源网格坐标 -> 跨网格插值投影 (200×200 -> 20×20)
        print(f"\n[Cross-mesh projection] {source_coords.shape[0]} nodes -> {coords.shape[0]} nodes")
        true_fun_projected = project(
            true_fun_vals, target_coords=coords, source_coords=source_coords
        )
        print(f"  => true_fun (on inversion mesh): shape={true_fun_projected.shape}")
    elif len(true_fun_vals) == coords.shape[0]:
        # 情况B: 维度一致 -> 同网格，直接使用
        true_fun_projected = np.asarray(true_fun_vals, dtype=np.float64)
        print(f"\n[Same mesh] true_fun shape={true_fun_projected.shape}, no projection needed")
    else:
        # 情况C: 无法正确匹配 -> 截断兜底
        dim = min(len(true_fun_vals), coords.shape[0])
        true_fun_projected = np.asarray(true_fun_vals[:dim], dtype=np.float64)
        print(f"\n[WARNING] Dimension mismatch! Truncated: {len(true_fun_vals)} -> {dim}")


# ============================================================
#  迭代数据记录器
# ============================================================
class IterationLogger:
    """记录每步迭代的损失、误差等数据，支持保存到txt"""
    
    def __init__(self, results_folder, method_name, has_true_solution=False):
        self.results_folder = results_folder
        self.method_name = method_name
        self.has_true_solution = has_true_solution
        self.iterations = []
        self.loss_history = []
        self.loss_relative_change = []     # |loss_k - loss_{k-1}| / |loss_{k-1}|
        self.gradient_norms = []
        self.step_sizes = []
        self.error_L2_history = []         # 每步相对真解的L2误差（如果有）
        self.error_max_history = []
        self.phase_info = []               # 记录阶段信息如 "GradDescent", "NewtonCG"
        self.start_time = time.time()
        
    def log(self, itr, loss, gradient_norm=None, step_size=None, 
            error_L2=None, error_max=None, phase=""):
        self.iterations.append(itr)
        self.loss_history.append(loss)
        self.gradient_norms.append(gradient_norm)
        self.step_sizes.append(step_size)
        self.error_L2_history.append(error_L2)
        self.error_max_history.append(error_max)
        self.phase_info.append(phase)
        
        if len(self.loss_history) > 1:
            rel_change = abs(loss - self.loss_history[-2]) / max(abs(self.loss_history[-2]), 1e-15)
        else:
            rel_change = np.nan
        self.loss_relative_change.append(rel_change)
    
    def save_to_txt(self):
        """保存迭代历史到txt文件"""
        filepath = self.results_folder / f"iteration_log_{self.method_name}.txt"
        
        header_lines = [
            f"# Optimization Method: {self.method_name}",
            f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Mesh: nx={nx}, num_dofs={equ_solver.num_dofs}",
            f"# Noise level: {noise_level}",
            f"# Total observations: {len(data['data'])}",
            f"# Prior params: theta={params['theta'](None):.3f}, ax={params['ax'](None):.3f}",
            f"# Elapsed time: {time.time() - self.start_time:.4f} seconds",
            f"#",
        ]
        
        columns = ["iter", "loss", "rel_change"]
        if self.has_true_solution:
            columns.extend(["error_L2", "error_max"])
        if any(g is not None for g in self.gradient_norms):
            columns.append("grad_norm")
        if any(s is not None for s in self.step_sizes):
            columns.append("step_size")
        if any(p for p in self.phase_info):
            columns.append("phase")
        
        header_lines.append("# " + "\t".join(columns))
        header_lines.append("#" + "-" * 100)
        
        with open(filepath, 'w') as f:
            f.write("\n".join(header_lines) + "\n")
            
            for i in range(len(self.iterations)):
                row = [
                    f"{self.iterations[i]:6d}",
                    f"{self.loss_history[i]:.8e}",
                    f"{self.loss_relative_change[i]:.6e}",
                ]
                if self.has_true_solution:
                    row.append(f"{self.error_L2_history[i]:.8e}" if self.error_L2_history[i] is not None else "N/A")
                    row.append(f"{self.error_max_history[i]:.8e}" if self.error_max_history[i] is not None else "N/A")
                if self.gradient_norms[i] is not None:
                    row.append(f"{self.gradient_norms[i]:.8e}")
                if self.step_sizes[i] is not None:
                    row.append(f"{self.step_sizes[i]:.8e}")
                if self.phase_info[i]:
                    row.append(self.phase_info[i])
                
                f.write("\t".join(row) + "\n")
        
        print(f"Iteration log saved to {filepath}")
        return filepath


logger = IterationLogger(results_folder, method, has_true_solution=(true_fun_vals is not None))


# ============================================================
#  辅助函数：计算每步的L2误差
# ============================================================
def compute_error(param, true_vals, coords_local=None, src_coords=None):
    """计算参数向量相对于真解的误差

    注意：推荐传入已投影到反演网格的 true_fun_projected，
         此时 true_vals 与 param 在同一空间，无需额外投影。

    Parameters
    ----------
    param : np.ndarray
        估计参数（在反演网格上）
    true_vals : np.ndarray
        真实参数值（优先传入已投影的 true_fun_projected）
    coords_local : np.ndarray, optional (保留向后兼容)
        反演网格坐标
    src_coords : np.ndarray, optional (保留向后兼容)
        数据生成网格坐标
    """
    if true_vals is None:
        return None, None

    # 如果维度一致，直接比较（true_vals 已在反演网格上的标准情况）
    if len(param) == len(true_vals):
        true_proj = np.asarray(true_vals, dtype=np.float64)
    elif src_coords is not None and coords_local is not None:
        # 向后兼容：需要做跨网格投影
        true_proj = project(
            true_vals, target_coords=coords_local, source_coords=src_coords
        )
    else:
        # 兜底：截断对齐
        dim = min(len(param), len(true_vals))
        true_proj = np.asarray(true_vals[:dim], dtype=np.float64)
        param = np.asarray(param[:dim], dtype=np.float64)

    err_L2 = np.linalg.norm(np.asarray(param, dtype=np.float64) - true_proj) / \
             (np.linalg.norm(true_proj) + 1e-15)
    err_max = np.max(np.abs(np.asarray(param, dtype=np.float64) - true_proj)) / \
              (np.max(np.abs(true_proj)) + 1e-15)
    return float(err_L2), float(err_max)


# ============================================================
#  优化主循环
# ============================================================

if method == "NewtonCG":
    optimizer = NewtonCG(model=model)
    max_iter = 100
    init_val = np.zeros(model.num_dofs)
    model.smoother.set_degree(1e-2)
    
    optimizer.re_init(init_val)
    loss_pre = model.loss()[0]
    print(f"\n{'='*60}")
    print(f"Method: Newton-CG Optimization")
    print(f"Init loss: {loss_pre:.6e}")
    print('='*60)
    
    for itr in range(max_iter):
        optimizer.if_pre_cond = True
        optimizer.descent_direction(cg_max=3, method='bicgstab')
        if optimizer.hessian_terminate_info != 0:
            print(f"  Hessian terminate info: {optimizer.hessian_terminate_info}")
        optimizer.step(method='armijo', show_step=False)
        if not optimizer.converged:
            break
        
        loss = model.loss()[0]
        grad_norm = np.linalg.norm(model.eval_grad_total(optimizer.mk))
        eL2, eMax = compute_error(optimizer.mk, true_fun_projected)

        logger.log(itr+1, loss, grad_norm, None, eL2, eMax, "NewtonCG")
        
        print(f"  iter = {itr+1:3d}/{max_iter}, loss = {loss:.6e}, "
              f"|grad| = {grad_norm:.4e}"
              + (f", err_L2 = {eL2:.4e}" if eL2 is not None else ""))
        
        if np.abs(loss - loss_pre) < 1e-3*loss:
            print(f"  Converged at iteration {itr+1} (relative change < 0.1%)")
            break
        loss_pre = loss

elif method == "GradientDescent":
    model.smoother.set_degree(1e-2)
    optimizer = GradientDescent(model=model)
    max_iter = 100
    init_val = np.zeros(model.num_dofs)
    optimizer.re_init(init_val)
    
    loss_pre = model.loss()[0]
    print(f"\n{'='*60}")
    print(f"Method: Gradient Descent Optimization")
    print(f"Init loss: {loss_pre:.6e}")
    print('='*60)
    
    for itr in range(max_iter):
        optimizer.descent_direction(model.smoother.smoothing)
        optimizer.step(method='armijo', show_step=False)
        if not optimizer.converged:
            break
        
        loss = model.loss()[0]
        grad_norm = np.linalg.norm(model.eval_grad_total(optimizer.mk))
        eL2, eMax = compute_error(optimizer.mk, true_fun_projected)

        logger.log(itr+1, loss, grad_norm, None, eL2, eMax, "GradDesc")
        
        print(f"  iter = {itr+1:3d}/{max_iter}, loss = {loss:.6e}, "
              f"|grad| = {grad_norm:.4e}"
              + (f", err_L2 = {eL2:.4e}" if eL2 is not None else ""))
        
        if np.abs(loss - loss_pre) < 1e-3 * loss:
            print(f"  Converged at iteration {itr+1} (relative change < 0.1%)")
            break
        loss_pre = loss

elif method == "NewtonCG+GradDescent":
    # ---- Phase 1: Gradient Descent warm-start ----
    model.smoother.set_degree(1e-2)
    optimizer = GradientDescent(model=model)
    gd_max_iter = 50
    init_val = np.zeros(model.num_dofs)
    optimizer.re_init(init_val)
    
    loss_pre = model.loss()[0]
    print(f"\n{'='*60}")
    print(f"Method: NewtonCG + Gradient Descent Hybrid")
    print(f"{'='*60}")
    print(f"\n[Phase 1: Gradient Descent Warm-Start]")
    print(f"Init loss: {loss_pre:.6e}")
    
    for itr in range(gd_max_iter):
        optimizer.descent_direction(model.smoother.smoothing)
        optimizer.step(method='armijo', show_step=False)
        if not optimizer.converged:
            break
        
        loss = model.loss()[0]
        grad_norm = np.linalg.norm(model.eval_grad_total(optimizer.mk))
        eL2, eMax = compute_error(optimizer.mk, true_fun_projected)

        logger.log(itr+1, loss, grad_norm, None, eL2, eMax, "Phase1_GradDesc")
        
        print(f"  iter = {itr+1:3d}/{gd_max_iter}, loss = {loss:.6e}, "
              f"|grad| = {grad_norm:.4e}"
              + (f", err_L2 = {eL2:.4e}" if eL2 is not None else ""))
        
        if np.abs(loss - loss_pre) < 1e-3 * loss:
            print(f"  Phase 1 converged at iteration {itr+1}")
            break
        loss_pre = loss
    
    estimated_init = optimizer.mk.copy()
    
    # ---- Phase 2: Newton-CG refinement ----
    optimizer = NewtonCG(model=model)
    newton_max_iter = 50
    
    init_val = estimated_init.copy()
    optimizer.re_init(init_val)
    
    loss_pre = model.loss()[0]
    print(f"\n[Phase 2: Newton-CG Refinement]")
    print(f"Init loss: {loss_pre:.6e}")
    
    base_itr = gd_max_iter
    for itr in range(newton_max_iter):
        optimizer.if_pre_cond = True
        optimizer.descent_direction(cg_max=50, method='bicgstab')
        if optimizer.hessian_terminate_info != 0:
            print(f"  Hessian terminate info: {optimizer.hessian_terminate_info}")
        optimizer.step(method='armijo', show_step=False)
        if not optimizer.converged:
            break
        
        loss = model.loss()[0]
        grad_norm = np.linalg.norm(model.eval_grad_total(optimizer.mk))
        eL2, eMax = compute_error(optimizer.mk, true_fun_projected)

        logger.log(base_itr + itr + 1, loss, grad_norm, None, eL2, eMax, "Phase2_NewtonCG")
        
        print(f"  iter = {base_itr+itr+1:3d}, loss = {loss:.6e}, "
              f"|grad| = {grad_norm:.4e}"
              + (f", err_L2 = {eL2:.4e}" if eL2 is not None else ""))
        
        if np.abs(loss - loss_pre) < 1e-3 * loss:
            print(f"  Phase 2 converged at iteration {base_itr + itr + 1}")
            break
        loss_pre = loss

else:
    raise TypeError("Unknown optimization method: {}".format(method))

# ============================================================
#  最终结果提取
# ============================================================
estimated_param = optimizer.mk.copy()
model.update_param(estimated_param, update_sol=True)
d_est = model.get_data(model.equ_solver.sol_forward)
final_loss = model.loss()[0]
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*60}")
print(f"OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"  Method:           {method}")
print(f"  Final Loss:       {final_loss:.8e}")
print(f"  Initial Loss:     {logger.loss_history[0]:.8e}")
print(f"  Total Iterations: {len(logger.loss_history)}")
print(f"  Elapsed Time:     {elapsed_time:.4f} s")

if true_fun_projected is not None:
    # true_fun_projected 已在反演网格上，直接比较
    final_err_L2, final_err_max = compute_error(estimated_param, true_fun_projected)
    init_err_L2, _ = compute_error(np.zeros_like(estimated_param), true_fun_projected)
    print(f"  Initial Error L2: {init_err_L2:.6e}")
    print(f"  Final Error L2:   {final_err_L2:.6e}")
    print(f"  Final Error Max:  {final_err_max:.6e}")
else:
    final_err_L2, final_err_max = None, None

print(f"  Result folder:    {results_folder}")

# ============================================================
#  保存迭代日志
# ============================================================
log_file = logger.save_to_txt()

# ============================================================
#  绘图：论文风格
# ============================================================

# ---------- Figure 1: Loss 收敛曲线 ----------
fig1, ax1 = plt.subplots(figsize=(7, 5))
iters = np.arange(1, len(logger.loss_history) + 1)
ax1.semilogy(iters, logger.loss_history, 'b-', linewidth=1.5, marker='o', 
             markersize=3, markevery=max(1, len(iters)//20), label=r'$J(u^{(k)})$')

ax1.set_xlabel('Iteration $k$', fontsize=12)
ax1.set_ylabel('Objective Function Value', fontsize=12)
ax1.set_title(f'Convergence History — {method}', fontsize=13)

# 标记不同阶段（如果有）
phase_changes = []
prev_phase = logger.phase_info[0] if logger.phase_info else ""
for i, p in enumerate(logger.phase_info):
    if p != prev_phase and p != "":
        phase_changes.append((i+1, p))
        prev_phase = p

for change_itr, phase_name in phase_changes:
    ax1.axvline(x=change_itr, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.text(change_itr + 0.5, ax1.get_ylim()[1]*0.9, phase_name.split('_')[-1],
             fontsize=9, rotation=90, va='top', alpha=0.7)

ax1.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.6)
ax1.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
fig1.savefig(results_folder / f"loss_convergence_{method}.pdf")
plt.close(fig1)


# ---------- Figure 2: 估计参数 vs 真实参数 (组合对比图) ----------
# 注意：estimated_param 和 true_fun_projected 现在都在同一反演网格空间（coords）
if true_fun_projected is not None:
    from scipy.interpolate import griddata

    # 确保 estimated_param 转为 numpy 数组并截取到反演网格维度
    est_param = np.asarray(estimated_param, dtype=np.float64)
    dim = min(len(est_param), len(true_fun_projected), coords.shape[0])
    error_field = np.abs(est_param[:dim] - true_fun_projected[:dim])

    fig2_combined, axes2c = plt.subplots(1, 3, figsize=(16, 4.5))

    xi = np.linspace(coords[:,0].min(), coords[:,0].max(), 150)
    yi = np.linspace(coords[:,1].min(), coords[:,1].max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)

    for idx, (data_arr, title_str, cmap_name) in enumerate([
        (true_fun_projected[:dim], '(a) True $u^\\dagger$', 'RdBu_r'),
        (est_param[:dim], f'(b) Estimated $\\tilde{{u}}$', 'RdBu_r'),
        (error_field, f'(c) Error (L$_2$={final_err_L2:.4e})', 'hot')
    ]):
        Zi = griddata((coords[:,0], coords[:,1]), data_arr, (Xi, Yi), method='linear')
        im = axes2c[idx].contourf(Xi, Yi, Zi, levels=30, cmap=cmap_name)
        axes2c[idx].set_title(title_str, fontsize=11)
        axes2c[idx].set_xlabel('$x_1$', fontsize=10)
        axes2c[idx].set_ylabel('$x_2$', fontsize=10)
        axes2c[idx].set_aspect('equal')
        plt.colorbar(im, ax=axes2c[idx], shrink=0.8)

    plt.tight_layout()
    fig2_combined.savefig(results_folder / f"estimate_vs_truth_{method}.pdf")
    plt.close(fig2_combined)

    # 单独保存三张独立高质量图（使用 plot_fun2d）
    plot_fun2d(
        true_fun_projected, coords=coords, elements=elements, nx=200,
        show=False, path=results_folder / "true_parameter.pdf",
        grid_on=False, package="matplotlib"
    )
    plot_fun2d(
        estimated_param, coords=coords, elements=elements, nx=200,
        show=False, path=results_folder / f"estimated_{method}.pdf",
        grid_on=False, package="matplotlib"
    )
    plot_fun2d(
        error_field, coords=coords, elements=elements, nx=200,
        show=False, path=results_folder / f"error_field_{method}.pdf",
        grid_on=False, package="matplotlib", cmap='hot'
    )

else:
    # 无真实解时单独绘制估计参数
    plot_fun2d(
        estimated_param, coords=coords, elements=elements, nx=200,
        show=False, path=results_folder / f"estimated_{method}.pdf",
        grid_on=False, package="matplotlib"
    )


# ---------- Figure 3: 数据拟合 (预测数据 vs 观测数据) ----------
fig3, ax3 = plt.subplots(figsize=(7, 5))
obs_indices = np.arange(len(d_est))
ax3.plot(obs_indices, d_est, 'b-', linewidth=1.2, label=r'$\mathcal{G}(\tilde{u})$ (Predicted)', alpha=0.9)
ax3.plot(obs_indices, data['data'], 'ro', markersize=2.5, label=r'$d^{\delta}$ (Observed)', alpha=0.6)

ax3.set_xlabel('Observation Index', fontsize=12)
ax3.set_ylabel('Value', fontsize=12)
ax3.set_title(f'Data Fitting Comparison — {method}', fontsize=13)
ax3.legend(loc='best', framealpha=0.9)
ax3.grid(True, linestyle='-', linewidth=0.3, alpha=0.6)

residual = np.linalg.norm(d_est - data['data']) / (np.linalg.norm(data['data']) + 1e-15)
ax3.text(0.03, 0.97, f'Relative Residual: {residual:.4e}', transform=ax3.transAxes,
         fontsize=10, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig3.savefig(results_folder / f"data_fitting_{method}.pdf")
plt.close(fig3)


# ---------- Figure 4: 误差收敛曲线 (如果有真解) ----------
if true_fun_projected is not None and all(e is not None for e in logger.error_L2_history):
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    
    valid_idx = [i for i, e in enumerate(logger.error_L2_history) if e is not None]
    valid_iters = [logger.iterations[i] for i in valid_idx]
    valid_err_L2 = [logger.error_L2_history[i] for i in valid_idx]
    valid_err_max = [logger.error_max_history[i] for i in valid_idx]
    
    ax4.semilogy(valid_iters, valid_err_L2, 'r-o', linewidth=1.5, markersize=3,
                 markevery=max(1, len(valid_iters)//20), label=r'$\|u^{(k)} - u^\dagger\|_{L_2}$')
    ax4.semilogy(valid_iters, valid_err_max, 'b--s', linewidth=1.2, markersize=3,
                 markevery=max(1, len(valid_iters)//20), label=r'$\|u^{(k)} - u^\dagger\|_\infty$')
    
    ax4.set_xlabel('Iteration $k$', fontsize=12)
    ax4.set_ylabel('Relative Error', fontsize=12)
    ax4.set_title(f'Error Convergence — {method}', fontsize=13)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.6)
    
    plt.tight_layout()
    fig4.savefig(results_folder / f"error_convergence_{method}.pdf")
    plt.close(fig4)


# ---------- Figure 5: 梯度范数变化 (如果记录了) ----------
if any(g is not None for g in logger.gradient_norms):
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    valid_grad = [(logger.iterations[i], g) 
                   for i, g in enumerate(logger.gradient_norms) if g is not None]
    grad_iters, grad_norms = zip(*valid_grad)
    
    ax5.semilogy(grad_iters, grad_norms, 'g-^', linewidth=1.5, markersize=3,
                  markevery=max(1, len(grad_iters)//15), label=r'$\|\nabla J(u^{(k)})\|$')
    
    ax5.set_xlabel('Iteration $k$', fontsize=12)
    ax5.set_ylabel('Gradient Norm', fontsize=12)
    ax5.set_title(f'Gradient Norm History — {method}', fontsize=13)
    ax5.legend(loc='upper right', framealpha=0.9)
    ax5.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.6)
    
    plt.tight_layout()
    fig5.savefig(results_folder / f"gradient_norm_{method}.pdf")
    plt.close(fig5)


# ============================================================
#  输出汇总文件
# ============================================================
summary_path = results_folder / f"summary_{method}.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  OPTIMIZATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  Method:              {method}\n")
    f.write(f"  Mesh Resolution:     {nx} x {nx}\n")
    f.write(f"  Number of DOFs:      {equ_solver.num_dofs}\n")
    f.write(f"  Number of Obs:       {len(data['data'])}\n")
    f.write(f"  Noise Level:         {noise_level}\n")
    f.write(f"  Prior theta:         {params['theta'](None)}\n")
    f.write(f"  Prior ax:            {params['ax'](None)}\n")
    f.write(f"  \n")
    f.write("-" * 70 + "\n")
    f.write("  RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Initial Loss:       {logger.loss_history[0]:.10e}\n")
    f.write(f"  Final Loss:         {final_loss:.10e}\n")
    f.write(f"  Loss Reduction:     {(1 - final_loss/logger.loss_history[0])*100:.4f}%\n")
    f.write(f"  Total Iterations:   {len(logger.loss_history)}\n")
    f.write(f"  Elapsed Time:       {elapsed_time:.4f} s\n")
    if final_err_L2 is not None:
        f.write(f"  Final Error L2:     {final_err_L2:.10e}\n")
        f.write(f"  Final Error Max:    {final_err_max:.10e}\n")
    f.write(f"  Data Residual:      {residual:.10e}\n")
    f.write(f"  \n")
    f.write("-" * 70 + "\n")
    f.write("  OUTPUT FILES\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Iteration Log:      {log_file.name if hasattr(log_file, 'name') else 'iteration_log.txt'}\n")
    f.write(f"  Summary File:       {summary_path.name}\n")
    f.write(f"  Figures (PDF):\n")
    f.write(f"    - loss_convergence_{method}.pdf\n")
    if true_fun_projected is not None:
        f.write(f"    - estimate_vs_truth_{method}.pdf\n")
        f.write(f"    - error_convergence_{method}.pdf\n")
        f.write(f"    - true_parameter.pdf\n")
        f.write(f"    - estimated_{method}.pdf\n")
        f.write(f"    - error_field_{method}.pdf\n")
    else:
        f.write(f"    - estimated_{method}.pdf\n")
    f.write(f"    - data_fitting_{method}.pdf\n")
    if any(g is not None for g in logger.gradient_norms):
        f.write(f"    - gradient_norm_{method}.pdf\n")
    f.write("=" * 70 + "\n")

print(f"\nSummary saved to {summary_path}")
print(f"All results saved to {results_folder}/")
