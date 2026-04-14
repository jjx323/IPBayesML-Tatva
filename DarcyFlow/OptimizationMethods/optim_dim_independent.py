# -*- coding: utf-8 -*-
"""
维度独立的优化方法 - 执行脚本

本脚本实现针对 Darcy 流问题的维度独立优化策略：
将高维参数空间分解为多个低维子空间，在每个子空间中独立进行优化。

结果存储在当前目录下的 RESULTS/ 文件夹中。
"""

## The following codes control the number of threads the numpy and scipy can employed.
## Sometimes, using more cores will lead to worser performance.
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')  # 强制 float64 双精度
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
import time

import sys, os
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
from DarcyFlow.misc import error_compare, create_mesh_2d
from core.plot import plot_fun2d


# 数据和结果路径
data_folder = Path(darcyflow_dir) / "DATA"
results_folder = Path(script_dir) / "RESULTS"  # 结果存放在当前方法目录下
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

start_time = time.time()

# method_type = "gradient_descent"
method_type = "newton_cg"

## Load true function if available
true_fun_vals = None
if (data_folder/"fun_data.npy").exists():
    true_fun_vals = np.load(data_folder/"fun_data.npy")

## Set noise level and load data
noise_level = 0.05
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy") if (data_folder/"clean_data.npy").exists() else data["data"]

## Test different mesh sizes for dimension-independent analysis
nxs = [100, 200, 300, 400]
err_list = []
max_iter = 200

for nx in nxs:
    print(f"\n{'='*60}")
    print(f"Testing mesh size: nx = {nx}")
    print('='*60)
    
    ## Create mesh
    coords, elements = create_mesh_2d(nx=nx, ny=nx, element_type='tri')
    equ_solver = EquSolverDarcyFlow(coords, elements)
    
    ## Generate prior distribution
    params = {
        "theta": lambda x: 0.1 + 0.0*x[0] if hasattr(x, '__len__') else 0.1,
        "ax": lambda x: 0.1 + 0.0*x[0] if hasattr(x, '__len__') else 0.1,
        "mean": lambda x: 0.0*x[0] if hasattr(x, '__len__') else 0.0
    }
    prior = GaussianElliptic2(coords, elements, params)
    noise = NoiseGaussianIID(len(data["data"]))
    noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
    model = ModelDarcyFlow(prior, equ_solver, noise, data)
    
    ## Set optimizer
    if method_type == "newton_cg":
        optimizer = NewtonCG(model=model)
    elif method_type == "gradient_descent":
        optimizer = GradientDescent(model=model)
    else:
        raise TypeError("method_type should be newton_cg or gradient_descent")

    ## Initialize optimizer
    init_val = np.zeros(model.num_dofs)
    optimizer.re_init(init_val)

    loss_pre = model.loss()[0]
    errors = []
    
    for itr in range(max_iter):
        if method_type == "newton_cg":
            optimizer.descent_direction(cg_max=50, method='bicgstab')
        elif method_type == "gradient_descent":
            smoother_op = lambda x: model.smoother.smoothing(x, degree=0.1)
            optimizer.descent_direction(smoother_op)
        
        optimizer.step(method='armijo', show_step=False)
        
        if not optimizer.converged:
            break
            
        loss = model.loss()[0]
        print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        
        if np.abs(loss - loss_pre) < 1e-3*loss:
            print("Iteration stopped at iter = %d" % itr)
            break
        loss_pre = loss
        
        if itr == 0:
            m_optimizer_pre = optimizer.mk.copy()
        else:
            m_optimizer = optimizer.mk.copy()
            error_L2 = np.linalg.norm(m_optimizer - m_optimizer_pre) / (np.linalg.norm(m_optimizer_pre) + 1e-10)
            errors.append(error_L2)
            m_optimizer_pre = m_optimizer.copy()

    err_list.append(np.array(errors))
    print(f"nx={nx} completed, error history length: {len(errors)}")


## Plot dimension-independent convergence comparison
plt.figure(figsize=(10, 6))
min_len = len(min(err_list, key=len))
for idx, nx in enumerate(nxs):
    plt.plot(err_list[idx][:min_len], label="dim = " + str(nx))
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Step Error (L2)")
plt.title("Dimension-Independent Convergence: " + method_type)
plt.grid(True, alpha=0.3)
plt.savefig(results_folder/("dim_independent_" + method_type + ".png"), dpi=150, bbox_inches='tight')
plt.close()

end_time = time.time()
print(f"\nElapsed time = {end_time - start_time:.5f} seconds.")
print(f"Results saved to {results_folder}")
