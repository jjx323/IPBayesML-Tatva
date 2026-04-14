# -*- coding: utf-8 -*-
"""
Newton-enhanced preconditioned Crank-Nicolson with Langevin (Newton-pCNL) MCMC

执行脚本：调用 core/sample 中定义的 Newton_pCNL 类进行采样。
结果存储在当前目录下的 RESULTS/ 文件夹中。
"""

## The following codes control the number of threads the numpy and scipy can employed.
## Sometimes, using more cores will lead to worser performance.
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')  # 强制 float64 双精度
nthreads = 1
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
from core.sample import Newton_pCNL
from core.plot import project, plot_fun2d
from DarcyFlow.misc import create_mesh_2d


# 数据和结果路径
data_folder = Path(darcyflow_dir) / "DATA"
results_folder = Path(script_dir) / "RESULTS"  # 结果存放在当前方法目录下
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

start_time = time.time()

## Load true function if available
true_fun_vals = None
true_fun_projected = None
if (data_folder/"fun_data.npy").exists():
    true_fun_vals = np.load(data_folder/"fun_data.npy")

## Set noise level and load data
noise_level = 0.05
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy") if (data_folder/"clean_data.npy").exists() else data["data"]

## Create mesh using tatva/scipy
nx = 50
coords, elements = create_mesh_2d(nx=nx, ny=nx, element_type='tri')
equ_solver = EquSolverDarcyFlow(coords, elements)

# 执行跨网格投影：将真实参数从数据生成网格投影到反演网格
if true_fun_vals is not None:
    source_coords = None
    if (data_folder/"coords.npy").exists():
        src_c = np.load(data_folder/"coords.npy")
        if src_c.ndim == 2:
            source_coords = np.array(src_c, dtype=np.float64)
    true_fun_projected = project(
        true_fun_vals, target_coords=coords, source_coords=source_coords
    )

## Generate prior distribution
params = {
    "theta": lambda x: 0.1 + 0.0*x[0] if hasattr(x, '__len__') else 0.1,
    "ax": lambda x: 0.5 + 0.0*x[0] if hasattr(x, '__len__') else 0.5,
    "mean": lambda x: 0.0*x[0] if hasattr(x, '__len__') else 0.0
}
prior = GaussianElliptic2(coords, elements, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)

## Newton-pCNL sampling
len_chain = np.int64(1e5)
sample_file = Path(results_folder/("Newton_pCNL_samples_" + str(noise_level)))
sample_file.mkdir(exist_ok=True, parents=True)

newton_pcnl = Newton_pCNL(
    model, dt=0.1, beta=0.01, reduce_chain=np.int64(1e4), save_path=sample_file
)
newton_pcnl.mode = "map_hessian"
newton_pcnl.optim_options["info_optim"] = True
newton_pcnl.optim_options["cg_max"] = 50
newton_pcnl.optim_options["max_iter"] = [100, 50]
newton_pcnl.optim_options["newton_method"] = 'bicgstab'
newton_pcnl.eigensystem_optims["num_eigval"] = 30
newton_pcnl.eigensystem_optims["cut_val"] = 0.01
newton_pcnl.eigensystem_optims["oversampling_factor"] = 20
newton_pcnl.eigensystem_optims["method"] = "scipy_eigsh"
newton_pcnl.eigensystem_optims["hessian_type"] = "linear_approximate"

figure_file = Path(results_folder/("Newton_pCNL_figures_" + str(noise_level)))
figure_file.mkdir(exist_ok=True, parents=True)

## Callback for monitoring and visualization
global num_
num_ = 0

class CallBack(object):
    def __init__(self, num_=0, coords=equ_solver.coords, elements=equ_solver.elements,
                 truth=true_fun_projected, save_path=figure_file, len_chain=len_chain):
        self.num_ = num_
        self.coords = coords
        self.elements = elements
        self.truth = truth
        self.save_path = save_path
        self.num_fre = 1000
        self.len_chain = len_chain
        self.phi = model.loss_res

    def callback_fun(self, params):
        # params = [uk, iter_num, accept_rate, accept_num]
        num = params[1]
        if num % self.num_fre == 0:
            print("-"*70)
            print('IterNum = %d/%d' % (num, self.len_chain), end='; ')
            print('AccRate = %4.4f percent' % (params[2]*100), end='; ')
            print('Phi = %4.4f' % self.phi(params[0]))

            # Plot current sample
            plot_fun2d(
                params[0], coords=self.coords, elements=self.elements,
                nx=200, show=False, 
                path=self.save_path/("fun_" + str(num) + ".png"), 
                grid_on=False, package="matplotlib"
            )

callback = CallBack()

print("Starting Newton-pCNL sampling ......")
print("Computing MAP estimate ......")
newton_pcnl.eval_map()
print("Computing eigensystem ......")
newton_pcnl.eval_eigsystem()
print("Starting sampling ......")
newton_pcnl.sampling(
    len_chain=len_chain, u0=None, callback=callback.callback_fun
)

## Plot the trace of sampling function u
path_samples = sample_file
num_total = np.int64(len(os.listdir(path_samples)))
num_start = 0
trace_u = []

for i in range(num_start, num_total):
    temp = np.load(path_samples/('sample_' + str(i) + '.npy'))
    for data_item in temp:
        trace_u.append(data_item[10])

plt.figure()
plt.plot(trace_u)
plt.savefig(results_folder/"trace_newton_pcnl.png")
plt.close()

end_time = time.time()
print(f"\nElapsed time = {end_time - start_time:.5f} seconds.")
print(f"Results saved to {results_folder}")
