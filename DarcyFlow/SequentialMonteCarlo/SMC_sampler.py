# -*- coding: utf-8 -*-
"""
序贯蒙特卡洛 (Sequential Monte Carlo, SMC) 采样 - 执行脚本

本脚本使用 Newton-pCNL 作为 MCMC 移动步骤的采样器，
通过一系列中间分布逐步逼近目标后验分布。

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
from scipy.special import logsumexp
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
from core.approximate_sample import rMAP
from core.sample import Newton_pCNL, SMC
from core.plot import project, plot_fun2d
from DarcyFlow.misc import error_compare, create_mesh_2d


## Basic setup (MPI simulation for single process)
class DummyComm:
    """Dummy MPI communicator for single-process execution"""
    def __init__(self):
        self.rank = 0
        self.size = 1
    
    def gather(self, obj, root=0):
        return [obj]
    
    def bcast(self, obj, root=0):
        return obj

comm = DummyComm()
rank = comm.rank

## 数据和结果路径 (结果存放在当前方法目录下)
data_folder = Path(darcyflow_dir) / "DATA"
results_folder = Path(script_dir) / "RESULTS"
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

start_time = time.time()

## Load measured data
noise_level = 0.05
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy") if (data_folder/"clean_data.npy").exists() else data["data"]

## Construct prior, noise, equ_solver, and finally construct model
nx = 50
coords, elements = create_mesh_2d(nx=nx, ny=nx, element_type='tri')
equ_solver = EquSolverDarcyFlow(coords, elements)

params = {
    "theta": lambda x: 0.1 + 0.0*x[0] if hasattr(x, '__len__') else 0.1,
    "ax": lambda x: 0.5 + 0.0*x[0] if hasattr(x, '__len__') else 0.5,
    "mean": lambda x: 0.0*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]) if hasattr(x, '__len__') else 0.0
}
prior = GaussianElliptic2(coords, elements, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level * max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)

## SMC with Newton-pCNL
num_particles = 500
num_layers = 10

original_std_dev = model.noise.std_dev
smc = SMC(comm, model, num_particles)
smc.prepare()

for num in range(num_layers):
    print(f"\n{'='*60}")
    print(f"SMC Layer {num + 1}/{num_layers}")
    print('='*60)
    
    model.noise.set_parameters(model.noise.mean, original_std_dev/np.sqrt((num + 1)/num_layers))
    smc.resampling(potential_fun=model.loss_res)
    
    ## Set parameters of Newton-pCNL sampler for MCMC transition step
    sampler = Newton_pCNL(model, dt=0.005, beta=0.01, reduce_chain=np.int64(100))
    sampler.mode = "map_hessian"
    sampler.optim_options["max_iter"] = [0, 50]
    sampler.optim_options["grad_smooth_degree"] = 1e-1
    sampler.optim_options["cg_max"] = 50
    sampler.optim_options["newton_method"] = 'bicgstab'
    sampler.optim_options["info_optim"] = False
    sampler.eigensystem_optims["num_eigval"] = 20
    sampler.eigensystem_optims["method"] = "scipy_eigsh"
    sampler.eigensystem_optims["cut_val"] = 0.0
    sampler.eigensystem_optims["hessian_type"] = "linear_approximate"
    sampler.eigensystem_optims["oversampling_factor"] = 10
    
    smc.transition(sampler=sampler, len_chain=50, info_acc_rate=False)
    
    if rank == 0:
        print("weights: ", smc.weights)
        print("num_layer = %d" % num)

particles = smc.gather_samples()
end_time = time.time()
print(f"\nElapsed time = {end_time - start_time:.5f} seconds.")

if rank == 0:
    ## Calculate the posterior mean function
    post_mean = np.mean(particles, axis=0)
    
    plot_fun2d(
        post_mean, coords=equ_solver.coords, elements=equ_solver.elements,
        nx=200, show=False, path=results_folder / "post_mean_smc_newton_pcnl.png",
        grid_on=False, package="matplotlib"
    )
    np.save(results_folder/"samples_smc_newton_pcnl", particles)
    
    ## Save the figures of the samples
    samples_folder = results_folder / "samples_smc_newton_pcnl"
    samples_folder.mkdir(exist_ok=True, parents=True)
    
    for idx, particle in enumerate(particles):
        plot_fun2d(
            particle, coords=equ_solver.coords, elements=equ_solver.elements,
            nx=200, show=False, 
            path=samples_folder / ("sample_" + str(idx) + ".png"),
            grid_on=False, package="matplotlib"
        )
        
        if (idx + 1) % 50 == 0:
            print(f"Saved sample figure {idx+1}/{len(particles)}")

    ## Calculate the relative error of posterior mean with truth (if available)
    true_fun_projected = None
    if (data_folder/"fun_data.npy").exists():
        true_vals = np.load(data_folder/"fun_data.npy")
        # 加载源网格坐标（数据生成网格），用于跨网格投影
        source_coords = None
        if (data_folder/"coords.npy").exists():
            src_c = np.load(data_folder/"coords.npy")
            if src_c.ndim == 2:
                source_coords = np.array(src_c, dtype=np.float64)
        # 投影到反演网格
        true_fun_projected = project(
            true_vals, target_coords=coords, source_coords=source_coords
        )

    if true_fun_projected is not None:
        error_L2 = np.linalg.norm(post_mean - true_fun_projected) / (np.linalg.norm(true_fun_projected) + 1e-10)
        error_max = np.max(np.abs(post_mean - true_fun_projected)) / (np.max(np.abs(true_fun_projected)) + 1e-10)
        print("error_L2 = %.5f, error_max = %.5f" % (error_L2, error_max))
    
    print(f"All results saved to {results_folder}")
