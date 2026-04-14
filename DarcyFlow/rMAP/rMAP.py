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
import sys, os
# 设置正确的路径
current_file = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file)
darcyflow_dir = os.path.dirname(script_dir)  # DarcyFlow/
project_root = os.path.dirname(darcyflow_dir)   # IPBayesML-Tatva/

sys.path.insert(0, project_root)
sys.path.insert(0, darcyflow_dir)
os.chdir(project_root)

import numpy as np
from pathlib import Path
import time

from DarcyFlow.common import EquSolverDarcyFlow, ModelDarcyFlow
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.approximate_sample import rMAP
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
name = "local"
size = comm.size
rank = comm.rank

## Set the number of samples
num_samples = 500

## Set the data and results path (结果存放在当前方法目录下)
data_folder = Path(darcyflow_dir) / "data"
results_folder = Path(script_dir) / "RESULTS"
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

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
equ_solver = EquSolverDarcyFlow(coords, elements, degree=1)

## Here, we ideally use the same Gaussian measure as for generating the truth.
params = {
    "theta": lambda x: 0.1 + 0.0*x[0] if hasattr(x, '__len__') else 0.1,
    "ax": lambda x: 0.5 + 0.0*x[0] if hasattr(x, '__len__') else 0.5,
    "mean": lambda x: 0.0*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]) if hasattr(x, '__call__') else lambda x: 0.0
}
prior = GaussianElliptic2(coords, elements, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level * max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)

start_time = time.time()
rmap_sampler = rMAP(comm=comm, num_size=num_samples, model=model, comm_size=size)

if rank == 0:
    rmap_sampler.optim_options = {
        "max_iter": [500, 100], "init_val": None, "info_optim": True,
        "cg_max": 200, "newton_method": 'bicgstab', "grad_smooth_degree": 1e-2,
        "if_normalize_dd": True
    }  ## "newton_method" == cg_my, bicgstab, cg, cgs
    estimate_map = rmap_sampler.optimizing()
else:
    estimate_map = None

estimate_map = comm.bcast(estimate_map, root=0)

rmap_sampler.optim_options = {
    "max_iter": [500, 100], "info_optim": True,
    "cg_max": 200, "newton_method": "bicgstab", "grad_smooth_degree": 1e-2,
    "if_normalize_dd": True
}  ## "newton_method" == cg_my, bicgstab, cg, cgs
samples = rmap_sampler.sampling(init_vec=estimate_map)
end_time = time.time()

end_times = comm.gather(end_time, root=0)
start_times = comm.gather(start_time, root=0)

## Post processing
if rank == 0:
    start_time = min(start_times)
    end_time = max(end_times)
    print("Elapsed time: %.5fs" % (end_time - start_time))
    posterior_mean = np.mean(samples, axis=0)
    
    plot_fun2d(posterior_mean, coords=coords, elements=elements, nx=200, 
               show=False, path=results_folder/"posterior_mean.png", 
               grid_on=False, package="matplotlib")
    
    plot_fun2d(estimate_map, coords=coords, elements=elements, nx=200, 
               show=False, path=results_folder / "estimate_map.png", 
               grid_on=False, package="matplotlib")
    
    ## Save individual sample figures (存放到当前方法的 RESULTS 目录下)
    samples_folder = results_folder / "samples"
    samples_folder.mkdir(exist_ok=True, parents=True)
    num_samples_actual = samples.shape[0]
    for idx in range(num_samples_actual):
        plot_fun2d(samples[idx, :], coords=coords, elements=elements, nx=200, 
                   show=False, path=samples_folder/("sample_" + str(idx) + ".png"), 
                   grid_on=False, package="matplotlib")
    
    np.save(results_folder/"samples", samples)

    ## Calculate error with background truth
    true_fun_vals = None
    source_coords = None  # 数据生成网格坐标（用于跨网格投影）
    if (data_folder/"fun_data.npy").exists():
        true_fun_vals = np.load(data_folder/"fun_data.npy")
        if (data_folder/"coords.npy").exists():
            src_c = np.load(data_folder/"coords.npy")
            if src_c.ndim == 2:
                source_coords = np.array(src_c, dtype=np.float64)
    
    if true_fun_vals is not None:
        # 跨网格投影到反演网格空间
        if source_coords is not None and len(true_fun_vals) == source_coords.shape[0]:
            true_fun_projected = project(
                true_fun_vals, target_coords=coords, source_coords=source_coords
            )
        elif len(true_fun_vals) == coords.shape[0]:
            true_fun_projected = np.asarray(true_fun_vals, dtype=np.float64)
        else:
            dim = min(len(true_fun_vals), coords.shape[0])
            true_fun_projected = np.asarray(true_fun_vals[:dim], dtype=np.float64)
        
        error_L2, error_max, _ = error_compare(
            posterior_mean,
            true_fun_projected, coords
        )
        print("error_L2 = %.5f, error_max = %.5f" % (error_L2, error_max))

print(f"\nrMAP sampling completed! Results saved to {results_folder}")
