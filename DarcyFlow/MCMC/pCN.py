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
from core.sample import pCN
from core.plot import project, plot_fun2d
from DarcyFlow.misc import create_mesh_2d


# 数据和结果路径
data_folder = Path(darcyflow_dir) / "DATA"
results_folder = Path(script_dir) / "RESULTS"  # 结果存放在当前方法目录下
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

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
    "theta": lambda x: 0.1 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.1,
    "ax": lambda x: 0.5 * np.ones(len(x)) if len(np.array(x).shape) > 1 else 0.5,
    "mean": lambda x: np.zeros(len(x)) if len(np.array(x).shape) > 1 else 0.0
}
prior = GaussianElliptic2(coords, elements, params)
noise = NoiseGaussianIID(len(data["data"]))
noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
model = ModelDarcyFlow(prior, equ_solver, noise, data)

len_chain = np.int64(50000)  # Reduced for testing
sample_file = Path(results_folder/("pCN_samples_" + str(noise_level)))
sample_file.mkdir(exist_ok=True, parents=True)
pcn = pCN(
    model, beta=0.01, reduce_chain=np.int64(5000), num_select=np.int64(100),
    save_path=sample_file
)
figure_file = Path(results_folder/("pCN_figures_" + str(noise_level)))
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
        self.num_fre = 10
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

print("Starting pCN sampling ......")
print("len_chain = ", len_chain)
pcn.sampling(len_chain=len_chain, callback=callback.callback_fun)

## Plot the trace of sampling function u
path_samples = sample_file
print(f"Sampling completed. Results saved to {results_folder}")
