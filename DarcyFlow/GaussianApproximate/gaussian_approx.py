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
from core.approximate_sample import GaussianApproximate
from core.plot import project, plot_fun2d, plot_mesh
from DarcyFlow.misc import error_compare, construct_measure_matrix, create_mesh_2d


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
equ_solver = EquSolverDarcyFlow(coords, elements, degree=1)

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
else:
    true_fun_projected = None

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

## Step 1: Find MAP estimate using gradient descent
model.smoother.set_degree(1e-3)
optimizer = GradientDescent(model=model)
max_iter = 2000

init_val = np.zeros(model.num_dofs)
optimizer.re_init(init_val)

loss_pre = model.loss()[0]
print("Init loss (GradDescent): ", loss_pre)
for itr in range(max_iter):
    optimizer.descent_direction(model.smoother.smoothing)
    optimizer.step(method='armijo', show_step=False)
    if optimizer.converged == False:
        break
    loss = model.loss()[0]
    print("iter = %2d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
    if np.abs(loss - loss_pre) < 1e-5 * loss:
        print("Iteration stoped at iter = %d" % itr)
        break
    loss_pre = loss
estimated_init = optimizer.mk.copy()

## Step 2: Refine MAP estimate using Newton-CG
model.smoother.set_degree(1e-3)
optimizer = NewtonCG(model=model)
max_iter = 500

init_val = estimated_init.copy()
optimizer.re_init(init_val)

loss_pre = model.loss()[0]
print("Init loss (NewtonCG): ", loss_pre)
for itr in range(max_iter):
    optimizer.descent_direction(cg_max=100, method='bicgstab')
    print("Hessian terminate info: ", optimizer.hessian_terminate_info)
    optimizer.step(method='armijo', show_step=False)
    if optimizer.converged == False:
        break
    loss = model.loss()[0]
    print("iter = %2d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
    if np.abs(loss - loss_pre) < 1e-5 * loss:
        print("Iteration stoped at iter = %d" % itr)
        break
    loss_pre = loss

estimated_param = optimizer.mk.copy()

## Step 3: Calculate posterior variance via Laplace approximation
start_time = time.time()
laplace_approximate = GaussianApproximate(model)
laplace_approximate.eval_eigensystem(num_eigval=40, method="scipy_eigsh")
laplace_approximate.set_mean(estimated_param)
end_time = time.time()
print("Calculate eigen-system consumes %.5fs" % (end_time-start_time))

## Sample from approximate posterior
sample_vec = laplace_approximate.generate_sample()

## Plot results
plot_fun2d(estimated_param, coords=coords, elements=elements, nx=200, 
           show=False, path=results_folder/"estimated_fun.jpeg", 
           grid_on=False, package="matplotlib")
plot_fun2d(sample_vec, coords=coords, elements=elements, nx=200, 
           show=False, path=results_folder/"sample_fun.jpeg", 
           grid_on=False, package="matplotlib")

## Plot eigenvalues
plt.figure()
plt.plot(np.log(laplace_approximate.eigval))
plt.xlabel('Eigenvalue index')
plt.ylabel('log(Eigenvalue)')
plt.title('Posterior Eigenvalues')
plt.savefig(results_folder/"eigvals1.jpeg")
plt.close()

## Calculate pointwise variance along diagonal
xx = np.linspace(0, 1, 100)
coor = np.zeros((len(xx), 2))
coor[:, 0] = xx
coor[:, 1] = xx

var_xx_prior = model.prior.pointwise_variance_field(coor) if hasattr(model.prior, 'pointwise_variance_field') else None
start_time = time.time()
var_xx = laplace_approximate.pointwise_variance_field(coor)
end_time = time.time()
print("Calculate the pointwise variance field consumes %.5fs" % (end_time-start_time))

SS = construct_measure_matrix(coords, coor)
vals = SS @ sample_vec

plt.figure()
plt.plot(vals, linestyle="solid", label="estimated", color="red")
if true_fun_projected is not None:
    SST = construct_measure_matrix(coords, coor)
    plt.plot(SST @ true_fun_projected,
             linestyle="solid", label="true", color="blue")
plt.plot(vals + 2*np.diag(var_xx), linestyle="dashed", color="orange")
plt.plot(vals - 2*np.diag(var_xx), linestyle="dashed", color="orange")
plt.legend()
plt.xlabel('Position along diagonal')
plt.ylabel('Value')
plt.title('Posterior Uncertainty')
plt.savefig(results_folder/"var_slice.jpeg")
plt.close()

## Generate samples based on Gaussian approximate
samples_all = []
for idx in range(1000):
    samples_all.append(laplace_approximate.generate_sample())
samples_all = np.array(samples_all)
np.save(results_folder/"samples_all.npy", samples_all)

print(f"\nGaussian approximation completed! Results saved to {results_folder}")
print(f"Generated {len(samples_all)} posterior samples")
