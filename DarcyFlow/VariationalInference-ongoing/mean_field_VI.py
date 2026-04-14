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
from core.optimizer import NewtonCG
from core.plot import project, plot_fun2d
from DarcyFlow.misc import error_compare, create_mesh_2d


start_time = time.time()
# 数据和结果路径 (结果存放在当前方法目录下)
data_folder = Path(darcyflow_dir) / "DATA"
results_folder = Path(script_dir) / "RESULTS"
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

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

## Create mesh
nx = 50
coords, elements = create_mesh_2d(nx=nx, ny=nx, element_type='tri')
equ_solver = EquSolverDarcyFlow(coords, elements, degree=1)

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

## Eigen-decomposition of the prior covariance operator
model.prior.eval_eigensystem(
    num_eigval=50, method="scipy_eigsh", oversampling_factor=20, low_val=0.0
)

print(f"Prior eigensystem computed. Time elapsed: {time.time()-start_time:.2f}s")

## Mean-field Variational Inference would continue here...
## Using mean-field Gaussian approximation q(u) = N(m, diag(s^2))
## where m and s are optimized to minimize KL(q || p)

## Placeholder for VI optimization loop
## In full implementation, this would optimize ELBO
num_params = model.num_dofs
mean_vi = np.zeros(num_params)
log_std_vi = np.zeros(num_params) - 2.0  # Initialize small std

vi_max_iter = 100
elbo_history = []

print("\nMean-field Variational Inference (simplified)")
print("="*60)
for itr in range(vi_max_iter):
    ## Simplified ELBO computation
    ## Full implementation requires reparameterization trick and gradient estimation
    
    # Monte Carlo estimate of ELBO using reparameterization
    n_samples_mc = 10
    elbo_estimate = 0.0
    
    for s in range(n_samples_mc):
        # Reparameterization: u = m + s * epsilon, epsilon ~ N(0,I)
        eps = np.random.randn(num_params)
        u_sample = mean_vi + np.exp(log_std_vi) * eps
        
        # Log joint probability
        log_joint = model.compute_log_joint(u_sample)
        
        # Entropy of q: 0.5 * sum(log(2*pi*e*s^2))
        entropy = 0.5 * np.sum(np.log(2*np.pi*np.e) + 2*log_std_vi)
        
        elbo_estimate += log_joint + entropy
    
    elbo_estimate /= n_samples_mc
    elbo_history.append(elbo_estimate)
    
    # Simple gradient ascent on ELBO (using finite differences)
    if itr < vi_max_iter - 1:
        eps_grad = 1e-3
        grad_mean = np.zeros(num_params)
        grad_log_std = np.zeros(num_params)
        
        for d in range(min(num_params, 10)):  # Limit dimensionality for speed
            # Gradient w.r.t. mean[d]
            mean_vi[d] += eps_grad
            elbo_plus = model.compute_log_joint(mean_vi + np.exp(log_std_vi)*np.random.randn(num_params))
            mean_vi[d] -= eps_grad
            grad_mean[d] = (elbo_plus - elbo_estimate) / eps_grad
            
            # Gradient w.r.t. log_std[d]
            log_std_vi[d] += eps_grad
            elbo_plus = model.compute_log_joint(mean_vi + np.exp(log_std_vi)*np.random.randn(num_params))
            log_std_vi[d] -= eps_grad
            grad_log_std[d] = (elbo_plus - elbo_estimate) / eps_grad
        
        # Update parameters
        lr = 0.01
        mean_vi[:10] += lr * grad_mean[:10]
        log_std_vi[:10] += lr * grad_log_std[:10]
    
    if itr % 10 == 0:
        print("VI Iter %3d/%d, ELBO = %.6f" % (itr+1, vi_max_iter, elbo_estimate))

## Save results
plot_fun2d(mean_vi, coords=coords, elements=elements, nx=200, 
           show=False, path=results_folder/"vi_mean_field.png",
           grid_on=False, package="matplotlib", title="VI Posterior Mean")

std_vi = np.exp(log_std_vi)
plot_fun2d(std_vi, coords=coords, elements=elements, nx=200,
           show=False, path=results_folder/"vi_std_field.png",
           grid_on=False, package="matplotlib", title="VI Posterior Std")

## Plot ELBO convergence
plt.figure()
plt.plot(elbo_history)
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.title('ELBO Convergence')
plt.savefig(results_folder/"vi_elbo_convergence.png")
plt.close()

end_time = time.time()
print("\nTotal VI computation time = %.5fs" % (end_time - start_time))
print(f"Results saved to {results_folder}")
