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
from core.plot import project, plot_fun2d
from DarcyFlow.misc import error_compare, create_mesh_2d


start_time = time.time()
# 数据和结果路径 (结果存放在当前方法目录下)
data_folder = Path(darcyflow_dir) / "data"
results_folder = Path(script_dir) / "RESULTS"
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

## Load true function if available
true_fun_vals = None
true_projected = None
if (data_folder/"fun_data.npy").exists():
    true_fun_vals = np.load(data_folder/"fun_data.npy")
    # 加载源网格坐标（数据生成网格），用于跨网格投影
    source_coords = None
    if (data_folder/"coords.npy").exists():
        src_c = np.load(data_folder/"coords.npy")
        if src_c.ndim == 2:
            source_coords = np.array(src_c, dtype=np.float64)
    # 投影到反演网格（在创建coords之后进行）
    pass  # 延迟到coords创建后执行

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

# 执行跨网格投影：将真实参数从数据生成网格投影到反演网格
if true_fun_vals is not None:
    source_coords = None
    if (data_folder/"coords.npy").exists():
        src_c = np.load(data_folder/"coords.npy")
        if src_c.ndim == 2:
            source_coords = np.array(src_c, dtype=np.float64)
    true_projected = project(
        true_fun_vals, target_coords=coords, source_coords=source_coords
    )
else:
    true_projected = None

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

## SVGD: Stein Variational Gradient Descent
## Reference: Liu & Wang (2016). "Stein Variational Gradient Descent"

print("Starting SVGD ...")
print("="*60)

num_particles = 50  # Number of particles to represent the distribution
max_iter = 100      # Maximum number of SVGD iterations
step_size = 0.001   # Learning rate

# Initialize particles randomly from prior
particles = np.array([prior.generate_sample() for _ in range(num_particles)])
n_dims = model.num_dofs

# Store history for visualization
particle_history = []

print(f"Number of particles: {num_particles}")
print(f"Number of dimensions per particle: {n_dims}")

for iteration in range(max_iter):
    start_iter = time.time()
    
    ## Compute kernel matrix and its gradient
    # Using RBF kernel: k(x,y) = exp(-||x-y|^2 / (2*h^2))
    h = median_distance = np.median([
        np.linalg.norm(particles[i] - particles[j])
        for i in range(num_particles) for j in range(i+1, num_particles)
    ]) if num_particles > 1 else 1.0
    h = max(h, 1e-6)  # Avoid division by zero
    
    # Kernel matrix K_ij = k(x_i, x_j)
    diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]  # (N, N, D)
    dist_sq = np.sum(diff**2, axis=2)  # (N, N)
    K = np.exp(-dist_sq / (2 * h**2))  # (N, N)
    
    # Gradient of kernel: grad_k(x_i, x_j) = k(x_i,x_j) * (x_j - x_i) / h^2
    grad_K = -K[:, :, np.newaxis] * diff / (h**2)  # (N, N, D)
    
    ## Compute score function (gradient of log posterior)
    scores = np.zeros_like(particles)
    for i in range(num_particles):
        # Score = grad log pi(u) = grad_prior(u) + grad_likelihood(u)
        scores[i] = model.eval_grad_total(particles[i])
    
    ## SVGD update rule:
    ## phi(x_i) = (1/N) sum_j [k(x_i,x_j) grad_log_pi(x_j) + grad_x_i k(x_i,x_j)]
    svgd_update = np.zeros_like(particles)
    for i in range(num_particles):
        phi = (1.0/num_particles) * (
            np.sum(K[i, :, np.newaxis] * scores, axis=0) +
            np.sum(grad_K[i, :, :], axis=0)
        )
        svgd_update[i] = phi
    
    # Update particles
    particles += step_size * svgd_update
    
    # Monitor progress
    if iteration % 10 == 0 or iteration == max_iter - 1:
        # Compute average log posterior
        avg_log_post = np.mean([model.compute_log_joint(p) for p in particles[:5]])  # Use subset for speed
        print("SVGD iter %3d/%d, avg_log_posterior ≈ %.4f, time = %.2fs" % (
            iteration+1, max_iter, avg_log_post, time.time()-start_iter))
        
        # Store snapshot
        particle_history.append(np.mean(particles, axis=0).copy())

print("\nSVGD optimization completed!")

## Visualize results
posterior_mean = np.mean(particles, axis=0)
posterior_std = np.std(particles, axis=0)

plot_fun2d(posterior_mean, coords=coords, elements=elements, nx=200,
           show=False, path=results_folder/"svgd_posterior_mean.png",
           grid_on=False, package="matplotlib", title="SVGD Posterior Mean")

plot_fun2d(posterior_std, coords=coords, elements=elements, nx=200,
           show=False, path=results_folder/"svgd_posterior_std.png",
           grid_on=False, package="matplotlib", title="SVGD Posterior Std")

## Plot particle trajectories at selected points
plt.figure(figsize=(10, 6))
history_array = np.array(particle_history)
selected_points = [0, n_dims//4, n_dims//2, 3*n_dims//4]  # Select some DOF indices
for pt_idx in selected_points:
    if pt_idx < n_dims:
        plt.plot(history_array[:, pt_idx], label=f'DOF {pt_idx}')
plt.xlabel('Snapshot (every 10 iterations)')
plt.ylabel('Value')
plt.title('SVGD Particle Trajectories')
plt.legend()
plt.savefig(results_folder/"svgd_trajectories.png")
plt.close()

## Save final particles
np.save(results_folder/"svgd_particles", particles)
np.save(results_folder/"svgd_mean", posterior_mean)

## Compute error if true solution available
if true_projected is not None:
    error_L2, error_max, _ = error_compare(
        true_projected,
        posterior_mean, coords
    )
    print("error_L2 = %.5f, error_max = %.5f" % (error_L2, error_max))

end_time = time.time()
print("\nTotal SVGD computation time = %.5fs" % (end_time - start_time))
print(f"Results saved to {results_folder}")
