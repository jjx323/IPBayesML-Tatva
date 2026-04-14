# IPBayesML-Tatva

**IPBayesML-Tatva** is a Python library for Bayesian inference in infinite-dimensional inverse problems governed by partial differential equations (PDEs). It is built on top of the [Tatva / JAX-FEM](https://github.com/tianjuxue/jax-am) framework and uses JAX / SciPy sparse backends, replacing the earlier FEniCSx/PETSc dependency of the predecessor library [IPBayesML-FEniCSx09](https://github.com/jjx323/IPBayesML-FEniCSx09).

The library targets researchers in **statistical inverse problems**, **Bayesian computation**, and **scientific machine learning** who need dimension-robust MCMC samplers and MAP-based approximations for PDE-constrained problems.

---

## Features

- **Dimension-robust MCMC samplers** (`core/sample.py`):
  - `pCN` — preconditioned Crank–Nicolson (Cotter et al., 2013)
  - `pCNL` — pCN with Langevin gradient correction
  - `Newton_pCNL` — Hessian-preconditioned pCNL via low-rank spectral decomposition at the MAP point
  - `VanillaMCMC` — standard (pseudo-)MCMC on the full posterior
  - `SMC` — Sequential Monte Carlo with systematic resampling
- **Gaussian posterior approximation** (`core/approximate_sample.py`): low-rank Laplace approximation via eigendecomposition of the data-misfit Hessian
- **Optimization** (`core/optimizer.py`): gradient descent and Newton-CG for MAP estimation, with Armijo line search
- **Modular model interface** (`core/model.py`): abstract base classes `EquSolverBase` and `ModelBase` for plugging in any PDE forward solver
- **Gaussian priors** (`core/probability.py`): Matérn-type covariance operators assembled with JAX-sparse matrices
- **Noise models** (`core/noise.py`): Gaussian observational noise
- **Demonstration problem** (`DarcyFlow/`): full Bayesian inversion pipeline for a 2-D Darcy flow coefficient identification problem, including data generation, MAP estimation, MCMC sampling, Gaussian approximation, SMC, and rMAP

---

## Mathematical Background

The library targets the Bayesian formulation of the parameter identification problem:

$$\mu^y(\mathrm{d}u) \propto \exp(-\Phi(u;\, y))\, \mu_0(\mathrm{d}u)$$

where:
- $u$ is the unknown parameter field (e.g., log-permeability),
- $y$ are the noisy observations,
- $\Phi(u; y) = \frac{1}{2}\|y - \mathcal{G}(u)\|^2_{\Gamma^{-1}}$ is the negative log-likelihood (data-misfit potential),
- $\mu_0 = \mathcal{N}(0, \mathcal{C}_0)$ is a Gaussian prior with covariance operator $\mathcal{C}_0$.

All MCMC algorithms are designed to be **mesh-invariant** (well-defined in the function-space limit), following the framework of Cotter, Roberts, Stuart & White (2013) and Dashti & Stuart (2017).

---

## Repository Structure

```
IPBayesML-Tatva/
├── core/                         # Core library modules
│   ├── model.py                  # Abstract base classes: EquSolverBase, ModelBase
│   ├── sample.py                 # MCMC samplers: pCN, pCNL, Newton_pCNL, VanillaMCMC, SMC
│   ├── approximate_sample.py     # Laplace / Gaussian posterior approximation
│   ├── optimizer.py              # MAP optimizers: GradientDescent, NewtonCG
│   ├── probability.py            # Gaussian prior distributions
│   ├── noise.py                  # Noise models
│   ├── jax_backend.py            # JAX-based sparse linear algebra utilities
│   ├── linear_eq_solver.py       # Linear equation solvers (CG, BiCGStab, ...)
│   ├── misc.py                   # Miscellaneous utilities
│   └── plot.py                   # Plotting helpers
│
├── DarcyFlow/                    # Example: 2-D Darcy flow inverse problem
│   ├── common.py                 # Shared PDE solver, prior, model for Darcy flow
│   ├── generate_data.py          # Synthetic data generation
│   ├── misc.py                   # Darcy-specific utilities (mesh, mass matrix, ...)
│   ├── DATA/                     # Saved synthetic datasets
│   ├── MCMC/                     # pCN / pCNL / Newton_pCNL sampling scripts
│   ├── GaussianApproximate/      # Laplace approximation scripts
│   ├── OptimizationMethods/      # MAP estimation scripts
│   ├── SequentialMonteCarlo/     # SMC scripts
│   ├── VariationalInference-ongoing/  # (Work in progress) VI methods
│   └── rMAP/                     # Randomized MAP sampling
│
└── test/                         # Unit and integration tests
```

---

## Installation

### Prerequisites

- Python >= 3.9
- [JAX](https://github.com/google/jax) (CPU or GPU build)
- [jax-am / Tatva](https://github.com/tianjuxue/jax-am) (for FEM mesh and assembly)
- NumPy, SciPy, Matplotlib

### Install dependencies

```bash
# Install JAX (CPU)
pip install --upgrade "jax[cpu]"

# Install remaining dependencies
pip install numpy scipy matplotlib
```

> **GPU users**: Follow the [JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html) before installing other packages.

### Clone the repository

```bash
git clone https://github.com/jjx323/IPBayesML-Tatva.git
cd IPBayesML-Tatva
```

No additional `pip install` step is required; simply add the repository root to your `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/IPBayesML-Tatva:$PYTHONPATH"
```

---

## Quick Start: Darcy Flow Example

### Step 1 — Generate synthetic data

```bash
cd DarcyFlow
python generate_data.py
```

This solves the forward Darcy problem on a 2-D mesh, adds Gaussian noise to point observations, and saves the dataset to `DarcyFlow/DATA/`.

### Step 2 — Compute the MAP estimate

```bash
cd DarcyFlow/OptimizationMethods
python run_map.py
```

### Step 3 — Run pCN sampling

```bash
cd DarcyFlow/MCMC
python run_pCN.py
```

### Step 4 — Compute Gaussian (Laplace) approximation

```bash
cd DarcyFlow/GaussianApproximate
python run_gaussian_approx.py
```

---

## Implementing a Custom Problem

To apply the library to a new PDE-constrained inverse problem, subclass `EquSolverBase` and `ModelBase`:

```python
from core.model import EquSolverBase, ModelBase

class MyEquSolver(EquSolverBase):
    def __init__(self, mesh_coords, mesh_elements):
        super().__init__()
        self.coords = mesh_coords
        self.elements = mesh_elements
        self.num_dofs = mesh_coords.shape[0]

    def forward_solve(self, u=None):
        # Solve your PDE given parameter field u
        ...

    def adjoint_solve(self, vec, u=None):
        # Solve the adjoint PDE
        ...

class MyModel(ModelBase):
    def _init_measurement_matrix(self, coordinates):
        # Build observation operator S such that d = S @ w
        ...

    def loss_res(self, u=None):
        # Return the data-misfit potential Φ(u)
        ...

    def eval_grad_res(self, u):
        # Return gradient ∇_u Φ(u) via adjoint method
        ...

    def eval_hessian_res(self, u_hat):
        # Return Hessian-vector product H_res @ u_hat
        ...
```

Then pass your model to any sampler:

```python
from core.sample import pCN

sampler = pCN(model=my_model, beta=0.1)
sampler.sampling(len_chain=10000)
samples = sampler.chain
```

---

## MCMC Samplers Reference

| Class | Algorithm | Reference |
|---|---|---|
| `pCN` | Preconditioned Crank–Nicolson | Cotter et al. (2013), §4.1 |
| `pCNL` | pCN + Langevin gradient term | Cotter et al. (2013), §4.3 |
| `Newton_pCNL` | Hessian-preconditioned pCNL | Extension of Cotter et al. (2013) |
| `VanillaMCMC` | Random-walk MH on full posterior | — |
| `SMC` | Sequential Monte Carlo | Dashti & Stuart (2017), §5.3 |

All chain-based samplers share the following interface:

```python
sampler.sampling(
    len_chain=50000,   # number of MCMC steps
    u0=init_vec,       # optional starting point (np.ndarray)
    callback=fn,       # optional callable(state) called each step
)
print(sampler.acc_rate)   # acceptance rate
samples = sampler.chain   # list of accepted/current samples
```

Samples can be saved to disk in batches by setting `reduce_chain` and `save_path`:

```python
sampler = pCN(model, beta=0.05, reduce_chain=1000, save_path="./results/")
```

---

## Known Issues

The following bugs and limitations have been identified in `core/sample.py`:

| Severity | Location | Description |
|---|---|---|
| 🔴 Critical | `pCNL.sampling` | `self.loss()` called without passing `y` — acceptance ratio is computed at the wrong point |
| 🔴 Critical | `Newton_pCNL.sampling` | Same missing-argument issue as above |
| 🔴 Critical | `Newton_pCNL` (`every_step_hessian` mode) | `map_estimate` is `None` when `u0=None`; causes runtime error in `eval_eigsystem` |
| 🟡 Warning | `pCNL.rho` | Inner product structure of gradient correction term may not satisfy detailed balance exactly |
| 🟡 Warning | `Newton_pCNL.eval_eigsystem` | Missing `break` after convergence check causes redundant eigensolves |
| 🔵 Minor | `ModelBase.__init__` | No guard when `equ_solver` lacks `coords` / `elements`; passes `None` to JAX and crashes |

Contributions and bug fixes are very welcome — please open an issue or pull request.

---

## References

1. Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). **MCMC methods for functions: modifying old algorithms to make them faster.** *Statistical Science*, 28(3), 424–446.
2. Dashti, M., & Stuart, A. M. (2017). **The Bayesian approach to inverse problems.** In *Handbook of Uncertainty Quantification*, Springer.
3. Bui-Thanh, T., Ghattas, O., Martin, J., & Stadler, G. (2013). **A computational framework for infinite-dimensional Bayesian inverse problems Part I: The linearized case.** *SIAM Journal on Scientific Computing*, 35(6), A2494–A2523.

---

## License

This project is released for academic and research use. Please contact the author for licensing inquiries.

---

## Author

**Junxiong Jia** — researcher in statistical inverse problems and Bayesian machine learning.  
GitHub: [@jjx323](https://github.com/jjx323)
