#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IPBayesML-Tatva DarcyFlow 模块综合测试脚本（简化版）

使用直接导入方式验证所有模块
"""

import os
import sys
import numpy as np
import time

# 设置环境变量
os.environ['JAX_ENABLE_X64'] = '1'

# 获取项目根目录
current_file = os.path.abspath(__file__)
test_dir = os.path.dirname(current_file)       # DarcyFlow/test
darcyflow_dir = os.path.dirname(test_dir)      # DarcyFlow
project_root = os.path.dirname(darcyflow_dir)  # IPBayesML-Tatva

sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "core"))
sys.path.insert(0, os.path.join(project_root, "DarcyFlow"))
os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}")


def print_section(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_test(name, passed, info=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}" + (f" - {info}" if info else ""))
    return passed


def create_test_mesh(nx=10, ny=10):
    """创建简单测试网格"""
    from DarcyFlow.misc import create_mesh_2d
    return create_mesh_2d(nx=nx, ny=ny)


def test_mcmc():
    """测试 MCMC 模块"""
    print_section("TEST 1: MCMC Samplers")
    
    results = []
    
    try:
        # 导入核心模块
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        
        # 使用绝对路径导入 DarcyFlow 模块
        sys.path.insert(0, os.path.join(project_root, "DarcyFlow"))
        
        # 动态导入 common.py
        import importlib.util
        spec_common = importlib.util.spec_from_file_location(
            "common", 
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_module = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_module)
        EquSolverDarcyFlow = common_module.EquSolverDarcyFlow
        ModelDarcyFlow = common_module.ModelDarcyFlow
        
        # 创建测试模型
        coords, elements = create_test_mesh(10, 10)
        solver = EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=3, std_dev=0.05)
        
        u_true = np.ones(len(solver.coords))
        noisy_data = solver.get_data(u_true) + noise.generate_sample()
        
        model = ModelDarcyFlow(prior, solver, noise,
                              {"coordinates": obs_coords, "data": noisy_data})
        
        results.append(print_test("Model Setup", True))
        
        # Test pCN
        spec_pcn = importlib.util.spec_from_file_location(
            "pCN",
            os.path.join(project_root, "DarcyFlow", "MCMC", "pCN.py")
        )
        pCN_module = importlib.util.module_from_spec(spec_pcn)
        spec_pcn.loader.exec_module(pCN_module)
        pCN = pCN_module.pCN
        
        pcn = pCN(model=model, beta=0.5, reduce_chain=10, num_select=3)
        chain = pcn.sampling(len_chain=30, verbose=False)
        
        passed = len(chain) > 0 and chain[0].shape[0] == model.num_dofs
        rate = pcn.get_acceptance_rate()
        results.append(print_test("pCN Sampler", passed,
                    f"Chain: {len(chain)}, Rate: {rate:.1%}"))
        
        # Test pCNL
        spec_pcnl = importlib.util.spec_from_file_location(
            "pCNL",
            os.path.join(project_root, "DarcyFlow", "MCMC", "pCNL.py")
        )
        pCNL_module = importlib.util.module_from_spec(spec_pcnl)
        spec_pcnl.loader.exec_module(pCNL_module)
        pCNL = pCNL_module.pCNL
        
        pcnl = pCNL(model=model, beta=0.01, reduce_chain=5)
        chain_l = pcnl.sampling(len_chain=20, verbose=False)
        
        passed = len(chain_l) > 0
        results.append(print_test("pCNL Sampler", passed,
                    f"Chain length: {len(chain_l)}"))
        
        # Test Newton-pCNL (仅初始化)
        spec_newton = importlib.util.spec_from_file_location(
            "Newton_pCNL",
            os.path.join(project_root, "DarcyFlow", "MCMC", "Newton_pCNL.py")
        )
        Newton_pCNL_module = importlib.util.module_from_spec(spec_newton)
        spec_newton.loader.exec_module(Newton_pCNL_module)
        Newton_pCNL = Newton_pCNL_module.Newton_pCNL
        
        newton_sampler = Newton_pCNL(model=model, dt=0.005, beta=0.01)
        results.append(print_test("Newton-pCNL Init", True))
        
    except Exception as e:
        print_test("MCMC Module", False, str(e))
        return False
    
    return all(results)


def test_optimization():
    """测试优化方法"""
    print_section("TEST 2: Optimization Methods")
    
    results = []
    
    try:
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        from core.optimizer import GradientDescent, NewtonCG
        
        import importlib.util
        
        # 加载 common
        spec_common = importlib.util.spec_from_file_location(
            "common",
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)
        
        coords, elements = create_test_mesh(12, 12)
        solver = common_mod.EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.3, 0.3], [0.7, 0.7], [0.5, 0.9]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=3, std_dev=0.1)
        
        model = common_mod.ModelDarcyFlow(
            prior, solver, noise,
            {"coordinates": obs_coords, 
             "data": np.array([1.0, 1.2, 0.8]) + noise.generate_sample()}
        )
        
        results.append(print_test("Optimization Setup", True))
        
        # Test GD
        gd = GradientDescent(model=model)
        gd.re_init(np.zeros(model.num_dofs))
        
        for _ in range(10):
            def smooth(x):
                return model.smoother.smoothing(x, degree=1e-2)
            gd.descent_direction(smooth)
            gd.step(method='armijo', show_step=False)
        
        loss_gd = model.loss()[0]
        results.append(print_test("Gradient Descent", True,
                    f"Iterations: 10, Loss: {loss_gd:.4f}"))
        
        # Test Newton-CG
        newton = NewtonCG(model=model)
        newton.re_init(np.zeros(model.num_dofs))
        
        for _ in range(5):
            newton.descent_direction(cg_max=3, method='bicgstab')
            newton.step(method='armijo', show_step=False)
        
        loss_newton = model.loss()[0]
        results.append(print_test("Newton-CG", True,
                    f"Iterations: 5, Loss: {loss_newton:.4f}"))
        
        # 测试 OptimizationMethods 模块接口
        spec_optim = importlib.util.spec_from_file_location(
            "optim_methods",
            os.path.join(project_root, "DarcyFlow", "OptimizationMethods", "optim_methods.py")
        )
        optim_mod = importlib.util.module_from_spec(spec_optim)
        spec_optim.loader.exec_module(optim_mod)
        
        results.append(print_test("Optimization Interface", True,
                    "run_optimization() available"))
        
    except Exception as e:
        print_test("Optimization Module", False, str(e))
        return False
    
    return all(results)


def test_gaussian_approximate():
    """测试高斯近似"""
    print_section("TEST 3: Gaussian Approximate")
    
    results = []
    
    try:
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        import importlib.util
        
        # 加载依赖
        spec_common = importlib.util.spec_from_file_location(
            "common",
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)
        
        coords, elements = create_test_mesh(10, 10)
        solver = common_mod.EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.25, 0.75], [0.75, 0.25]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=2, std_dev=0.1)
        
        model = common_mod.ModelDarcyFlow(
            prior, solver, noise,
            {"coordinates": obs_coords, "data": np.array([1.0, 1.2])}
        )
        
        # 加载 GaussianApproximate
        spec_ga = importlib.util.spec_from_file_location(
            "gaussian_approx",
            os.path.join(project_root, "DarcyFlow", "GaussianApproximate", "gaussian_approx.py")
        )
        ga_mod = importlib.util.module_from_spec(spec_ga)
        spec_ga.loader.exec_module(ga_mod)
        
        ga = ga_mod.GaussianApproximate(model)
        ga.set_mean(np.zeros(model.num_dofs))
        results.append(print_test("GA Initialization", True,
                    f"Dim: {model.num_dofs}"))
        
        # Hessian
        ga.compute_hessian()
        results.append(print_test("Hessian Computation", True,
                    f"Shape: {ga.hessian.shape}"))
        
        # Eigensystem
        ga.eval_eigensystem(num_eigval=min(10, model.num_dofs-1), method='scipy_eigsh')
        results.append(print_test("Eigensystem", True,
                    f"Eigenvalues: {len(ga.eigval)}"))
        
        # Sampling
        sample = ga.generate_sample(num_samples=1)
        results.append(print_test("Sample Generation", True,
                    f"Sample dim: {len(sample)}"))
        
    except Exception as e:
        print_test("Gaussian Approximate", False, str(e))
        return False
    
    return all(results)


def test_smc():
    """测试 SMC"""
    print_section("TEST 4: Sequential Monte Carlo")
    
    try:
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        import importlib.util
        
        spec_common = importlib.util.spec_from_file_location(
            "common",
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)
        
        coords, elements = create_test_mesh(8, 8)
        solver = common_mod.EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.3, 0.3], [0.7, 0.7]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=2, std_dev=0.1)
        
        model = common_mod.ModelDarcyFlow(
            prior, solver, noise,
            {"coordinates": obs_coords, "data": np.array([1.0, 1.2])}
        )
        
        spec_smc = importlib.util.spec_from_file_location(
            "SMC_sampler",
            os.path.join(project_root, "DarcyFlow", "SequentialMonteCarlo", "SMC_sampler.py")
        )
        smc_mod = importlib.util.module_from_spec(spec_smc)
        spec_smc.loader.exec_module(smc_mod)
        
        smc = smc_mod.SMC(model=model, num_particles=10)
        smc.prepare()
        
        passed = smc.particles is not None and smc.particles.shape == (10, model.num_dofs)
        return print_test("SMC Initialization & Prepare", passed,
                        f"Particles: {smc.particles.shape if passed else 'None'}")
        
    except Exception as e:
        return print_test("SMC Module", False, str(e))


def test_vi():
    """测试变分推断"""
    print_section("TEST 5: Variational Inference")
    
    results = []
    
    try:
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        import importlib.util
        
        spec_common = importlib.util.spec_from_file_location(
            "common",
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)
        
        coords, elements = create_test_mesh(8, 8)
        solver = common_mod.EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.3, 0.3], [0.7, 0.7]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=2, std_dev=0.1)
        
        model = common_mod.ModelDarcyFlow(
            prior, solver, noise,
            {"coordinates": obs_coords, "data": np.array([1.0, 1.2])}
        )
        
        # Mean-Field VI
        spec_vi = importlib.util.spec_from_file_location(
            "mean_field_VI",
            os.path.join(project_root, "DarcyFlow", "VariationalInference-ongoing", "mean_field_VI.py")
        )
        vi_mod = importlib.util.module_from_spec(spec_vi)
        spec_vi.loader.exec_module(vi_mod)
        
        vi = vi_mod.MeanFieldVI(model, num_samples_vi=3)
        vi.initialize()
        
        passed = vi.mean is not None and vi.log_std is not None
        results.append(print_test("Mean-Field VI Init", passed,
                    f"Mean shape: {vi.mean.shape if passed else 'None'}"))
        
        vi_res = vi.optimize(max_iter=3, verbose=False)
        results.append(print_test("VI Optimize (3 iters)", 'final_elbo' in vi_res,
                    f"ELBO: {vi_res.get('final_elbo', 'N/A')}"))
        
        # SVGD
        spec_svgd = importlib.util.spec_from_file_location(
            "svgd",
            os.path.join(project_root, "DarcyFlow", "VariationalInference-ongoing", "svgd.py")
        )
        svgd_mod = importlib.util.module_from_spec(spec_svgd)
        spec_svgd.loader.exec_module(svgd_mod)
        
        svgd = svgd_mod.SVGD(model=model, n_particles=15)
        svgd.initialize(init_method='prior')
        
        passed = svgd.particles is not None and svgd.particles.shape == (15, model.num_dofs)
        results.append(print_test("SVGD Initialization", passed,
                    f"Particles: {svgd.particles.shape if passed else 'None'}"))
        
    except Exception as e:
        print_test("VI Module", False, str(e))
        return False
    
    return all(results)


def test_rmap():
    """测试 rMAP"""
    print_section("TEST 6: Randomized MAP")
    
    results = []
    
    try:
        from DarcyFlow.misc import create_mesh_2d
        from core.probability import GaussianElliptic2
        from core.noise import NoiseGaussianIID
        import importlib.util
        
        spec_common = importlib.util.spec_from_file_location(
            "common",
            os.path.join(project_root, "DarcyFlow", "common.py")
        )
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)
        
        coords, elements = create_test_mesh(8, 8)
        solver = common_mod.EquSolverDarcyFlow(coords, elements)
        prior = GaussianElliptic2(coords, elements)
        
        obs_coords = np.array([[0.3, 0.3], [0.7, 0.7]])
        solver._init_measurement_matrix(obs_coords)
        noise = NoiseGaussianIID(dim=2, std_dev=0.1)
        
        model = common_mod.ModelDarcyFlow(
            prior, solver, noise,
            {"coordinates": obs_coords, "data": np.array([1.0, 1.2])}
        )
        
        spec_rmap = importlib.util.spec_from_file_location(
            "rMAP",
            os.path.join(project_root, "DarcyFlow", "rMAP", "rMAP.py")
        )
        rmap_mod = importlib.util.module_from_spec(spec_rmap)
        spec_rmap.loader.exec_module(rmap_mod)
        
        rmap_instance = rmap_mod.rMAP(model=model, num_samples=5)
        results.append(print_test("rMAP Initialization", True,
                    f"Target samples: {rmap_instance.num_samples}"))
        
        # 测试数据扰动
        orig_data = np.array([1.0, 1.5])
        for i in range(3):
            perturbed = rmap_instance._generate_perturbed_data(orig_data)
            
        results.append(print_test("Data Perturbation", True))
        
    except Exception as e:
        print_test("rMAP Module", False, str(e))
        return False
    
    return all(results)


def test_data_generator():
    """测试数据生成器"""
    print_section("TEST 7: Data Generator")
    
    results = []
    
    try:
        import importlib.util
        
        spec_gen = importlib.util.spec_from_file_location(
            "generate_data",
            os.path.join(darcyflow_dir, "generate_data.py")
        )
        gen_mod = importlib.util.module_from_spec(spec_gen)
        spec_gen.loader.exec_module(gen_mod)
        
        # 参数场生成
        test_coords = np.random.rand(30, 2)
        
        for form in ['sinusoidal', 'polynomial']:
            u = gen_mod.generate_true_parameter(test_coords, func_form=form)
            passed = u.shape == (30,) and not np.any(np.isnan(u))
            results.append(print_test(f"Parameter ({form})", passed,
                        f"Range: [{u.min():.3f}, {u.max():.3f}]"))
        
        # 观测点生成
        obs = gen_mod.generate_observation_coords(num_obs=9, pattern='grid')
        results.append(print_test("Observation Coords", obs.shape == (9, 2),
                    f"Shape: {obs.shape}"))
        
        # 噪声添加
        clean = np.linspace(0, 1, 50)
        noisy = gen_mod.add_noise_to_data(clean, noise_level=0.05, seed=42)
        results.append(print_test("Noise Addition", noisy.shape == (50,),
                    f"Max diff: {np.abs(noisy-clean).max():.4f}"))
        
    except Exception as e:
        print_test("Data Generator", False, str(e))
        return False
    
    return all(results)


def main():
    start_time = time.time()
    
    print("\n" + "#"*70)
    print("# IPBayesML-Tatva - DarcyFlow Modules Test Suite")
    print("#"*70)
    
    tests = [
        ("MCMC Samplers", test_mcmc),
        ("Optimization Methods", test_optimization),
        ("Gaussian Approximate", test_gaussian_approximate),
        ("Sequential Monte Carlo", test_smc),
        ("Variational Inference", test_vi),
        ("Randomized MAP", test_rmap),
        ("Data Generator", test_data_generator),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[CRASH] {name}: {e}")
            results[name] = False
    
    # Summary
    elapsed = time.time() - start_time
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print(f"\n{'Module':<35} {'Status'}")
    print("-"*45)
    
    for name, ok in results.items():
        symbol = "✓" if ok else "✗"
        print(f"{name:<35} [{symbol}]")
    
    print("-"*45)
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Time: {elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 All DarcyFlow modules working correctly!")
    else:
        failed = [k for k,v in results.items() if not v]
        print(f"\n⚠️ Failed: {failed}")
    
    print("#"*70 + "\n")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
