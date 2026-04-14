#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMC 采样器模块

提供多种针对无限维贝叶斯逆问题的MCMC采样算法:
- VanillaMCMC: 标准 (拟) MCMC
- pCN: preconditioned Crank-Nicolson
- pCNL: pCN with Langevin gradient term
- Newton_pCNL: pCNL using Hessian-based preconditioning from MAP estimate
- SMC: Sequential Monte Carlo

参考:
- Cotter et al. (2013) "MCMC Methods for Functions"
- Dashti & Stuart (2017) "The Bayesian Approach to Inverse Problems"

基于 IPBayesML-FEniCSx09/core/sample.py 重构，使用 scipy sparse 替代 dolfinx/petsc
"""

import os
import numpy as np
from scipy.special import logsumexp


class MCMCBase:
    """MCMC 基类，提供通用的链管理和保存功能"""
    
    def __init__(self, model, reduce_chain=None, save_path=None, num_select=None):
        assert hasattr(model, "prior"), "model must have prior attribute"
        self.model = model
        self.prior = model.prior
        
        # reduce_chain: 每隔 reduce_chain 步保存一次样本到磁盘
        self.reduce_chain = reduce_chain
        # save_path: 样本保存路径
        if isinstance(save_path, str):
            save_path = os.path.abspath(save_path)
        self.save_path = save_path
        # num_select: 减薄间隔（每隔 num_select 步取一个）
        self.num_select = num_select
        
        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.chain = []
        self.acc_rate = 0.0
        self.index = 0
    
    def save_local(self):
        """按批次保存样本到磁盘"""
        if self.reduce_chain is not None:
            if self.save_path is not None:
                if int(len(self.chain)) >= int(self.reduce_chain):
                    if self.num_select is None:
                        np.save(
                            os.path.join(self.save_path, f'sample_{int(self.index)}.npy'),
                            np.array(self.chain)
                        )
                    elif type(self.num_select) in [np.int64, np.int32, int]:
                        np.save(
                            os.path.join(self.save_path, f'sample_{int(self.index)}.npy'),
                            np.array(self.chain[::int(self.num_select)])
                        )
                    self.chain = []
                    self.index += 1
            else:
                if int(len(self.chain)) >= int(self.reduce_chain):
                    self.chain = []
                    self.index += 1
    
    def save_all(self):
        """保存所有剩余样本"""
        if self.save_path is not None and self.reduce_chain is None and len(self.chain) > 0:
            np.save(os.path.join(self.save_path, 'samples_all.npy'), np.array(self.chain))

    def sampling(self, len_chain, callback=None, u0=None, index=None, **kwargs):
        """子类必须实现的采样方法"""
        raise NotImplementedError("Subclasses must implement sampling()")


class VanillaMCMC(MCMCBase):
    """
    Vanilla (拟) MCMC 方法
    
    参考: Cotter et al. (2013), Section 4.2
    """
    
    def __init__(self, model, beta, reduce_chain=None, save_path=None, num_select=None):
        super().__init__(model, reduce_chain, save_path, num_select)
        assert hasattr(model.prior, "eval_CM_inner"), "prior must have eval_CM_inner method"
        
        self.M = model.prior.M
        self.K = model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs
        self.beta = beta
        
        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)

    def rho(self, x_info, y_info):
        """计算目标函数值 ρ(x) = Φ(x) + 0.5 * <x, C^{-1}x>"""
        val = 0.5 * self.prior.eval_CM_inner(x_info[0])
        return x_info[1] + val

    def proposal(self, x_info):
        """
        生成提议 y = x + β * v, v ~ N(0, C_0)
        """
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return ans1 + self.beta * ans2

    def sampling(self, len_chain=int(1e5), callback=None, u0=None, index=None):
        """执行 Vanilla MCMC 采样"""
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        
        assert x.shape[0] == self.dim and x.ndim == 1
        
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        
        self.chain = [x]
        acc_num = 0
        
        if index is None:
            self.index = 0
        else:
            self.index = index
        
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])
        
        self.save_all()


class pCN(MCMCBase):
    """
    preconditioned Crank-Nicolson (pCN) 采样器
    
    pCN 是一种针对无限维贝叶斯逆问题的MCMC方法，
    利用先验分布作为预条件子，保证在网格细化时采样性质不变。
    
    算法流程:
    1. 从先验 N(0, C_0) 采样 v
    2. 提议: u' = sqrt(1-β²) u + β v
    3. 接受概率: α = min(1, exp(Φ(u') - Φ(u)))
    
    参考: Dashti & Stuart (2017), Sections 5.1-5.2
    """
    
    def __init__(self, model, beta, reduce_chain=None, save_path=None, num_select=None):
        super().__init__(model, reduce_chain, save_path, num_select)
        self.M = model.prior.M
        self.K = model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs
        
        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)
        self.beta = beta

    def rho(self, x_info, y_info):
        """pCN 的目标函数仅包含似然部分"""
        return x_info[1]

    def proposal(self, x_info):
        """
        pCN 提议: y = sqrt(1-β²) x + β v, v ~ N(0, C_0)
        """
        coef1 = np.sqrt(1 - self.beta**2)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return coef1 * ans1 + self.beta * ans2

    def sampling(self, len_chain=int(1e5), callback=None, u0=None, index=None):
        """执行 pCN 采样"""
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        
        assert x.shape[0] == self.dim and x.ndim == 1
        
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        
        self.chain = [x]
        acc_num = 0
        
        if index is None:
            self.index = 0
        else:
            self.index = index
        
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])
        
        self.save_all()


class pCNL(MCMCBase):
    """
    preconditioned Crank-Nicolson Langevin (pCNL) 采样器
    
    在 pCN 的基础上加入梯度项（Langevin 动力学），提高采样效率。
    
    参考: Cotter et al. (2013), "MCMC Methods for Functions"
    """
    
    def __init__(self, model, beta, grad_smooth_degree=1e-2, 
                 reduce_chain=None, save_path=None, num_select=None):
        super().__init__(model, reduce_chain, save_path, num_select)
        assert hasattr(model, 'smoother'), "model must have smoother for gradient smoothing"
        
        model.smoother.set_degree(grad_smooth_degree)
        
        self.model = model
        self.prior = model.prior
        self.M = model.prior.M
        self.K = model.prior.K
        self.loss = model.loss_res
        self.grad = model.eval_grad_res
        self.dim = model.num_dofs
        
        tmp = np.sqrt(1 - beta ** 2)
        self.dt = (2 - 2 * tmp) / (1 + tmp)

    def rho(self, x_info, y_info):
        """pCNL 目标函数（包含梯度项）"""
        x, grad_x, loss_x = x_info
        y = y_info[0]
        coef1, coef2, coef3, coef4 = 1, 1/2, self.dt/4, self.dt/4
        ans1 = loss_x
        ans2 = grad_x @ (self.M @ (y - x))
        ans3 = grad_x @ (self.M @ (x + y))
        ans4 = self.prior.eval_C(grad_x) @ (self.M @ grad_x)
        return coef1*ans1 + coef2*ans2 + coef3*ans3 + coef4*ans4

    def proposal(self, x_info):
        """pCNL 提议（包含梯度修正）"""
        dt = self.dt
        x, grad_x, loss_x = x_info
        
        coef1 = (2 - dt) / (2 + dt)
        coef2 = -2 * dt / (2 + dt)
        coef3 = np.sqrt(8 * dt) / (2 + dt)
        
        ans1 = x
        ans2 = self.prior.eval_C(grad_x)
        ans3 = self.prior.generate_sample()
        return coef1*ans1 + coef2*ans2 + coef3*ans3

    def sampling(self, len_chain=int(1e5), callback=None, u0=None, index=None):
        """执行 pCNL 采样"""
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        
        assert x.shape[0] == self.dim and x.ndim == 1
        
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        
        self.chain = [x]
        acc_num = 0
        
        if index is None:
            self.index = 0
        else:
            self.index = index
        
        grad_x = self.grad(x)
        grad_x = self.model.smoother.smoothing(grad_x)
        x_loss = self.loss()
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, grad_x, x_loss]
        
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            grad_y = self.grad(y)
            grad_y = self.model.smoother.smoothing(grad_y)
            y_loss = self.loss()
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, grad_y, y_loss]
            
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1], y_info[2]]
                acc_num += 1
            
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])
        
        self.save_all()


class Newton_pCNL(MCMCBase):
    """
    Newton-pCNL 采样器
    
    在 MAP 点处利用 Hessian 信息构建更优的预条件子，
    结合 pCNL 的梯度信息，大幅提高高维问题上的采样效率。
    
    仅对高斯先验有效。
    
    mode 选项:
    - "map_hessian": 在 MAP 点处计算一次 Hessian 并固定
    - "every_step_hessian": 每步重新计算 Hessian (较慢但更精确)
    - "init_hessian": 在初始点处计算 Hessian
    
    参考: 基于 Cotter et al. (2013) 的扩展
    """
    
    def __init__(self, model, dt, beta=0.01, 
                 reduce_chain=None, save_path=None, num_select=None):
        super().__init__(model, reduce_chain, save_path, num_select)
        
        assert hasattr(model.prior, 'M') and hasattr(model.prior, 'K')
        assert hasattr(model.prior, 'eval_sqrtM')
        assert hasattr(model.prior, 'eval_sqrtC')
        assert hasattr(model.prior, 'eval_C')
        assert hasattr(model.prior, 'eval_Cinv')
        assert hasattr(model, 'smoother')
        
        self.M = model.prior.M
        self.K = model.prior.K
        self.loss = model.loss_res
        self.grad = model.eval_grad_res
        self.dim = model.num_dofs
        
        self.dt, self.a = dt, 1 + dt / 2
        tmp = np.sqrt(1 - beta ** 2)
        self.dt_pcn = (2 - 2 * tmp) / (1 + tmp)
        
        # 优化器选项
        self.optim_options = {
            "max_iter": [50, 50], "init_val": None, "info_optim": False,
            "cg_max": 100, "newton_method": 'bicgstab', "grad_smooth_degree": 1e-2
        }
        model.smoother.set_degree(self.optim_options["grad_smooth_degree"])
        
        # 特征系统选项
        self.eigensystem_optims = {
            "cut_val": 0.1, "method": "scipy_eigsh", "num_eigval": 30,
            "hessian_type": "linear_approximate", "oversampling_factor": 20
        }
        
        self.mode = "every_step_hessian"
        self.eval_eigensystem_iter_max = 10
        self.if_grad = True
        self.if_eigensystem = True
        self.map_estimate = None

    def optimizing(self, optimizer, max_iter=50, init_val=None, info=True, **kwargs):
        """运行优化器找到 MAP 估计"""
        if init_val is None:
            init_val = np.zeros(self.dim)
        assert init_val.shape[0] == self.dim and init_val.ndim == 1
        
        optimizer.re_init(init_val)
        
        loss_pre = self.model.loss()[0]
        if info:
            print("Init loss: ", loss_pre)
        
        for itr in range(max_iter):
            if 'smoothing' in kwargs:
                optimizer.descent_direction(smoothing=kwargs['smoothing'])
            elif 'cg_max' in kwargs and 'method' in kwargs:
                optimizer.descent_direction(cg_max=kwargs['cg_max'], method=kwargs['method'])
            else:
                optimizer.descent_direction()
                
            optimizer.step(method='armijo', show_step=False)
            if optimizer.converged == False:
                break
            
            loss = self.model.loss()[0]
            if info:
                print("iter = %2d/%d, loss = %.4f" % (itr + 1, max_iter, loss))
            if np.abs(loss - loss_pre) < 1e-3 * loss:
                if info:
                    print("Iteration stopped at iter = %d" % itr)
                break
            loss_pre = loss
        
        return optimizer.mk.copy()

    def eval_map(self):
        """计算 MAP 估计（先用梯度下降再用 Newton-CG）"""
        from core.optimizer import GradientDescent, NewtonCG
        
        self.model.smoother.set_degree(self.optim_options["grad_smooth_degree"])
        
        # 第一阶段：梯度下降
        optimizer = GradientDescent(model=self.model)
        estimated_param = self.optimizing(
            optimizer, max_iter=self.optim_options["max_iter"][0], 
            init_val=self.optim_options["init_val"],
            info=self.optim_options["info_optim"], 
            smoothing=self.model.smoother.smoothing
        )
        
        # 第二阶段：Newton-CG
        optimizer = NewtonCG(model=self.model)
        estimated_param = self.optimizing(
            optimizer, max_iter=self.optim_options["max_iter"][1], 
            init_val=estimated_param,
            info=self.optim_options["info_optim"], 
            cg_max=self.optim_options["cg_max"],
            method=self.optim_options["newton_method"]
        )
        
        self.map_estimate = estimated_param

    def eval_eigsystem(self, param=None):
        """在当前参数处计算后验 Hessian 的特征系统"""
        from core.approximate_sample import GaussianApproximate
        
        if param is None:
            self.model.update_param(self.map_estimate, update_sol=True)
        else:
            assert param.ndim == 1 and param.shape[0] == self.model.num_dofs
            self.model.update_param(param, update_sol=True)
        
        gaussian_approximate = GaussianApproximate(
            self.model, hessian_type=self.eigensystem_optims["hessian_type"]
        )
        cut_val = self.eigensystem_optims["cut_val"]
        gaussian_approximate.eval_eigensystem(
            num_eigval=self.eigensystem_optims["num_eigval"],
            method=self.eigensystem_optims["method"],
            oversampling_factor=self.eigensystem_optims["oversampling_factor"],
            cut_val=cut_val
        )
        
        iter_num = 1
        if len(gaussian_approximate.eigval) == 0:
            self.if_eigensystem = False
        else:
            self.if_eigensystem = True
            
            if self.if_eigensystem is True:
                while iter_num <= self.eval_eigensystem_iter_max:
                    if gaussian_approximate.eigval[-1] >= 1:
                        cut_val += 10
                        gaussian_approximate.eval_eigensystem(
                            num_eigval=self.eigensystem_optims["num_eigval"],
                            method=self.eigensystem_optims["method"],
                            oversampling_factor=self.eigensystem_optims["oversampling_factor"],
                            cut_val=cut_val
                        )
                    iter_num += 1
                
                # 存储特征系统和相关矩阵
                self.eigvals, self.ch_eigvecs = gaussian_approximate.eigval, gaussian_approximate.ch_eigvec
                self.eigvecs = self.prior.eval_sqrtCinv(self.ch_eigvecs)
                
                self.num_eigvals = len(self.eigvals)
                self.eigvecs_lozenge = (self.eigvecs.T) @ self.M
                self.ch_eigvecs_lozenge = (self.ch_eigvecs.T) @ self.M
                
                # 预计算对角矩阵用于提议和接受率计算
                self.Da = np.diag(self.eigvals / (self.eigvals + self.a))
                self.D_proposal_part2 = np.diag(
                    self.a * np.sqrt(self.eigvals + 1) / (self.a + self.eigvals) - 1
                )
                self.D1 = np.diag(self.eigvals / (self.eigvals + 1))

    def preC(self, x):
        """预条件子 (C^{-1}+H)^{-1}"""
        return self.prior.eval_C(x) - self.ch_eigvecs @ (self.D1 @ (self.ch_eigvecs_lozenge @ x))

    def preC_Cinv(self, x):
        """预条件子 (C^{-1}+H)^{-1} C^{-1}"""
        return x - self.ch_eigvecs @ (self.D1 @ (self.ch_eigvecs_lozenge @ self.prior.eval_Cinv(x)))

    def proposal_part1(self, x):
        """提议的第一部分：确定性漂移"""
        ans1 = (1 - self.dt / self.a) * x
        tmp1 = self.prior.eval_sqrtCinv(x)
        ans2 = self.dt / self.a * (self.Da @ (self.eigvecs_lozenge @ tmp1))
        return [ans1, ans2]

    def proposal_part2(self, n):
        """提议的第二部分：随机扰动"""
        coef = np.sqrt(2 * self.dt) / self.a
        tmp1 = self.prior.eval_sqrtM(n)
        ans1 = self.D_proposal_part2 @ (self.eigvecs.T @ tmp1)
        ans2 = self.prior.eval_sqrtCsqrtMinv(n)
        return [coef*ans1, coef*ans2]

    def proposal_part3(self, grad_x):
        """提议的第三部分：梯度修正"""
        coef = self.dt / self.a
        ans1 = self.Da @ (self.ch_eigvecs_lozenge @ grad_x)
        ans2 = -1 * self.prior.eval_C(grad_x)
        return [coef*ans1, coef*ans2]

    def proposal(self, x_info):
        """完整的 Newton-pCNL 提议"""
        n = np.random.normal(0, 1, self.dim)
        x, grad_x, loss_x = x_info
        
        ans11, ans12 = self.proposal_part1(x)
        ans21, ans22 = self.proposal_part2(n)
        ans31, ans32 = self.proposal_part3(grad_x)
        
        # 如果梯度项产生 NaN，退化为标准 pCN
        if np.any(np.isnan(ans31)) or np.any(np.isnan(ans32)):
            self.if_grad = False
            print("Warning: Grad term is NaN, reducing to standard pCN!")
            return ans11 + ans22 + self.ch_eigvecs @ (ans12 + ans21)
        else:
            self.if_grad = True
            return ans11 + ans22 + ans32 + self.ch_eigvecs @ (ans12 + ans21 + ans31)

    def rho(self, x_info, y_info):
        """Newton-pCNL 接受率计算的目标函数"""
        x, grad_x, loss_x = x_info
        y = y_info[0]
        
        if self.if_grad == True:
            coef1, coef2, coef3, coef4 = 1, 1/2, self.dt/4, self.dt/4
            ans1 = loss_x
            ans2 = grad_x @ (self.M @ (y - x))
            ans3 = grad_x @ (self.M @ self.preC_Cinv(x + y))
            ans4 = self.preC(grad_x) @ self.M @ grad_x
            return coef1*ans1 + coef2*ans2 + coef3*ans3 + coef4*ans4
        elif self.if_grad == False:
            return loss_x

    def rho_pCN(self, x_info, y_info):
        """退化到标准 pCN 时的接受率"""
        return x_info[1]

    def proposal_pCN(self, x_info):
        """标准 pCN 提议（当特征系统计算失败时使用）"""
        dt = self.dt_pcn
        coef1 = (2 - dt) / (2 + dt)
        coef2 = np.sqrt(8 * dt) / (2 + dt)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return coef1 * ans1 + coef2 * ans2

    def sampling(self, len_chain=int(1e5), callback=None, u0=None, index=None):
        """执行 Newton-pCNL 采样"""
        data_info = np.finfo(np.float64)
        max_value = data_info.max
        
        if u0 is None:
            if self.mode == "map_hessian":
                x = self.map_estimate
            elif self.mode == "every_step_hessian":
                x = self.prior.generate_sample()
            else:
                raise NotImplementedError("mode must be map_hessian or every_step_hessian")
        else:
            if self.mode == "map_hessian":
                assert u0.shape[0] == self.dim and u0.ndim == 1
                self.optim_options["init_val"] = u0
                self.eval_map()
                self.eval_eigsystem()
                x = self.map_estimate
            elif self.mode == "every_step_hessian":
                x = u0.copy()
            elif self.mode == "init_hessian":
                assert u0.shape[0] == self.dim and u0.ndim == 1
                self.eval_eigsystem(u0)
                x = u0.copy()
            else:
                raise NotImplementedError
        
        assert x.shape[0] == self.dim and x.ndim == 1
        
        self.chain = [x]
        acc_num = 0
        
        if index is None:
            self.index = 0
        else:
            self.index = index
        
        # 初始化状态
        if self.if_eigensystem is True:
            grad_x = self.model.smoother.smoothing(self.grad(x))
            x_loss = self.loss()
            if np.isnan(x_loss):
                x_loss = max_value
            x_info = [x, grad_x, x_loss]
            if self.mode == "every_step_hessian":
                self.eval_eigsystem(x_info[0])
                if not self.if_eigensystem:
                    print("Warning: No positive eigenvalues, falling back to pCN!")
        else:
            x_loss = self.loss(x)
            if np.isnan(x_loss):
                x_loss = max_value
            x_info = [x, x_loss]
        
        i = 1
        while i <= len_chain:
            if self.if_eigensystem is True:
                y = self.proposal(x_info)
                grad_y = self.model.smoother.smoothing(self.grad(y))
                y_loss = self.loss()
                if np.isnan(y_loss):
                    y_loss = max_value
                y_info = [y, grad_y, y_loss]
                tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
                tem_acc = np.exp(min(0, tem))
            else:
                y = self.proposal_pCN(x_info)
                y_loss = self.loss(y)
                if np.isnan(y_loss):
                    y_loss = max_value
                y_info = [y, y_loss]
                tem = self.rho_pCN(x_info, y_info) - self.rho_pCN(y_info, x_info)
                tem_acc = np.exp(min(0, tem))
            
            if np.random.uniform() < tem_acc:
                x_info = y_info
                if self.mode == "every_step_hessian" and self.if_eigensystem:
                    self.eval_eigsystem(x_info[0])
                    if not self.if_eigensystem:
                        print("Warning: No positive eigenvalues, falling back to pCN!")
                acc_num += 1
            
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])
        
        self.save_all()


class SMC:
    """
    Sequential Monte Carlo (SMC) 采样器
    
    通过一系列中间分布逐步从先验过渡到后验，
    每层使用 MCMC 进行粒子转移。
    
    参考: Dashti & Stuart (2017), Section 5.3
    
    注意: 此实现为单进程版本，不依赖 MPI
    """
    
    def __init__(self, model, num_particles):
        self.model = model
        self.num_particles = num_particles
        self.rank = 0  # 单进程版本始终为 0

    def prepare(self):
        """初始化粒子（从先验采样）"""
        self.weights = np.ones(self.num_particles) * (1.0 / self.num_particles)
        
        # 从先验生成初始粒子
        particles_local = []
        for idx in range(self.num_particles):
            particles_local.append(self.model.prior.generate_sample())
        self.particles_local = np.array(particles_local).reshape(
            (self.num_particles, self.model.num_dofs)
        )
        
        self.dtype = self.particles_local[0, 0].dtype

    def transition(self, sampler, len_chain, info_acc_rate=True, **kwargs):
        """对每个粒子执行 MCMC 转移"""
        assert hasattr(sampler, 'sampling')
        assert hasattr(sampler, 'acc_rate')
        assert hasattr(sampler, 'chain')
        
        self.acc_rates = np.zeros(self.num_particles)
        
        for idx in range(self.num_particles):
            sampler.sampling(len_chain=len_chain, u0=self.particles_local[idx, :], **kwargs)
            if info_acc_rate:
                print(f"Particle {idx}: acc_rate = {sampler.acc_rate:.5f}")
            self.acc_rates[idx] = sampler.acc_rate
            
            tmp = np.array(sampler.chain[-1]).squeeze()
            if np.any(np.isnan(tmp)):
                raise ValueError("Particles should not contain NaN!")
            self.particles_local[idx, :] = tmp.copy()

    def gather_acc_rates(self):
        """收集所有粒子的平均接受率"""
        return np.mean(self.acc_rates)

    def resampling(self, potential_fun):
        """
        根据势函数重新加权并重采样粒子
        
        Parameters:
        -----------
        potential_fun : callable
            势函数 Φ(u)，通常为负对数似然
        """
        # 计算新权重
        tmp = np.zeros(self.num_particles, dtype=np.float64)
        for idx in range(self.num_particles):
            tmp_p = potential_fun(self.particles_local[idx, :])
            if np.isnan(tmp_p):
                tmp_p = np.finfo(self.dtype).max
            tmp[idx] = -tmp_p + np.log(self.weights[idx] + 1e-20)
        
        # 归一化权重
        self.weights = np.exp(tmp - logsumexp(tmp))
        
        # 系统重采样
        idx_resample = np.random.choice(
            len(self.weights), len(self.weights), True, self.weights
        )
        self.particles_local = self.particles_local[idx_resample, :]
        self.weights = np.ones(self.num_particles) * (1.0 / self.num_particles)

    def gather_samples(self):
        """收集所有粒子作为样本"""
        return self.particles_local.copy()
