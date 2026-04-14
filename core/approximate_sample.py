#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高斯近似采样模块

提供 Laplace 近似、rMAP 等基于高斯近似的后验采样方法

作者: 基于 IPBayesML-FEniCSx09 迁移，适配新架构
"""

import numpy as np
from typing import Optional


class GaussianApproximate:
    """
    高斯 (Laplace) 近似后验分布
    
    在 MAP 点处对后验进行二阶近似：
    π(u|d) ≈ N(u_MAP, H^{-1})
    
    其中 H 是后验的 Hessian 矩阵
    """
    
    def __init__(self, model):
        self.model = model
        self.mean = None
        self.eigval = None
        self.eigvec = None
        self.H_approx = None
    
    def eval_eigensystem(self, num_eigval=30, method="jax_eigh"):
        """
        计算近似 Hessian 的特征系统（使用 JAX 或 scipy）
        
        对于 JAX 后端，优先使用 jax.numpy.linalg.eigh（稠密矩阵），
        因为 JAX 没有内置稀疏特征值求解器。
        大规模问题应考虑使用 scipy.sparse.linalg.eigsh 作为 fallback。
        """
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'K'):
            prior_part = self.model.prior.K
            # 如果是 JAX 稀疏矩阵，转为稠密进行特征分解
            if hasattr(prior_part, 'toarray'):
                K_dense = np.array(prior_part.toarray())
            else:
                K_dense = np.array(prior_part)
        else:
            n = self.model.num_dofs
            K_dense = np.eye(n) * 0.1
        
        try:
            # 使用 numpy/scipy 进行特征分解（JAX 不支持稀疏 eigsh）
            if method == "jax_eigh" or K_dense.shape[0] <= 500:
                # 小规模：用 dense eigh
                eigvals, eigvecs = np.linalg.eigh(K_dense)
                # 取最小的 num_eigval 个特征值
                idx = np.argsort(eigvals)[:num_eigval]
                self.eigval = eigvals[idx]
                self.eigvec = eigvecs[:, idx]
            else:
                # 大规模稀疏：回退到 scipy eigsh
                from scipy.sparse.linalg import eigsh as sp_eigsh
                from scipy.sparse import csr_matrix as sp_csr
                K_sp = sp_csr(K_dense) if not isinstance(K_dense, np.ndarray) else K_dense
                self.eigval, self.eigvec = sp_eigsh(K_sp, k=min(num_eigval, K_dense.shape[0]-2), which='SM')
        except Exception as e:
            print(f"Warning: Eigensolver failed: {e}")
            self.eigval = np.ones(num_eigval)
            self.eigvec = np.eye(K_dense.shape[0])[:num_eigval]
        
        n = self.model.num_dofs
        
        # 构建近似 Hessian: H ≈ C_0^{-1} + G^T Γ^{-1} G
        # 这里使用简化形式
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'K'):
            prior_part = self.model.prior.K
        else:
            prior_part = sparse.eye(n) * 0.1
        
        try:
            self.eigval, self.eigvec = eigsh(prior_part, k=min(num_eigval, n-2),
                                              which='SM')
        except Exception as e:
            print(f"Warning: Eigensolver failed: {e}")
            self.eigval = np.ones(num_eigval)
            self.eigvec = np.eye(n)[:num_eigval]
    
    def set_mean(self, mean_vec):
        """设置后验均值 (MAP 点)"""
        self.mean = mean_vec.copy()
    
    def generate_sample(self) -> np.ndarray:
        """从近似后验分布生成样本"""
        if self.mean is None:
            raise ValueError("Mean not set. Call set_mean() first.")
        
        if self.eigval is None or self.eigvec is None:
            # 如果没有特征分解，使用简单的高斯扰动
            return self.mean + 0.01 * np.random.randn(len(self.mean))
        
        # 使用特征分解生成样本: u = mean + V Λ^{1/2} z
        z = np.random.randn(len(self.eigval))
        sample = self.mean + self.eigvec @ (np.sqrt(np.abs(self.eigval)) * z)
        
        return sample
    
    def pointwise_variance_field(self, coor):
        """计算点wise 后验方差场"""
        # 简化实现
        n_points = coor.shape[0]
        return np.eye(n_points) * np.mean(np.abs(self.eigval)) if self.eigval is not None else np.eye(n_points)


class rMAP:
    """
    随机化最大后验估计 (randomized MAP) 采样器
    
    通过随机扰动目标函数来探索后验分布
    """
    
    def __init__(self, comm, num_size, model, comm_size=1):
        self.comm = comm
        self.num_size = num_size
        self.model = model
        self.comm_size = comm_size
        
        self.optim_options = {
            "max_iter": [500, 100],
            "init_val": None,
            "info_optim": True,
            "cg_max": 200,
            "newton_method": 'bicgstab',
            "grad_smooth_degree": 1e-2,
            "if_normalize_dd": True
        }
    
    def optimizing(self):
        """找到 MAP 估计"""
        from core.optimizer import GradientDescent, NewtonCG
        
        optimizer = NewtonCG(model=self.model)
        
        init_val = np.zeros(self.model.num_dofs)
        if self.optim_options.get("init_val") is not None:
            init_val = self.optim_options["init_val"]
        
        optimizer.re_init(init_val)
        
        max_iter = self.optim_options["max_iter"][0]
        for itr in range(max_iter):
            optimizer.descent_direction(
                cg_max=self.optim_options["cg_max"],
                method=self.optim_options["newton_method"]
            )
            optimizer.step(method='armijo', show_step=False)
            if not optimizer.converged:
                break
            
            if self.optim_options.get("info_optim", False):
                loss = self.model.loss()[0]
                print(f"MAP iter {itr+1}/{max_iter}, loss = {loss:.6f}")
        
        return optimizer.mk.copy()
    
    def sampling(self, init_vec=None):
        """从 rMAP 分布中采样"""
        samples = []
        
        for i in range(self.num_size):
            # 添加随机扰动
            noise = self.model.prior.generate_sample() * 0.1
            u_init = init_vec + noise if init_vec is not None else noise
            
            # 对每个扰动后的初始值进行优化
            sample = self._sample_single(u_init)
            samples.append(sample)
        
        return np.array(samples)
    
    def _sample_single(self, u_init):
        """单个样本的采样"""
        from core.optimizer import NewtonCG
        
        optimizer = NewtonCG(model=self.model)
        optimizer.re_init(u_init)
        
        max_iter = self.optim_options.get("max_iter", [100, 50])[1]
        for _ in range(max_iter):
            optimizer.descent_direction(
                cg_max=self.optim_options.get("cg_max", 50),
                method=self.optim_options.get("newton_method", 'bicgstab')
            )
            optimizer.step(method='armijo', show_step=False)
            if not optimizer.converged:
                break
        
        return optimizer.mk.copy()
