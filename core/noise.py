#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
噪声模型模块

提供高斯独立同分布 (IID) 噪声模型

作者: 基于 IPBayesML-FEniCSx09 迁移
"""

import numpy as np
# scipy.sparse 不再需要（仅用于类型兼容）


class NoiseGaussianIID:
    """
    高斯 IID 噪声模型
    
    噪声分布: N(mean, σ² I)
    
    用于建模观测数据中的测量误差
    """
    
    def __init__(self, dim, dtype=np.float64, mean=None, std_dev=None):
        """
        初始化噪声模型
        
        Parameters:
        -----------
        dim : int
            数据维度
        dtype : np.dtype
            数值类型
        mean : np.ndarray or float, optional
            均值向量或标量（默认为0）
        std_dev : float or np.ndarray, optional
            标准差（默认为1）
        """
        assert type(dim) in [int, np.int32, np.int64]
        self.dim = dim
        self.dtype = dtype
        self.mean = np.zeros(self.dim, dtype=dtype)
        
        # 设置均值和标准差
        if mean is not None:
            if isinstance(mean, (int, float)):
                self.mean = np.full(self.dim, mean, dtype=dtype)
            else:
                self.mean = np.array(mean, dtype=dtype)
                
        if std_dev is not None:
            self.std_dev = np.array(std_dev, dtype=dtype)
        else:
            self.std_dev = np.array(1.0, dtype=dtype)
    
    def set_parameters(self, mean=None, std_dev=None):
        """
        设置噪声参数
        
        Parameters:
        -----------
        mean : np.ndarray, optional
            均值向量，默认为零
        std_dev : float or np.ndarray, optional
            标准差，默认为1
        """
        if mean is None:
            self.mean = np.zeros(self.dim, dtype=self.dtype)
        else:
            assert len(mean) == self.dim
            self.mean = np.array(mean, dtype=self.dtype)
        
        if std_dev is None:
            self.std_dev = np.array(1.0, dtype=self.dtype)
        else:
            self.std_dev = np.array(std_dev, dtype=self.dtype)
    
    def eval_CM_inner(self, u, v=None):
        """
        计算协方差矩阵的逆的内积
        
        <u, v>_{Γ^{-1}} = (u-mean)^T Γ^{-1} (v-mean)
        
        对于 IID: = Σ (u_i - mean_i)(v_i - mean_i) / σ²
        
        Parameters:
        -----------
        u : np.ndarray
            第一个向量
        v : np.ndarray, optional
            第二个向量，如果为None则使用u
            
        Returns:
        --------
        val : float
            内积值
        """
        if v is None:
            v = np.copy(u)
        assert len(u) == self.dim
        uu = u - self.mean
        vv = v - self.mean
        val = np.sum(uu * vv) / (self.std_dev**2)
        return val
    
    def generate_sample(self, num: int = 1) -> np.ndarray:
        """
        生成噪声样本
        
        Parameters:
        -----------
        num : int
            生成样本数量（如果>1，返回2D数组）
            
        Returns:
        --------
        sample : np.ndarray
            从 N(mean, σ²I) 采样的随机向量
            如果 num=1，形状为 (dim,)；否则为 (num, dim)
        """
        if num > 1:
            samples = np.zeros((num, self.dim), dtype=self.dtype)
            for i in range(num):
                samples[i] = self.mean + self.generate_sample_zero_mean()
            return samples
        else:
            val = self.mean + self.generate_sample_zero_mean()
            return np.array(val)
    
    def generate_sample_zero_mean(self) -> np.ndarray:
        """
        生成零均值噪声样本
        
        Returns:
        --------
        sample : np.ndarray
            从 N(0, σ²I) 采样的随机向量
        """
        rand_vec = np.random.normal(0, 1, (self.dim,))
        sample = self.std_dev * rand_vec
        return np.array(sample, dtype=self.dtype)
    
    def precision_times_param(self, vec: np.ndarray) -> np.ndarray:
        """
        应用精度矩阵（协方差逆）到向量
        
        Γ^{-1} @ vec = vec / σ²
        
        Parameters:
        -----------
        vec : np.ndarray
            输入向量
            
        Returns:
        --------
        result : np.ndarray
            精度矩阵乘以向量
        """
        return np.array(vec / (self.std_dev**2))
