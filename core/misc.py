#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core 通用工具函数模块

本模块仅提供与具体 PDE 问题无关的通用工具：
1. 彩色打印工具 (print_my)

注意: 与 Darcy Flow 或其他特定 PDE 相关的函数（如网格生成、
刚度/质量矩阵组装、测量矩阵构建、光滑化算子等）已迁移到各
问题专属目录中（例如 DarcyFlow/misc.py）。

作者: 基于原 IPBayesML-FEniCSx09 重构
"""


def print_my(*string, end=None, color=None):
    """彩色打印函数"""
    if color == 'red':
        print('\033[1;31m', end='')
        if end is None: 
            print(*string)
        else: 
            print(*string, end=end)
        print('\033[0m', end='')
    elif color is None:
        print(*string, end=end)
    elif color == 'blue':
        print('\033[1;34m', end='')
        if end is None: 
            print(*string)
        else: 
            print(*string, end=end)
        print('\033[0m', end='')
    elif color == 'green':
        print('\033[1;32m', end='')
        if end is None: 
            print(*string)
        else: 
            print(*string, end=end)
        print('\033[0m', end='')


__all__ = ['print_my']


if __name__ == "__main__":
    print("=" * 60)
    print("Testing core/misc.py module")
    print("=" * 60)
    
    # 测试彩色打印
    print_my("Red text", color='red')
    print_my("Blue text", color='blue')
    print_my("Green text", color='green')
    print_my("Normal text")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
