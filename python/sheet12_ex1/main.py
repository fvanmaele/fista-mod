#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 23:37:07 2022

@author: Ferdinand Vanmaele
"""

import numpy as np
import algorithms
from sensors import sensor_random, sensor_random_partial_fourier

# %% Generate matrices with normalized columns
np.random.seed(42)
n = 2**7
m = n//2
A = sensor_random(m, n)
P = sensor_random_partial_fourier(m, n)

# %% Generate problem instances
def generate_problems(n, s, n_reps):
    xs = []
    for k in range(0, n_reps):
        x = np.zeros(n)
        x_nnz = np.random.randn(s)          # nonzero values from standard Gaussian distribution
        S = np.random.permutation(n)[:s]    # nonzero locations uniformly at random
    
        x[S] = x_nnz
        assert np.nonzero(x)[0].size == s   # sanity check
        xs.append(np.copy(x))
    
    return xs

def recovery_error(x, xh, ord=None):
    return np.linalg.norm(x - xh, ord=ord) / np.linalg.norm(x)

# %% # Recover xh from measurements b = Ax, for every problem instance of sparsity 1 <= s <= m
tol = 10e-6
n_repetitions = 100

for s in range(1, m+1):
    xs_fre_bp     = np.array([])
    xs_fre_omp    = np.array([])
    xs_fre_mp     = np.array([])
    xs_fre_iht    = np.array([])
    xs_fre_cosamp = np.array([])
    xs_fre_htp    = np.array([])
    xs_fre_sp     = np.array([])

    for x in generate_problems(n, s, n_repetitions):
        b = A @ x

        # 1. basis pursuit
        xh = algorithms.basis_pursuit(A, b)
        xs_fre_bp.append(recovery_error(x, xh))

        # 2. orthogonal matching pursuit
        xh = algorithms.OMP(A, b)
        # 3. matching pursuit
    
        # 4. iterative hard thresholding
        
        # 5. compressive sampling matching pursuit
        
        # 6. basic thresholding
        
        # 7. hard thresholding pursuit
        
        # 8. subspace pursuit
        