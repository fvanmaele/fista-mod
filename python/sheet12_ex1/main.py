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
# TODO: repeat experiments for matrix P!
np.random.seed(42)
tol = 10e-6
n_repetitions = 100
max_iter = 500

# TODO: parallelize this code
for s in range(1, m+1):
    print("n: {}, sparsity: {}".format(n, s))

    # TODO: write results to .JSON
    x_fre_bp     = []
    x_fre_omp    = []
    x_fre_mp     = []
    x_fre_iht    = []
    x_fre_cosamp = []
    x_fre_bt     = []
    x_fre_htp    = []
    x_fre_sp     = []

    # TODO: keep tabs on how often an algorithm failed to reach TOL
    for k, x in enumerate(generate_problems(n, s, n_repetitions)):
        b = A @ x

        # 1. basis pursuit
        xh = algorithms.basis_pursuit(A, b)
        x_fre_bp.append(recovery_error(x, xh))

        # 2. orthogonal matching pursuit
        xh, conv = algorithms.OMP(A, b, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_omp.append(recovery_error(x, xh))

        # 3. matching pursuit
        xh, conv = algorithms.MP(A, b, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_mp.append(recovery_error(x, xh))

        # 4. iterative hard thresholding
        xh, conv = algorithms.IHT(A, b, s, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_iht.append(recovery_error(x, xh))

        # 5. compressive sampling matching pursuit
        xh, conv = algorithms.CoSaMP(A, b, s, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_cosamp.append(recovery_error(x, xh))
        
        # 6. basic thresholding
        xh = algorithms.BT(A, b, s)
        x_fre_bt.append(recovery_error(x, xh))

        # 7. hard thresholding pursuit
        xh, conv = algorithms.HTP(A, b, s, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_htp.append(recovery_error(x, xh))
        
        # 8. subspace pursuit
        xh, conv = algorithms.SP(A, b, s, tol_res=tol)
        # if conv is False:
        #     print("warning: TOL not reached")
        x_fre_sp.append(recovery_error(x, xh))

    # Summarize results for sparsity level s
    avg_bp = np.mean(x_fre_bp)
    print("n: {}, sparsity: {}, basis pursuit, average error: {}".format(n, s, avg_bp))
    
    avg_omp = np.mean(x_fre_omp)    
    print("n: {}, sparsity: {}, OMP, average error: {}".format(n, s, avg_omp))
    
    avg_mp = np.mean(x_fre_mp)
    print("n: {}, sparsity: {}, MP, average error: {}".format(n, s, avg_mp))
    
    avg_iht = np.mean(x_fre_iht)
    print("n: {}, sparsity: {}, IHT, average error: {}".format(n, s, avg_iht))
    
    avg_cosamp = np.mean(x_fre_cosamp)
    print("n: {}, sparsity: {}, CoSaMP, average error: {}".format(n, s, avg_cosamp))
    
    avg_bt = np.mean(x_fre_bt)
    print("n: {}, sparsity: {}, BT, average error: {}".format(n, s, avg_bt))
    
    avg_htp = np.mean(x_fre_htp)
    print("n: {}, sparsity: {}, HTP, average error: {}".format(n, s, avg_htp))
    
    avg_sp = np.mean(x_fre_sp)
    print("n: {}, sparsity: {}, SP, average error: {}".format(n, s, avg_sp))
