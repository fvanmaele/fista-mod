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

for s in range(1, m+1):
    x_fre_bp     = []
    x_fre_omp    = []
    x_fre_mp     = []
    x_fre_iht    = []
    x_fre_cosamp = []
    x_fre_bt     = []
    x_fre_htp    = []
    x_fre_sp     = []

    # TODO: add counters for "TOL not reached"
    for k, x in enumerate(generate_problems(n, s, n_repetitions)):
        b = A @ x

        # 1. basis pursuit
        print("n: {}, sparsity: {}, k: {}, basis pursuit".format(n, s, k))
        xh = algorithms.basis_pursuit(A, b)
        x_fre_bp.append(recovery_error(x, xh))

        # 2. orthogonal matching pursuit
        print("n: {}, sparsity: {}, k: {}, OMP".format(n, s, k))
        xh, conv = algorithms.OMP(A, b, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached") # TODO: add norm(residual) to message
        x_fre_omp.append(recovery_error(x, xh))

        # 3. matching pursuit
        print("n: {}, sparsity: {}, k: {}, MP".format(n, s, k))
        xh, conv = algorithms.MP(A, b, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached")
        x_fre_mp.append(recovery_error(x, xh))

        # 4. iterative hard thresholding
        print("n: {}, sparsity: {}, k: {}, IHT".format(n, s, k))
        xh, conv = algorithms.IHT(A, b, s, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached")
        x_fre_iht.append(recovery_error(x, xh))

        # 5. compressive sampling matching pursuit
        print("n: {}, sparsity: {}, k: {}, CoSaMP".format(n, s, k))
        xh, conv = algorithms.CoSaMP(A, b, s, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached")
        x_fre_cosamp.append(recovery_error(x, xh))
        
        # 6. basic thresholding
        print("n: {}, sparsity: {}, k: {}, BT".format(n, s, k))
        xh = algorithms.BT(A, b, s)
        x_fre_bt.append(recovery_error(x, xh))

        # 7. hard thresholding pursuit
        print("n: {}, sparsity: {}, k: {}, HTP".format(n, s, k))
        xh, conv = algorithms.HTP(A, b, s, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached")
        x_fre_htp.append(recovery_error(x, xh))
        
        # 8. subspace pursuit
        print("n: {}, sparsity: {}, k: {}, SP".format(n, s, k))
        xh, conv = algorithms.SP(A, b, s, tol_res=tol)
        if conv is False:
            print("warning: TOL not reached")
        x_fre_sp.append(recovery_error(x, xh))

        break