#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

import numpy as np
import json

from glob import glob
from skimage import io
from skimage.color import rgb2gray
import scipy.sparse as sparse
from scipy.io import mmread, mmwrite

from fista import fista
from fista_mod import fista_mod, fista_cd
from fista_restart import fista_rada, fista_greedy

# %%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def image_dictionary(files, rowsize, samples):
    colsize = len(samples)
    A = np.zeros((rowsize, colsize))

    for i in range(0, colsize):
        img = io.imread(files[int(samples[i])])
        img = rgb2gray(img)
        A[:, i] = np.copy(img.flatten())

    return A


# %%
def setup(files, imsize, dictsize, n_starts, seed):
    np.random.seed(seed=seed)

    # Select dsize<i> random images for the dictionary
    smp = np.random.permutation(len(files))[:dictsize]

    # Load images into dictionary
    filename = 'B_dsize{}_seed{}_imsize{}_files{}.mtx'.format(dictsize, seed, imsize, len(files))
    try:
        B = mmread(filename)
    except:
        B = sparse.hstack([sparse.csc_matrix(image_dictionary(files, imsize, smp)), 
                           sparse.eye(imsize)], format='csc')
        mmwrite(filename, B)

    BtB = B.T @ B
    L = sparse.linalg.norm(BtB)
    
    # Choose 20 different input images 'b' that are not in the dictionary
    candidates = np.setdiff1d(range(0, len(files)), smp)
    candidates = candidates[np.random.permutation(len(candidates))[:20]]
    
    # Random starting points x0
    starting_points = np.random.randn(B.shape[1], n_starts)
    
    return B, L, starting_points, candidates


def soft_thresholding(gamma, w, Lambada=1):
    return np.multiply(np.sign(w), np.maximum(0, np.subtract(np.abs(w), Lambada*gamma)))


def run_trials(method, B, L, starting_points, candidates, Lambada, dsize, max_iter, tol):
    for b_idx in candidates:
        b = io.imread(files[b_idx])
        b = rgb2gray(b).flatten()

        F = lambda w : 1/2 * np.dot(B@w - b, B@w - b)
        R = lambda w : Lambada * np.linalg.norm(w, ord=1)
        gradF = lambda w : B.T @ (B@w - b)
        proxR = lambda gamma, w : soft_thresholding(gamma, w, Lambada=Lambada)

        for x0_idx in range(0, starting_points.shape[1]):
            x0 = starting_points[:, x0_idx]
            out_conv_sol = []
            out_conv_obj = []
        
            if method == 'fista':
                xk, converged, k = fista(L, x0, proxR, gradF, max_iter=max_iter, tol_sol=tol,
                                         F=F, R=R, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)
            elif method == 'fista_mod':
                xk, converged, k = fista_mod(L, x0, 1/20, 1/2, 4, proxR, gradF, max_iter=max_iter, tol_sol=tol,
                                             F=F, R=R, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)
            elif method == 'fista_rada':
                xk, converged, k = fista_rada(L, x0, 1/20, 1/2, proxR, gradF, max_iter=max_iter, tol_sol=tol,
                                              F=F, R=R, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)
            elif method == 'fista_greedy':
                xk, converged, k = fista_greedy(L, 1.3/L, x0, 1, 0.96, proxR, gradF, max_iter=max_iter, tol_sol=tol,
                                                F=F, R=R, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)
            elif method == 'fista_cd':
                xk, converged, k = fista_cd(L, x0, 20, proxR, gradF, max_iter=max_iter, tol_sol=tol,
                                            F=F, R=F, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)

            with open("{}_dsize{}_lambda{:>1.1e}_tol{:>1.1e}_img{}_start{}.json".format(method, dsize, Lambada, tol, b_idx, x0_idx), 'w') as f:
                json.dump({'solution_norm_diff': out_conv_sol, 'objective_norm_diff': out_conv_obj, 'k': k, 'converged': converged, 'solution': xk}, f, cls=NumpyEncoder)
    
    #return xk  # TODO: array of solutions



# %% LFW images with deep funneling
files = glob("data/*")
assert(len(files) == 13233)
imsize = 250*250

# %% Setup phase
# seed = 42
# dsize = 994
# B, L, starting_points, candidates = setup(files, imsize, dsize, 42)

# %%
# lmb = 1e-6
# tol = 1e-3
# max_iter = 100000
# method = 'fista_mod'
# xk = run_trials(method, B, L, starting_points, candidates, lmb, max_iter, tol)

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # mandatory arguments
    parser.add_argument("dsize", type=int, help="Number of images selected for the dictionary")

    # optional arguments
    parser.add_argument("--seed", type=int, default=42, help="value for np.random.seed()")
    parser.add_argument("--tol", type=float, default=1e-4, help="threshold for difference ||x{k} - x{k-1}||")
    parser.add_argument("--parameter", type=float, default=1e-6, help="value of the regularization parameter")
    parser.add_argument("--n-trials", type=int, default=4, help="number of images checked with dictionary")
    parser.add_argument("--n-starts", type=int, default=2, help="number of random starting points (standard normal distribution)")
    parser.add_argument("--method", type=str, default='fista_mod', choices=['fista', 'fista_mod', 'fista_rada', 'fista_greedy', 'fista_cd'], 
                        help='fista algorithm used for numeric tests')
    parser.add_argument("--max-iter", type=str, default=100000, help='maximum number of iterations')
    args = parser.parse_args()

    starting_points = np.zeros((imsize+args.dsize, 1)) # FIXME
    B, L, _, candidates = setup(files, imsize, args.dsize, args.n_starts, args.seed)
    

    run_trials(args.method, B, L, starting_points, candidates, args.parameter, args.dsize, args.max_iter, args.tol)