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

from fista import fista, fista_mod, fista_cd, fista_rada, fista_greedy
from face_recognition import soft_thresholding, face_recognition


# %% class to write out a numpy array to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %% TODO: write documentation
def experiment(basename, gen, F=None, R=None, tol_sol=-1):
    """
    

    Parameters
    ----------
    basename : TYPE
        DESCRIPTION.
    gen : TYPE
        DESCRIPTION.
    F : TYPE, optional
        DESCRIPTION. The default is None.
    R : TYPE, optional
        DESCRIPTION. The default is None.
    tol_sol : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    sol_diff = []
    obj_diff = []

    for k, (xk, xk_prev) in enumerate(gen, start=1):
        sol_diff.append(np.linalg.norm(xk - xk_prev))

        if F is not None and R is not None:
            Fxk = F(xk) + R(xk)
            Fxk_prev = F(xk_prev) + R(xk)
            obj_diff.append(np.linalg.norm(Fxk - Fxk_prev))
            
        if sol_diff[-1] < tol:
            break

    data = {
        'solution_norm_diff': sol_diff, 
        'objective_norm_diff': obj_diff, 
        'k': k,
        'solution': xk
    }    
    return data


# %%
def run_trial(files, method, B, L, x0, candidate, lmb, dsize, max_iter, tol):
    """
    Solve the robust face recognition problem with various FISTA modifications. Due to the long
    running times for higher dimensions, intermediary results are stored to JSON files for later
    processing.

    Parameters
    ----------
    files : list
        List of file paths to input images. Includes both images for the dictionary and images
        to be recovered by FISTA.
    method : str
        Type of algorithm used. Can be one of 'fista', 'fista_mod', 'fista_rada', 'fista_greedy',
        or 'fista_cd'.
    B : np.array
        Matrix B for the (robust) face recognition problem.
    L : float
        Lipschitz constant for the gradient B.T @ (Bw - b).
    starting_points : np.array
        List of starting approximations x0.
    candidates : list
        List of image indices to be recovered.
    lmb : float
        Regularization parameter.
    dsize : int
        Dictionary size.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the difference norm ||x{k} - x{k-1}|| for iterates.

    Returns
    -------
    exp_data : dict

    """
    b = io.imread(files[candidate])
    b = rgb2gray(b).flatten()
    F = lambda w : 1/2 * np.dot(B@w - b, B@w - b)
    gradF = lambda w : B.T @ (B@w - b)
    R = lambda w : lmb * np.linalg.norm(w, ord=1)
    proxR = lambda gamma, w : soft_thresholding(gamma, w, lmb=lmb)
    
    if method == 'fista':
        generator = fista(L, x0, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(orig, b, generator, F=F, R=R, tol_sol=tol)  
    
    elif method == 'fista_mod':
        generator = fista_mod(L, x0, 1/20, 1/2, 4, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(orig, b, generator, F=F, R=R, tol_sol=tol)        
    
    elif method == 'fista_rada':
        generator = fista_rada(L, x0, 1/20, 1/2, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(orig, b, generator, F=F, R=R, tol_sol=tol)
    
    elif method == 'fista_greedy':
        generator = fista_greedy(L, 1.3/L, x0, 1, 0.96, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(orig, b, generator, F=F, R=R, tol_sol=tol)
    
    elif method == 'fista_cd':
        generator = fista_cd(L, x0, 20, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(orig, b, generator, F=F, R=R, tol_sol=tol)

    else:
        raise ArgumentError("unknown method")

    with open("{}_dsize{}_sigma{:>1.1e}_tol{:>1.1e}_img{}_start{}.json".format(
            method, dsize, lmb, tol, b_idx, x0_idx), 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)

    return exp_data


# %% LFW images with deep funneling
files = glob("data/*")
assert(len(files) > 2000)
imsize = 250*250


# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # mandatory arguments
    parser.add_argument("dsize", type=int, help="Number of images selected for the dictionary")

    # optional arguments
    parser.add_argument("--seed", type=int, default=42, help="value for np.random.seed()")
    parser.add_argument("--tol", type=float, default=1e-4, help="threshold for difference ||x{k} - x{k-1}||")
    parser.add_argument("--sigma", type=float, default=1e-6, dest='sigma', help="value of the regularization parameter")
    parser.add_argument("--n-trials", type=int, default=5, help="number of images checked with dictionary")
    parser.add_argument("--method", type=str, default='fista_mod', choices=['fista', 'fista_mod', 'fista_rada', 'fista_greedy', 'fista_cd'], 
                        help='fista algorithm used for numeric tests')
    parser.add_argument("--max-iter", type=str, default=100000, help='maximum number of iterations')
    
    # options for robust face recognition (noise/occlusion)
    parser.add_argument("--robust", action='store_true', help='solve the robust face recognition problem (slow)')
    parser.add_argument("--robust-mean", type=float, default=0, help='mean for noise added to sampled images')
    parser.add_argument("--robust-stddev", type=float, default=0.05, help='standard deviation for noise added to sampled images')
    args = parser.parse_args()

    # generate input data
    B, L, candidates = face_recognition(files, imsize, args.dsize, args.n_trials, seed=args.seed, 
                                        robust=args.robust, noise_mean=args.robust_mean, noise_stddev=args.robust_stddev)
    # XXX: this seemed to have worked better than a random vector (but I forgot to add noise or occlusion to the tested images...)
    x0 = np.zeros(np.shape(B)[1], 1)

    # run trials for different images
    for cand in candidates:
        run_trials(files, args.method, B, L, x0, cand, args.sigma, args.dsize, args.max_iter, args.tol)