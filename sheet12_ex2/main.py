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

from fista import fista, fista_mod, fista_cd
from face_recognition import soft_thresholding, face_recognition
import matplotlib.pyplot as plt

# %% class to write out a numpy array to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %% compute objective and iterate difference norms in every FISTA step
def experiment(gen, F=None, R=None, tol=-1):
    sol_diff = []
    obj_diff = []

    for k, (xk, xk_prev) in enumerate(gen, start=1):
        sol_diff.append(np.linalg.norm(xk - xk_prev))
        #print(k, sol_diff[-1])

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
def run_trial(candidate, method, B, L, x0, sigma, max_iter, tol):
    """
    Solve the robust face recognition problem with various FISTA modifications.

    Parameters
    ----------
    candidate : list
        Face image to be recovered.
    method : str
        Type of algorithm used. Can be one of 'fista', 'fista_mod' or 'fista_cd'.
    B : np.array
        Matrix B for the (robust) face recognition problem.
    L : float
        Lipschitz constant for the gradient B.T @ (Bw - b).
    sigma : float
        Regularization parameter.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance on the difference norm ||x{k} - x{k-1}|| for iterates.

    Returns
    -------
    exp_data : dict

    """
    F     = lambda w : 1/2 * np.dot(B@w - b, B@w - b)
    gradF = lambda w : B.T @ (B@w - b)
    #R     = L1(sigma=sigma)
    #proxR = lambda gamma, w : R.prox(w, gamma)
    R     = lambda w : sigma * np.linalg.norm(w, ord=1)
    proxR = lambda gamma, w : soft_thresholding(gamma, w, sigma=sigma)

    if method == 'fista':
        generator = fista(L, x0, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(generator, F=F, R=R, tol=tol)  
    
    elif method == 'fista_mod':
        generator = fista_mod(L, x0, 1/20, 1/2, 4, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(generator, F=F, R=R, tol=tol)        
    
    elif method == 'fista_cd':
        generator = fista_cd(L, x0, 20, proxR, gradF, max_iter=max_iter)
        exp_data  = experiment(generator, F=F, R=R, tol=tol)

    else:
        raise RuntimeError("unknown method")

    return exp_data


# %% LFW images with deep funneling
train_set = glob("data_training/*")
verification_set = glob("data_verification/*")
imsize = 130*130


# %% test case
# B, L = face_recognition(train_set, imsize)

# # select n_trials random images from the verification set
# files = np.random.choice(verification_set, 1)
# candidates = []
# for f in files:
#     b = io.imread(f)
#     b = rgb2gray(b).flatten()
#     candidates.append(np.copy(b))

# %%
# sigma = 1
# tol   = 1e-4
# max_iter = 5000
# np.random.seed(None)
# x0 = np.zeros(np.shape(B)[1])
# #x0 = np.random.randn(np.shape(B)[1])
# data = run_trial(candidates[0], 'fista_mod', B, L, x0, sigma, max_iter, tol)

# with open("{}_sigma{:>1.1e}_tol{:>1.1e}_img0.json".format('fista_mod', sigma, tol), 'w') as f:
#     json.dump(data, f, cls=NumpyEncoder)
    
# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # mandatory arguments
    parser.add_argument("trials", type=int, help="number of images checked with dictionary")
    #parser.add_argument("dsize", type=int, help="Number of images selected for the dictionary")

    # optional arguments
    parser.add_argument("--seed", type=int, default=None, help="value for np.random.seed()")
    parser.add_argument("--tol", type=float, default=1e-4, help="threshold for difference ||x{k} - x{k-1}||")
    parser.add_argument("--sigma", type=float, default=1, dest='sigma', help="value of the regularization parameter")
    parser.add_argument("--method", type=str, default='fista_mod', choices=['fista', 'fista_mod', 'fista_cd'], 
                        help='fista algorithm used for numeric tests')
    parser.add_argument("--max-iter", type=str, default=5000, help='maximum number of iterations')
    
    # options for robust face recognition (noise/occlusion)
    parser.add_argument("--robust", action='store_true', help='solve the robust face recognition problem (slow)')
    parser.add_argument("--robust-mean", type=float, default=0, help='mean for noise added to sampled images')
    parser.add_argument("--robust-stddev", type=float, default=0.05, help='standard deviation for noise added to sampled images')
    args = parser.parse_args()

    # generate input data
    B, L = face_recognition(train_set, imsize, seed=args.seed, robust=args.robust, 
                            noise_mean=args.robust_mean, noise_stddev=args.robust_stddev)
    x0 = np.zeros(np.shape(B)[1])

    # select n_trials random images from the verification set
    files = np.random.choice(verification_set, args.trials)
    candidates = []
    for f in files:
        b = io.imread(f)
        b = rgb2gray(b).flatten()
        candidates.append(np.copy(b))

    for i, c in enumerate(candidates):
        data = run_trial(c, args.method, B, L, x0, args.sigma, args.max_iter, args.tol)
        
        with open("{}_sigma{:>1.1e}_tol{:>1.1e}_img{}.json".format(args.method, args.sigma, args.tol, i), 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)
            