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
from skimage.util import random_noise

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
        xdiff = np.linalg.norm(xk - xk_prev)
        sol_diff.append(xdiff)

        if F is not None and R is not None:
            Fxk = F(xk) + R(xk)
            Fxk_prev = F(xk_prev) + R(xk)
            Fdiff = np.linalg.norm(Fxk - Fxk_prev)

            obj_diff.append(Fdiff)
            print(k, xdiff, Fdiff)

        if xdiff < tol:
            break

    data = {
        'solution_norm_diff': sol_diff, 
        'objective_norm_diff': obj_diff, 
        'k': k,
        'solution': xk,
        'solution_argsort': np.flip(np.argsort(xk))
    }
    return data


# %%
def run_trial(b, method, B, L, x0, sigma, max_iter, tol):
    """
    Solve the robust face recognition problem with various FISTA modifications.

    Parameters
    ----------
    b : np.array
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
train_set.sort()
verification_set = glob("data_verification/*")
verification_set.sort()
imsize = 130*130

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # Mandatory arguments
    parser.add_argument("trials", type=int, help="number of images checked with dictionary")
    #parser.add_argument("dsize", type=int, help="Number of images selected for the dictionary")

    # Optional arguments
    parser.add_argument("--seed", type=int, default=None, help="value for np.random.seed()")
    parser.add_argument("--tol", type=float, default=1e-4, help="threshold for difference ||x{k} - x{k-1}||")
    parser.add_argument("--sigma", type=float, default=1, dest='sigma', help="value of the regularization parameter")
    parser.add_argument("--method", type=str, default='fista_mod', choices=['fista', 'fista_mod', 'fista_cd'], 
                        help='fista algorithm used for numeric tests')
    parser.add_argument("--max-iter", type=int, default=10000, help='maximum number of iterations')

    # Options for robust face recognition (noise/occlusion)
    parser.add_argument("--robust", action='store_true', help='solve the robust face recognition problem (slow)')
    parser.add_argument("--robust-mean", type=float, default=0, help='mean for noise added to sampled images')
    parser.add_argument("--robust-var", type=float, default=0.02, help='variance for noise added to sampled images')
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    # Generate dictionary from training set
    Bn = len(train_set) + imsize if args.robust else len(train_set)
    print("Constructing dictionary... [robust={}, m={}, n={}]".format(args.robust, imsize, Bn))
    B, L = face_recognition(train_set, imsize, robust=args.robust)
    x0 = np.zeros(np.shape(B)[1])

    # Select random images from the verification set
    candidates_idx = np.random.choice(range(0, len(verification_set)), args.trials)
    candidates = []
    for sample in candidates_idx:
        b = io.imread(verification_set[sample])
        b = rgb2gray(b).flatten()
        if args.robust:
            b = random_noise(b, mode='gaussian', seed=args.seed,
                             mean=args.robust_mean, var=args.robust_var)
        candidates.append(np.copy(b))

    # Solve (robust) face recognition problem
    for i, sample in enumerate(candidates_idx):
        b = candidates[i]

        # Write out input image
        inputname = "img{}_input".format(sample)
        if args.robust:
            inputname += "_noisy"
        plt.imsave(inputname + ".jpg", b.reshape(130, 130), cmap='gray')

        # Apply FISTA to (robust) face recongition
        data = run_trial(b, args.method, B, L, x0, args.sigma, args.max_iter, args.tol)
        
        # Store JSON data for later processing
        basename = "img{}_{}_sigma{:>1.1e}_tol{:>1.1e}".format(
            sample, args.method, args.sigma, args.tol)
        if args.robust:
            basename += "_robust"
        with open(basename + ".json", 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)

        # Write out recovered image B@x*
        plt.imsave(basename + "_recovered.jpg",
                   (B @ data['solution']).reshape(130, 130), cmap='gray')
        
        # Write out recognized images (corresponding to indices pof highest magnitude)
        for i in range(0, 3):
            if args.robust:
                recognized_i = B[:, data['solution_argsort'][i]].todense()
            else:
                recognized_i = B[:, data['solution_argsort'][i]]

            plt.imsave(basename + "_recognized{}.jpg".format(i), recognized_i.reshape(130, 130), cmap='gray')
            
