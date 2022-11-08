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

# %% Class to write out a numpy array to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %%
def face_recognition(files, imsize, dictsize, n_trials, noise_mean=0, noise_stddev=0.1, seed=None, robust=False):
    """
    Generate input data for solving the robust face recognition problem using convex optimization
    solvers. The dictionary is built from randomly sampling a collection of images, all assumed
    to be of the same size. Images are then converted to grayscale and stacked as column vectors.
    
    The returned matrix is a sparse concatenation of a dense matrix A of dimension (m, n) and the 
    identity matrix (m, m), where n << m.

    Parameters
    ----------
    files : list
        List of paths pointing to image files. All images are assumed to have the same size.
    imsize : int
        Fixed image size m*n of image files.
    dictsize : int
        The number of random images sampled.
    n_trials : int
        The number of images to be reconstructed. Candidates are taken from the complement of
        images in the dictionary.
    seed : int, optional
        Value for np.random.seed. Defaults to None.

    Returns
    -------
    B : np.array
        Sparse matrix for robust face recognition.
    L : float
        Lipschitz constant for grad F(w) = B.T @ (Bw - b)
    candidates : list
        Indices of images to be reconstructed.

    """
    np.random.seed(seed=seed)

    # Select dsize<i> random images for the dictionary
    # TODO: sample persons, instead images? (uniform distribution of faces in dictionary)
    # -> use people dictionary as input argument (see filter_data.py)
    samples = np.random.permutation(len(files))[:dictsize]
    colsize = len(samples)
    
    # Load images into dictionary
    if robust is True:
        filename = 'B_dsize{}_seed{}_imsize{}_files{}_robust'.format(dictsize, seed, imsize, len(files))
    else:
        filename = 'B_dsize{}_seed{}_imsize{}_files{}'.format(dictsize, seed, imsize, len(files))

    try:
        B = mmread(filename)
    except:
        A = np.zeros((imsize, colsize))
    
        for i in range(0, colsize):
            img = io.imread(files[int(samples[i])])
            img = rgb2gray(img)
            
            A[:, i] = np.copy(img.flatten())

        if robust is True:
            B = sparse.hstack([sparse.csc_matrix(A), sparse.eye(imsize)], format='csc')
        else:
            B = A
        
        mmwrite(filename, B)

    # Compute Lipschitz constant of l2 gradient
    BtB = B.T @ B
    if sparse.issparse(B):
        L = sparse.linalg.norm(BtB)
    else:
        L = np.linalg.norm(BtB)
    
    # Choose N different input images 'b' that are not in the dictionary
    # TODO: when robust is True, add noise (or occlusion) to the random images 
    # (alternatively, do so in run_trials)
    candidates = np.setdiff1d(range(0, len(files)), samples)
    candidates = candidates[np.random.permutation(len(candidates))[:n_trials]]
    
    return B, L, candidates
    

def soft_thresholding(gamma, w, lmb=1):
    """
    Proximal operator for the scaled l1 norm lambda*||w||_1.

    Parameters
    ----------
    gamma : float
        Scaling factor for the proximal operator.
    w : np.array
        Input vector.
    lmb : float, optional
        Scaling factor the l1 norm. The default is 1.

    Returns
    -------
    np.array

    """
    return np.multiply(np.sign(w), np.maximum(0, np.subtract(np.abs(w), lmb*gamma)))


def run_trials(files, method, B, L, starting_points, candidates, lmb, dsize, max_iter, tol):
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
    None.

    """
    for b_idx in candidates:
        b = io.imread(files[b_idx])
        b = rgb2gray(b).flatten()

        F = lambda w : 1/2 * np.dot(B@w - b, B@w - b)
        R = lambda w : lmb * np.linalg.norm(w, ord=1)
        gradF = lambda w : B.T @ (B@w - b)
        proxR = lambda gamma, w : soft_thresholding(gamma, w, lmb=lmb)

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

            with open("{}_dsize{}_lambda{:>1.1e}_tol{:>1.1e}_img{}_start{}.json".format(method, dsize, lmb, tol, b_idx, x0_idx), 'w') as f:
                json.dump({'solution_norm_diff': out_conv_sol, 'objective_norm_diff': out_conv_obj, 'k': k, 'converged': converged, 'solution': xk}, f, cls=NumpyEncoder)
    

# %% LFW images with deep funneling
files = glob("data/*")
assert(len(files) > 2000)
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
    parser.add_argument("--lambda", type=float, default=1e-6, dest='parameter', help="value of the regularization parameter")
    parser.add_argument("--n-trials", type=int, default=5, help="number of images checked with dictionary")
    parser.add_argument("--method", type=str, default='fista_mod', choices=['fista', 'fista_mod', 'fista_rada', 'fista_greedy', 'fista_cd'], 
                        help='fista algorithm used for numeric tests')
    parser.add_argument("--max-iter", type=str, default=100000, help='maximum number of iterations')
    parser.add_argument("--robust", action='store_true', help='solve the robust face recognition problem (slow)')
    parser.add_argument("--robust-mean", type=float, default=0, help='mean for noise added to sampled images')
    parser.add_argument("--robust-stddev", type=float, default=0.1, help='standard deviation for noise added to sampled images')
    args = parser.parse_args()

    # generate input data
    B, L, candidates = face_recognition(files, imsize, args.dsize, args.n_trials, seed=args.seed, 
                                        robust=args.robust, noise_mean=args.robust_mean, noise_stddev=args.robust_stddev)
    # XXX: this seemed to have worked better than a random vector (but I forgot to add noise to the tested images...)
    starting_points = np.zeros((np.shape(B)[1], 1))

    # run trials for different images
    run_trials(files, args.method, B, L, starting_points, candidates, 
               args.parameter, args.dsize, args.max_iter, args.tol)