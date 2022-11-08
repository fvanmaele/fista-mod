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

from fista import fista
from fista_mod import fista_mod, fista_cd
from fista_restart import fista_rada, fista_greedy

# %%
def image_dictionary(files, rowsize, samples):
    colsize = len(samples)
    A = np.zeros((rowsize, colsize))

    for i in range(0, colsize):
        img = io.imread(files[int(samples[i])])
        img = rgb2gray(img)
        A[:, i] = np.copy(img.flatten())

    return A

def soft_thresholding(gamma, x):
    return np.multiply(np.sign(x), np.maximum(0, np.subtract(np.abs(x), gamma)))

# %% Model parameters
import argparse
parser = argparse.ArgumentParser(description='Retrieve arguments')

# mandatory arguments
parser.add_argument("dsize", type=int, help="Number of images selected for the dictionary")

# optional arguments
parser.add_argument("--seed", type=int, default=42, help="value for np.random.seed()")
parser.add_argument("--tol", type=float, default=1e-3, help="threshold for difference ||x{k} - x{k-1}||")
parser.add_argument("--lmb", type=float, default=1e-6, help="value of the regularization parameter")
parser.add_argument("--n-trials", type=int, default=20, help="number of images checked with dictionary")
parser.add_argument("--n-starts", type=int, default=4, help="number of random starting points")
args = parser.parse_args()

dsize = args.dsize
#dsize1 = 440
#dsize2 = 944
#dsize3 = 4412  # TODO: build smaller cases from this one

# %% LFW images with deep funneling
files = glob("data/*")
assert(len(files) == 13233)
imsize = 250*250

# %% Select dsize<i> random images for the dictionary
np.random.seed(args.seed)
smp = np.random.permutation(len(files))[:dsize]
#smp = np.random.permutation(len(files))[:dsize1]
#smp = np.random.permutation(len(files))[:dsize2]
#smp = np.random.permutation(len(files))[:dsize3]

# %% Load images into dictionary
B = sparse.hstack([sparse.csc_matrix(image_dictionary(files, imsize, smp)), sparse.eye(imsize)])
BtB = B.T @ B
L = sparse.linalg.norm(BtB)

# %% Choose 20 different input images 'b' that are not in the dictionary
candidates = np.setdiff1d(range(0, len(files)), smp)
candidates = candidates[np.random.permutation(len(candidates))[:20]]

# %% Random (normally distributed) starting points x0
# XXX: also check x0 = 0
x0_v = np.random.randn(B.shape[1], 4)

# %% TODO: Reference solution using cvxpy
import cvxpy as cp

# %% Apply FISTA
#lmb = 1e-6
lmb = args.lmb
#tol = 1e-3
tol = args.tol

# %%
for b_idx in candidates:
    b = io.imread(files[b_idx])
    b = rgb2gray(b).flatten()

    gradF = lambda w : B.T @ (B@w - b)
    F = lambda w : 1/2 * np.dot(B@w - b, B@w - b)
    R = lambda w : np.linalg.norm(w, ord=1)

    for x0_i in range(0, x0_v.shape[1]):
        x0 = x0_v[:, x0_i]
    
        out_conv_sol = []
        out_conv_obj = []
        xk, converged, k = fista_mod(L, x0, 1/20, 1/2, 4, soft_thresholding, gradF, max_iter=100000, tol_sol=tol,
            F=F, R=R, out_conv_sol=out_conv_sol, out_conv_obj=out_conv_obj)
    
        #xk, converged, k = fista_bt(L, x0, soft_thresholding, gradF, max_iter=5000, tol_sol=1e-3)
        #xk, converged, k = rada_fista(L, x0, 1/20, 1/2, soft_thresholding, gradF, max_iter=5000, tol_sol=1e-3)
        #xk, converged, k = greedy_fista(L, 1.3/L, x0, 1, 0.96, soft_thresholding, gradF, max_iter=500, tol_sol=1e-3)
        break
    break