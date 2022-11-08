#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

import numpy as np
#import json

from glob import glob
from skimage import io
from skimage.color import rgb2gray
import scipy.sparse as sparse

from fista import fista_bt, fista_cd, fista_mod, rada_fista, greedy_fista

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
dsize1 = 440
#dsize2 = 944
#dsize3 = 4412  # TODO: build smaller cases from this one

# %% LFW images with deep funneling
files = glob("data/*")
assert(len(files) == 13233)
imsize = 250*250

# %% Select dsize<i> random images for the dictionary
np.random.seed(42)
smp = np.random.permutation(len(files))[:dsize1]
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
x0_v = np.random.randn(B.shape[1], 4)

# %% Apply FISTA
lmb = 1e-6  # TODO: fine tune the parameter lambda for a good result

for b in candidates:
    gradF = lambda w : B.T @ (B@w - b)
    
    for x0_i in range(0, x0_v.shape[1]):
        x0 = x0_v[:, x0_i]
        fista_mod(L, x0, 1/20, 1/2, 4, soft_thresholding, gradF, max_iter=2000, tol_sol=None)
        break
    break
        