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
from scipy.sparse import csc_matrix
from scipy.sparse import eye, hstack

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
    return np.multiply(np.sign(x), np.maximum(0, np.subtract((np.abs(x), gamma))))

# %% Model parameters
dsize1 = 440
dsize2 = 944
dsize3 = 4412

# %% LFW images with deep funneling
files = glob("data/lfw/*")
assert(len(files) == 13233)
imsize = 250*250

# %% Select dsize<i> random images
np.random.seed(42)
smp = np.random.permutation(len(files))[:dsize3]

# %% Load images into dictionary
B = hstack([csc_matrix(image_dictionary(files, imsize, smp)), eye(imsize)])

# %% Apply FISTA
