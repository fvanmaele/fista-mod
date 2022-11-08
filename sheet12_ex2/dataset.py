#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""
# %%
import numpy as np
import json

from glob import glob
from skimage import io
from skimage.color import rgb2gray
from scipy.sparse import csc_matrix
from scipy.sparse import eye, hstack
from scipy.io import mmwrite

def image_dictionary(files, rowsize, samples):
    colsize = len(samples)
    A = np.zeros((rowsize, colsize))

    for i in range(0, colsize):
        img = io.imread(files[int(samples[i])])
        img = rgb2gray(img)
        A[:, i] = np.copy(img.flatten())

    return A

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open("A_samples_d4412.json", "w") as f:
    json.dump(smp, f, cls=NumpyEncoder)

# %% Load images into dictionary
A = image_dictionary(files, imsize, smp)
A = csc_matrix(A)
B = hstack([A, eye(imsize)])

# %% Save to file for later processing
mmwrite("B_n62500_d4412.mtx", B)
del B