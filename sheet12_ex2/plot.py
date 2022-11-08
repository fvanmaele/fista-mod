#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.io import mmread
from skimage import io
from skimage.color import rgb2gray
import scipy.sparse as sparse

# %%
files = glob("data/*")
#assert(len(files) == 13233)
imsize = 250*250

# %%
fista_mod = glob("fista_mod_*.json")
fista_mod.sort()
trials = []

for trial in fista_mod:
    with open(trial, 'r') as j:
        trials.append(json.load(j))

# %%
B = mmread('B_dsize1000_seed42_imsize62500_files{}.mtx'.format(len(files)))
#B = sparse.csc_matrix(B)

# %%
exp1 = trials[0]
xk1  = exp1['solution']

# %%
plt.imshow((B@xk1).reshape(250, 250))

# %%
plt.imshow(rgb2gray(io.imread(files[1520])))

# %%
plt.plot(range(0, 1000), xk1)