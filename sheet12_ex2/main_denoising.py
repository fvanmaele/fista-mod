#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

import numpy as np

from skimage import io
from skimage.color import rgb2gray
from glob import glob
import matplotlib.pyplot as plt

from scipy import sparse
import pyproximal

from fista import fista
from fista_mod import fista_mod, fista_cd
from fista_restart import fista_rada, fista_greedy

# %%
def add_gaussian_noise(image, mean=0, stddev=0.25):
    return image + np.random.normal(loc=mean, scale=stddev, size=np.shape(image))
    
# %%
data = glob('stylegan2/*')
data.sort()
maxit = 10000
tol   = 1e-12
sigma = 0.5

# %%
for k, id in enumerate(data[:20]):
    image = rgb2gray(io.imread(id))
    plt.imsave('image_{}.png'.format(k), image, cmap='gray')
    
    noisy = np.array(add_gaussian_noise(image))
    nsize = noisy.size
    plt.imsave('image_{}_noisy.png'.format(k), noisy, cmap='gray')

    gradF = lambda w : w - noisy.flatten()
    L     = sparse.linalg.norm(sparse.eye(nsize))
    TV    = pyproximal.TV(dims=(nsize, ), sigma=sigma)
    proxR = lambda gamma, w : TV.prox(w, gamma)

    xk, _, _ = fista_mod(L, 0, 1/20, 1/2, 4, proxR, gradF, max_iter=maxit, tol_sol=tol)
    plt.imsave('image_{}_fista_mod.png'.format(k), xk.reshape(1024, 1024), cmap='gray')
