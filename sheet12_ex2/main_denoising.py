#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

import numpy as np

from skimage import io
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.transform import resize

from glob import glob
import json
import matplotlib.pyplot as plt

from scipy import sparse
import pyproximal

from fista import fista, fista_mod, fista_cd, fista_rada, fista_greedy


# %% Class to write out a numpy array to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %%
def PSNR(orig, noisy):
    mse = np.mean((orig - noisy) ** 2)
    if(mse == 0):
        return 100

    psnr = 20 * np.log10(np.max(orig) / np.sqrt(mse))
    return psnr


# %%
def experiment(basename, orig, noisy, gen, F=None, R=None, tol_sol=-1):
    assert np.size(orig) == np.size(noisy)

    sol_diff = []
    obj_diff = []
    sol_psnr = []

    for k, (xk, xk_prev) in enumerate(gen):
        sol_diff.append(np.linalg.norm(xk - xk_prev))
        sol_psnr.append(PSNR(orig, xk))

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
        'solution': xk, 
        'psnr': sol_psnr
    }
    with open(basename + '.json', 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
    
    nx, ny = np.shape(orig)
    plt.imsave(basename + '.png', xk.reshape(nx, ny), cmap='gray')
    
    return data


# %% Compare FISTA variations for 5 different StyleGAN2 images
data = glob('stylegan2/*')
data.sort()
max_iter = 5000
tol = 1e-8
sigma = 0.09
noise_var = 0.01


# %%
for k, id in enumerate(data[:5]):
    # Load images
    image = resize(rgb2gray(io.imread(id)), (256, 256), anti_aliasing=True)
    plt.imsave('image{}.png'.format(k), image, cmap='gray')

    noisy = random_noise(image, mode='gaussian', var=noise_var)
    nsize = noisy.size
    plt.imsave('image{}_gaussian_var{}.png'.format(k, noise_var), noisy, cmap='gray')

    # Formulate problem
    b     = noisy.flatten()
    L     = sparse.linalg.norm(sparse.eye(nsize))
    F     = lambda w : 1/2 * np.dot(w - b, w - b)
    gradF = lambda w : w - b

    R     = pyproximal.TV(dims=(nsize, ), sigma=sigma)
    proxR = lambda gamma, w : R.prox(w, gamma)
    x0    = np.zeros(nsize)
    
    # FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(k, 'fista', sigma, tol)
    generator = fista(L, x0, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b ,generator, F=F, R=R, tol_sol=tol)
    
    # Modified FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(k, 'fista_mod', sigma, tol)
    generator = fista_mod(L, x0, 1/20, 1/2, 4, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b, generator, F=F, R=R, tol_sol=tol)
            
    # Restarting FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(k, 'fista_rada', sigma, tol)
    generator = fista_rada(L, x0, 1/20, 1/2, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b ,generator, F=F, R=R, tol_sol=tol)
        
    # Greedy FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(k, 'fista_greedy', sigma, tol)
    generator = fista_greedy(L, 1.3/L, x0, 1, 0.96, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b ,generator, F=F, R=R, tol_sol=tol)    
    
    # FISTA (Chambolle & Dossal)
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(k, 'fista_cd', sigma, tol)
    generator = fista_cd(L, x0, 20, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b ,generator, F=F, R=R, tol_sol=tol)