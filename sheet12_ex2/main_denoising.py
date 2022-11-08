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


# %%
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
    return data


# %%
def main_denoising(imfile, id, sigma, max_iter, tol, rsize=(256, 256), **kwargs):
    # Load images
    image = resize(rgb2gray(io.imread(imfile)), rsize, anti_aliasing=True)
    nx, ny = np.shape(image)
    plt.imsave('image{}.png'.format(id), image, cmap='gray')

    noisy = random_noise(image, kwargs)
    nsize = noisy.size
    plt.imsave('image{}_random_noise.png'.format(id), noisy, cmap='gray')

    # Formulate problem
    orig  = image.flatten()
    b     = noisy.flatten()
    L     = sparse.linalg.norm(sparse.eye(nsize))
    F     = lambda w : 1/2 * np.dot(w - b, w - b)
    gradF = lambda w : w - b

    R     = pyproximal.TV(dims=(nsize, ), sigma=sigma)
    proxR = lambda gamma, w : R.prox(w, gamma)
    x0    = np.zeros(nsize)
    
    # FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(id, 'fista', sigma, tol)
    generator = fista(L, x0, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, image, b ,generator, F=F, R=R, tol_sol=tol)
    
    with open(basename + '.json', 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)

    plt.imsave(basename + '.png', exp_data['solution'].reshape(nx, ny), cmap='gray')

    # Modified FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(id, 'fista_mod', sigma, tol)
    generator = fista_mod(L, x0, 1/20, 1/2, 4, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, orig, b, generator, F=F, R=R, tol_sol=tol)
    
    with open(basename + '.json', 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)
    
    plt.imsave(basename + '.png', exp_data['solution'].reshape(nx, ny), cmap='gray')        

    # Restarting FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(id, 'fista_rada', sigma, tol)
    generator = fista_rada(L, x0, 1/20, 1/2, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, orig, b ,generator, F=F, R=R, tol_sol=tol)
    
    with open(basename + '.json', 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)

    plt.imsave(basename + '.png', exp_data['solution'].reshape(nx, ny), cmap='gray')

    # Greedy FISTA
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(id, 'fista_greedy', sigma, tol)
    generator = fista_greedy(L, 1.3/L, x0, 1, 0.96, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, orig, b ,generator, F=F, R=R, tol_sol=tol)    
    
    with open(basename + '.json', 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)
    
    plt.imsave(basename + '.png', exp_data['solution'].reshape(nx, ny), cmap='gray')

    # FISTA (Chambolle & Dossal)
    basename  = "image{}_{}_sigma{}_tol{:>1.1e}".format(id, 'fista_cd', sigma, tol)
    generator = fista_cd(L, x0, 20, proxR, gradF, max_iter=max_iter)
    exp_data  = experiment(basename, orig, b ,generator, F=F, R=R, tol_sol=tol)
    
    with open(basename + '.json', 'w') as f:
        json.dump(exp_data, f, cls=NumpyEncoder)
        
    plt.imsave(basename + '.png', exp_data['solution'].reshape(nx, ny), cmap='gray')

    
# %%
# max_iter = 5000
# tol = 1e-8
# sigma = 0.09
# noise_var = 0.01


# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # mandatory arguments
    parser.add_argument("n_images", type=int, help="number of images denoised")

    # optional arguments
    parser.add_argument("--method",         type=str,       default='fista_mod', help='fista algorithm used for numeric tests',
                        choices=['fista', 'fista_mod', 'fista_rada', 'fista_greedy', 'fista_cd'])
    parser.add_argument("--tol",            type=float,     default=1e-8,   help="threshold for difference ||x{k} - x{k-1}||")
    parser.add_argument("--sigma",          type=float,     default=0.09,   help="value of the regularization parameter")
    parser.add_argument("--max-iter",       type=int,       default=5000,   help="maximum number of iterations for each method")
    parser.add_argument("--nx",             type=int,       default=256,    help="size to resize input images to")
    parser.add_argument("--ny",             type=int,       default=256,    help="size to resize input images to")
    parser.add_argument("--seed",           type=int,       default=42,     help="value for np.random.seed()")
    
    # options to control noise adedd to images
    parser.add_argument("--noise-mode",     type=str,       default='gaussian', help="type of noise to add to images", 
                        choices=['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
    parser.add_argument("--noise-var",      type=float,     default=0.01,   help="variance of added Gaussian noise")
    parser.add_argument("--noise-mean",     type=float,     default=0,      help="mean of added Gaussian noise")
    parser.add_argument("--noise-svp",      type=float,     default=0.5,    help="proportion of salt vs. pepper noise")
    parser.add_argument("--noise-amount",   type=float,     default=0.05,   help="proportion of image pixels to replace with noise on range [0, 1]")

    args = parser.parse_args()

    # assign variables
    assert args.n_images >= 1
    max_iter     = args.max_iter
    tol          = args.tol
    sigma        = args.sigma
    noise_var    = args.noise_var
    noise_svp    = args.noise_svp
    noise_mode   = args.noise_mode
    noise_mean   = args.noise_mean
    noise_amount = args.noise_amount
    nx           = args.nx
    ny           = args.ny

    # load list of images files in stylegan2 directory
    data = glob('stylegan2/*')
    data.sort()
    
    # run FISTA experiments
    kwargs = {}
    if noise_mode == 'gaussian' or noise_mode == 'speckle':
        kwargs['var']  = noise_var
        kwargs['mean'] = noise_mean

    elif noise_mode == 's&p':
        kwargs['salt_vs_pepper'] = noise_svp
        kwargs['amount'] = noise_amount

    for id, imfile in enumerate(data[:args.n_images]):
        main_denoising(imfile, id, sigma, max_iter, tol, rsize=(nx, ny), mode=noise_mode, **kwargs)