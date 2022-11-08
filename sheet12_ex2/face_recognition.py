#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:50:19 2022

@author: archie
"""

import numpy as np
from scipy.io import mmread, mmwrite
from scipy import sparse
from skimage import io
from skimage.color import rgb2gray


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
