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
def face_recognition(train_set, imsize, robust=False):
    """
    Generate input data for solving the robust face recognition problem using convex optimization
    solvers. The dictionary is built from a collection of training images, all assumed
    to be of the same size. Images are then converted to grayscale and stacked as column vectors.
    
    The returned matrix is a sparse concatenation of a dense matrix A of dimension (m, n) and the 
    identity matrix (m, m), where n << m.

    Parameters
    ----------
    train_set : list
        List of paths pointing to image files. All images are assumed to have the same size.
    imsize : int
        Fixed image size m*n of image files.
    seed : int, optional
        Value for np.random.seed. Defaults to None.

    Returns
    -------
    B : np.array
        Sparse matrix for robust face recognition.
    L : float
        Lipschitz constant for grad F(w) = B.T @ (Bw - b)

    """
    colsize = len(train_set)
    
    # Load images into dictionary
    if robust is True:
        filename = 'B_dsize{}_imsize{}_robust'.format(colsize, imsize)
    else:
        filename = 'B_dsize{}_imsize{}'.format(colsize, imsize)

    try:
        B = mmread(filename)
    except:
        A = np.zeros((imsize, colsize))
    
        for i in range(0, colsize):
            img = io.imread(train_set[i])
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

    return B, L
    

def soft_thresholding(gamma, w, sigma=1):
    """
    Proximal operator for the scaled l1 norm lambda*||w||_1.

    Parameters
    ----------
    gamma : float
        Scaling factor for the proximal operator.
    w : np.array
        Input vector.
    sigma : float, optional
        Scaling factor the l1 norm. The default is 1.

    Returns
    -------
    np.array

    """
    return np.multiply(np.sign(w), np.maximum(0, np.subtract(np.abs(w), sigma*gamma)))
