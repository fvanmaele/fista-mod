import numpy as np
from scipy.linalg import dft
"""
Construction of sensor matrices for Compressed Sensing, Sheet 12, Exercise 1.

@author: Ferdinand Vanmaele
"""

def sensor_normalize(A, axis=0, ord=None):
    """
    

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.
    ord : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    func1d = lambda a: np.divide(a, np.linalg.norm(a, ord=ord))
    return np.apply_along_axis(func1d, axis, A)


def sensor_random(m, n, mean=0, std=1, seed=None):
    """
    

    Parameters
    ----------
    m : int
        DESCRIPTION.
    n : int
        DESCRIPTION.
    mean : float, optional
        DESCRIPTION. The default is 0.
    std : float, optional
        DESCRIPTION. The default is 1.
    seed : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    """
    assert m > 0, "m must be positive"
    assert n > 0, "n must be positive"

    # Initialize random state
    np.random.seed(seed)
    
    # Generate matrix with normally distributed entries
    A = np.random.normal(loc=mean, scale=std, size=(m, n))
    
    # Normalize the columns of A in the 2-norm
    A = sensor_normalize(A, axis=0)
    
    return A


def sensor_random_partial_fourier(m, n, seed=None):
    """
    

    Parameters
    ----------
    m : int
        DESCRIPTION.
    n : int
        DESCRIPTION.
    seed : int, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    assert n > 0, "n must be positive"
    assert m <= n, "m must be less or equal n"

    # Initialize random state
    np.random.seed(seed)
    
    # Normalized discrete Fourier transform matrix of dimension n
    A = sensor_normalize(dft(n), axis=0)
    
    # Choose m rows uniformly at random
    idx = np.random.permutation(n)[:m]

    # Construct partial Fourier matrix
    return A[idx, :]