import numpy as np
from scipy.linalg import dft
"""
Construction of sensor matrices for Compressed Sensing, Sheet 12, Exercise 1.

@author: Ferdinand Vanmaele
"""

def sensor_normalize(A, axis=0, ord=None):
    """
    Normalize the columns or rows of a given matrix. 

    Parameters
    ----------
    A : np.array
        Input matrix.
    axis : int, optional
        When 0 (default), normalize columns. When 1, normalize rows.
    ord : int, optional
        The type of norm as specified in np.linalg.norm. Defaults to None (2-norm for vectors).

    Returns
    -------
    np.array
        Matrix with normalized columns or rows.

    """
    func1d = lambda a: np.divide(a, np.linalg.norm(a, ord=ord))
    return np.apply_along_axis(func1d, axis, A)


def sensor_random(m, n, mean=0, std=1, seed=None, normalize=True):
    """
    Construct a matrix of dimension (m, n) with normally distributed entries. 

    Parameters
    ----------
    m : int
        Number of rows of the generated matrix.
    n : int
        Number of columns of the generated matrix.
    mean : float, optional
        Mean for the normal distribution. The default is 0.
    std : float, optional
        Standard deviation for the normal distribution. The default is 1.
    seed : int, optional
        Value for np.random.seed. The default is None.
    normalize : bool, optional
        If True, normalize the columns of A in the 2-norm. The default is True.

    Returns
    -------
    A : np.array

    """
    assert m > 0, "m must be positive"
    assert n > 0, "n must be positive"

    # Initialize random state
    np.random.seed(seed)
    
    # Generate matrix with normally distributed entries
    A = np.random.normal(loc=mean, scale=std, size=(m, n))
    
    # Normalize the columns of A in the 2-norm
    if normalize:
        A = sensor_normalize(A, axis=0)
    
    return A


def sensor_random_partial_fourier(m, n, seed=None, normalize=True):
    """
    Generate a discrete Fourier transform matrix of dimension (n, n), with m rows chosen
    uniformly at random.

    Parameters
    ----------
    m : int
        Number of rows of the generated matrix..
    n : int
        Number of columns of the generated matrix..
    seed : int, optional
        Value for np.random.seed. The default is None.
    normalize : bool, optional
        If True, normalize the columns of A in the 2-norm. The default is True.

    Returns
    -------
    A : np.array

    """
    assert n > 0, "n must be positive"
    assert m <= n, "m must be less or equal n"

    # Initialize random state
    np.random.seed(seed)
    
    # Discrete Fourier transform matrix of dimension n
    A = dft(n)
    
    # Choose m rows uniformly at random
    idx = np.random.permutation(n)[:m]

    # Normalize the columns of A in the 2-norm
    # Construct partial Fourier matrix with normalized columns
    if normalize:
        return sensor_normalize(A[idx, :], axis=0)
    else:
        return A[idx, :]
    