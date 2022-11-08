#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISTA and modifications as described in the paper "Improving Fast Iterative Shrinkage-Thresholding
Algorithm: Faster, Smarter and Greedier" by Liang et al. [1]

@author: Ferdinand Vanmaele
"""

import numpy as np


def fista(L, x0, proxR, gradF, max_iter=500, tol_sol=None,
          F=None, R=None, out_conv_sol=None, out_conv_obj=None):
    """
    The original FISTA scheme, to solve objectives in the form:
        Phi(x) = min F(x) + R(x)
        
    where F is continuously differentiable with Lipschitz gradient, and G is a proper convex lsc
    function. The original version has a convergence rate of O(1/k^2) for the objective, but a
    convergence rate for the iterates x_k is unknown. Determining a convergence rate for latter
    lead to a series of modified algorithms.

    The original algorithm may exhibit oscillatory behavior, with regards to the error norm to
    the "real" solution x*. If the objective is strongly convex, there exists an optimal a* such
    that the iteration no longer oscillates. Under weaker conditions, modifications exist which
    aim to reduce this effect (example: restarting FISTA).
    
    The gradient of F and its Lipschitz constant is required, as well as a way to compute the 
    proximal operator of R (or an approximation thereof, if no closed form is available).
    
    Parameters
    ----------
    L : float
        Lipschitz constant for the gradient of F.
    x0 : np.array
        Starting approximation.
    proxR : function(gamma, y)
        A function object computing the proximal operator for the proper convex lsc function R, or 
        an approximation thereof. The function is assumed to take two parameters: the step size 
        gamma, and an input value y.
    gradF : function(y)
        A function object computing the gradient of F, or an approximation thereof. It is assumed
        to be Lipschitz continuous, and takes an input value y.
    max_iter : int, optional
        Maximum number of iterations before the algorithm terminates. Defaults to 500.
    tol_sol : float, optional
        Terminate when ||x{k+1} - x{k}||_2 is less than the given tolerance. Defaults to None.

    Returns
    -------
    xk : np.array
        Approximate solution to the convex optimization problem min F(x) + R(x).
    converged : bool
        True if tol_sol is set and the algorithm converged in k < max_iter steps. False if tol_sol
        is set and the algorithm terminated after max_iter steps. None if tol_sol is None.
    k : int
        The number of iteration steps taken by the algorithm.

    """
    xk_prev = np.copy(x0) # x_{-1}
    xk = np.copy(x0) # x_{0}
    tk = 1
    gamma = 1./L

    for k in range(1, max_iter+1):
        tk_prev = tk
        tk = (1 + np.sqrt(1 + 4*tk_prev**2)) / 2
        ak = (tk_prev - 1) / tk
        yk = xk + ak*(xk - xk_prev)

        xk_prev = np.copy(xk)        
        xk = proxR(gamma, yk - gamma*gradF(yk))
        xk_diff = np.linalg.norm(xk - xk_prev)
        
        # differences for convergence plots
        if out_conv_sol is not None:
            out_conv_sol.append(xk_diff)
        if out_conv_obj is not None:
            out_conv_obj.append((F(xk)+R(xk)) - (F(xk_prev)+R(xk_prev)))

        # termination criterion
        print("FISTA: {} (iter {})".format(xk_diff, k))
        if tol_sol is not None and xk_diff < tol_sol:
            break

    if tol_sol is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return xk, converged, k
