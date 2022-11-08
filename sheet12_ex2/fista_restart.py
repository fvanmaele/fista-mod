#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreatFISTA and modifications as described in the paper "Improving Fast Iterative Shrinkage-Thresholding
Algorithm: Faster, Smarter and Greedier" by Liang et al. [1]

@author: Ferdinand Vanmaele
"""

import numpy as np


def fista_rada(L, x0, p, q, proxR, gradF, eps_m=20, eps=None, reset_t=False, max_iter=500, tol_sol=None):
    """
    A version of FISTA which uses a restarting technique to reduce oscillations in the iterates
    x_k. The basic idea of restarting is that, once the objective function value of Phi(x_k) is
    about to increase, the algorithm resets t_k and y_k. Doing so, the algorithm achieves an almost
    monotonic convergence in terms Phi(x_k) - Phi(x*) for the global minimum x*. [1, 5.2]
    
    The restarting technique is defined as setting r = eps*r and y_k = x_k if
        (y_k - x_{k+1})' (x_{k+1} - x_k) >= 0

    holds. Optionally, t_k is also reset to 1. When not set explicitly, the parameter eps is set
    on the first restart as eps = a_k^(1/m), for some sufficiently large m > 1.
    
    The benefit in the "gradient scheme" above over an evaluation F(x_k) > F(x_{k-1}) is that
    all quantities are already calculated, so that only an additional dot product is required.

    Parameters
    ----------
    L : float
        Lipschitz constant for the gradient of F.
    x0 : np.array
        Starting approximation.
    p, q : float
        Parameters for the update rule t_k. p and q are required in the half-open interval (0, 1].
    option : int, optional
        When True, reset t_k=1 in the restarting scheme. Defaults to False.
    eps_m : int, optional
        Parameter m > 1 for determining eps(a_k, m). Defaults to 20.
    eps : float, optional
        Fixed value < 1 for parameter eps. Defaults to None.
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
    np.array
        Approximate solution to the convex optimization problem min F(x) + R(x).

    """
    assert p > 0 and p <= 1, 'p must belong to (0, 1]'
    assert q > 0 and q <= 1, 'q must belong to (0, 1]'
    assert eps_m > 1, 'eps_m must be an integer larger 1'

    xk_prev = np.copy(x0) # x_{-1}
    xk = np.copy(x0) # x_{0}
    tk = 1
    r  = 4
    gamma = 1./L
    n_restarts = 0

    for k in range(1, max_iter+1):
        tk_prev = tk
        tk = (p + np.sqrt(q + r*tk_prev**2)) / 2
        ak = (tk_prev - 1) / tk
        yk = xk + ak*(xk - xk_prev)

        xk_prev = np.copy(xk)        
        xk = proxR(gamma, yk - gamma*gradF(yk))
        xk_diff = np.linalg.norm(xk - xk_prev)
        
        # termination criterion
        print("RadaFISTA: {} (iter {})".format(xk_diff, k))
        if tol_sol is not None and xk_diff < tol_sol:
            break

        # Restarting with automatic choice of eps(ak)
        if np.dot(yk - xk, xk - xk_prev) >= 0:
            n_restarts += 1
            print("RadaFISTA: Restart!")
            if n_restarts == 1 and eps is None:
                eps = ak**(1./eps_m)
            assert eps < 1

            r = r*eps
            yk = xk_prev
            if reset_t is True:
                tk = 1

    if tol_sol is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return xk, converged, k


# TODO: complete documentation
def fista_greedy(L, gamma, x0, S, eps, proxR, gradF, max_iter=500, tol_sol=None):
    """
    A variation on a restarting FISTA scheme which uses a larger step-size than 1/L in the proximal
    operator. The larger step-size can further shorten the oscillation period, but may lead to
    divergence. This leads to the introduction of a "safeguard" step which shrinks the step-size
    when a certain condition is satisfied.
    
    Parameters
    ----------

    Returns
    -------
    None.

    """
    assert gamma >= 1./L and gamma < 2./L
    assert eps < 1
    assert S >= 1
    
    xk_prev = np.copy(x0) # x_{-1}
    xk = np.copy(x0) # x_{0}
    n_restarts = 0

    for k in range(1, max_iter+1):
        yk = xk + (xk - xk_prev)
        xk_prev = np.copy(xk)
        xk = proxR(gamma, yk - gamma*gradF(yk))
        xk_diff = np.linalg.norm(xk - xk_prev)
        
        # Termination criterion
        print("GreedyFISTA: {} (iter {})".format(xk_diff, k))
        if tol_sol is not None and xk_diff < tol_sol:
            break
        # Copy of first iterate for safeguard
        if k == 1:
            x1 = np.copy(xk)
        # Restarting
        if np.dot(yk - xk, xk - xk_prev) >= 0:
            n_restarts += 1
            yk = xk_prev        
        # Safeguard
        if k >= 1 and np.linalg.norm(xk - xk_prev) > S * np.linalg.norm(x1 - x0):
            gamma = np.max([eps*gamma, 1./L])
        
    if tol_sol is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return xk, converged, k
        