#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISTA and modifications as described in the paper "Improving Fast Iterative Shrinkage-Thresholding
Algorithm: Faster, Smarter and Greedier" by Liang et al. [1]

@author: Ferdinand Vanmaele
"""

import numpy as np


def fista(L, x0, proxR, gradF, max_iter=500):
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

    Returns
    -------
    xk, xk_prev : np.array
        Approximate solution to the convex optimization problem min F(x) + R(x).

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

        yield xk, xk_prev


def fista_mod(L, x0, p, q, r, proxR, gradF, max_iter=500):
    """
    Modified version of FISTA which allows to regulate the step size t_k, by using the update rule:
        t_k = (p + sqrt(q + r*t_{k-1}^2)) / 2,  a_k = (t_{k-1} - 1)/k
        
    where p, q are in the interval (0, 1] and r in (0, 4]. When r = 4, the algorithm maintains the 
    O(1/k^2) convergence rate for the objective function, and allows proving the (weak) convergence 
    of {x_k} with rate o(1/k) for certain values of p and q. [1, Theorem 3.5]
    
    When r < 4, then the algorithm is a variant of the Forward-Backward splitting.

    By choosing p in [1/80, 1/10], the algorithm can be accelerated by slowing down the speed of
    a_k approaching 1. This is called a "lazy start" strategy [1, Proposition 4.1].
    
    Parameters
    ----------
    L : float
        Lipschitz constant for the gradient of F.
    x0 : np.array
        Starting approximation.
    p, q, r : float
        Parameters for the update rule t_k. p and q are required to be in (0, 1] and r in (0, 4].
    proxR : function(gamma, y)
        A function object computing the proximal operator for the proper convex lsc function R, or 
        an approximation thereof. The function is assumed to take two parameters: the step size 
        gamma, and an input value y.
    gradF : function(y)
        A function object computing the gradient of F, or an approximation thereof. It is assumed
        to be Lipschitz continuous, and takes an input value y.
    max_iter : int, optional
        Maximum number of iterations before the algorithm terminates. Defaults to 500.

    Returns
    -------
    xk, xk_prev : np.array
        Approximate solution to the convex optimization problem min F(x) + R(x).

    """
    assert p > 0 and p <= 1, 'p must belong to (0, 1]'
    assert q > 0 and q <= 1, 'q must belong to (0, 1]'
    assert r > 0 and r <= 4, 'r must belong to (0, 4]'
    
    xk_prev = np.copy(x0) # x_{-1}
    xk = np.copy(x0) # x_{0}
    tk = 1
    gamma = 1./L

    for k in range(1, max_iter+1):
        tk_prev = tk
        tk = (p + np.sqrt(q + r*tk_prev**2)) / 2
        ak = (tk_prev - 1) / tk
        yk = xk + ak*(xk - xk_prev)
        xk_prev = np.copy(xk)
        xk = proxR(gamma, yk - gamma*gradF(yk))

        yield xk, xk_prev


def fista_cd(L, x0, d, proxR, gradF, max_iter=500):
    """
    A version of FISTA by Chambolle and Dossal which maintains the O(1/k^2) objective convergence 
    rate, and allows to prove (weak) convergence rate for iterates {x_k}. The following rule is
    is used to update t_k:
        t_k = (k+d)/d,  a_k = (t_{k-1} - 1)/t_k = (k-1)/(k+d)
    
    The practical performance is almost identical to FISTA if the parameter d is chosen close to 2.
    By choosing d between 10 and 80, the algorithm can be accelerated by slowing down the speed of 
    a_k reaching 1. This is called a "lazy start" strategy [1, Proposition 4.1].

    Parameters
    ----------
    L : float
        Lipschitz constant for the gradient of F.
    x0 : np.array
        Starting approximation.
    d : float
        Additional parameter for the update rule tk = (k+d) / d.
    proxR : function(gamma, y)
        A function object computing (an approximation to) the proximal operator for the proper 
        convex lsc function R. The function is assumed to take two parameters: the step size gamma,
        and an input value y.
    gradF : function(y)
        A function object computing the gradient of F, or an approximation thereof. It is assumed
        to be Lipschitz continuous, and takes an input value y.
    max_iter : int, optional
        Maximum number of iterations before the algorithm terminates. Defaults to 500.

    Returns
    -------
    xk, xk_prev : np.array
        Approximate solution to the convex optimization problem min F(x) + R(x).

    """
    xk_prev = np.copy(x0) # x_{-1}
    xk = np.copy(x0) # x_{0}
    gamma = 1./L

    for k in range(1, max_iter+1):
        ak = (k - 1) / (k + d)
        yk = xk + ak*(xk - xk_prev)
        xk_prev = np.copy(xk)        
        xk = proxR(gamma, yk - gamma*gradF(yk))
        
        yield xk, xk_prev

