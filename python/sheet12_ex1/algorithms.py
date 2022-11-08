import numpy as np
import cvxpy as cp
"""
Collection of algorithms for Compressed Sensing, Sheet 12, Exercise 1.

@author: Ferdinand Vanmaele
"""

def basis_pursuit(A, b, nn=False, verbose=False):
    """
    Solve the l1 minimization problem:  (P1)
        min ||x||_1  s.t.  Ax = b
    
    Iff A has the null-space property of order s \in [n], then basis pursuit recovers the 
    sparsest solution to b = Ax.

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n.
    b : np.array
        Measurement vector of dimension m.
    nn : bool, optional
        Add the constraint u >= 0. Defaults to False.
    verbose : bool, optional
        Output solver information. Defaults to False.

    Returns
    -------
    np.array
        Solution to the minimization problem (P1), if feasible and bounded.

    """
    n = np.shape(A)[1]
    u = cp.Variable(n)

    cost = cp.norm(u, 1)
    if nn is True:
        cstr = [A@u == b, u >= 0]
    else:
        cstr = [A@u == b]
    
    prob = cp.Problem(cp.Minimize(cost), cstr)
    prob.solve(verbose=verbose)
    
    return u.value
        

# TODO: return number of iterations for each (greedy) algorithm, norm of residual

def OMP(A, b, max_iter=500, rcond=1e-15, tol_res=None):
    """
    Orthogonal matching pursuit is a greedy method which starts from an empty support set, and
    adds an index at every step. It does so by finding the maximum correlation between columns of A
    and the residual r(k) = b - Ax(k), and then orthogonally projecting on the new support set.
    Latter is done by taking the pseudo-inverse of A on the support set S(k) and multiplying by b.
    
    The downside of OMP is once an incorrect index has been selected in a support set S(k), it will
    remain in all subsequent support sets. Hence, if an incorrect index has been selected,
    s iterations of OMP are not enough to recover a vector with sparsity s that solves Ax = b.

    Every nonzero s-sparse vector with S = supp(x) of size s is recovered from b = Ax after 
    at most s iterations, if and only if the exact recovery condition (ERC) is fulfilled:
        ||pinv(A)[:, S] @ A[:, Sc]||_1 < 1

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n. The algorithm operates under the assumption
        that the columns of A are normalized.
    b : np.array
        Measurement vector of dimension m.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    tol_res : float, optional
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.

    Returns
    -------
    np.array
        Sparse representation x(k), minimizing ||b - Ax(k)||_2 on a support set. 

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"
    # expensive check, disable with `python -O`
    assert np.all(np.isclose(np.linalg.norm(A, axis=0), np.ones(n))), "columns of A are not normalized"
    
    x = np.zeros(n)
    S = np.array([])
    r = np.copy(b)

    for k in range(0, max_iter):
        j = np.argmax(A.T @ r)                  # maximum correlation
        np.append(S, j)                         # update index set
        Sc = np.setdiff1d(np.arange(0, m), S)   # complement of index set

        # orthogonal projection
        x[S]  = np.linalg.pinv(A[:, S], rcond=rcond) @ b
        x[Sc] = 0.0
        
        # update residual
        r = b - A@x
        
        # termination criterium
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
        
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged


def MP(A, b, max_iter=500, tol_res=None):
    """
    A greedy strategy that does not involve any orthogonal projection. Unlike OMP, only the
    component associated with the currently selected column is updated:
        x(k+1) = x(k) + t*e_j
    
    where t is a real number and the indices j are chosen to minimize the 
    residual ||b - Ax(n+1)||_2. 
    
    Matching pursuit cannot exclude that chosen columns are not chosen again in future iterations.

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n. The algorithm operates under the assumption that
        A has normalized columns.
    b : np.array
        Measurement vector of dimension m.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    tol_res : float, optional
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.

    Returns
    -------
    np.array
        Sparse representation x(k).

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"
    # expensive check, disable with `python -O`
    assert np.all(np.isclose(np.linalg.norm(A, axis=0), np.ones(n))), "columns of A are not normalized"
    
    x = np.zeros(n)
    r = np.copy(b)
    
    for k in range(0, max_iter):
        tmp = A.T @ r
        j = np.argmax(tmp)      # maximum correlation
        t = tmp[j]
        x[j] = x[j] + t

        # update residual
        r = b - A@x
        
        # termination criterium
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged


def threshold(x, s, rule, seed=None):
    """
    Hard thresholding operator.

    Parameters
    ----------
    x : np.array
        Input vector of dimension n >= s.
    s : int
        Number of entries with largest magnitudes that should be kept.
    rule : str
        Rule to resolve ties when elements have the same magnitudes. Can be one of 'lex' for
        lexicographic order, 'lex_reverse' for reverse lexicographic order, or 'random' for
        a random selection.
    seed : int, optional
        The seed used with rule='random'. Defaults to None.

    Returns
    -------
    np.array
        s entries of x with the largest magnitudes set to zero.

    """
    # precondition checks
    assert s > 0, "s must be a positive value"
    assert x.size <= s, "s must not exceed the length of x"
    Tx = np.zeros(x.size)

    if rule == 'lex':
        idx = np.flip(np.argsort(x))[:s]
    
    elif rule == 'lex_reverse':
        idx = np.argsort(np.flip(x))[:s]
    
    elif rule == 'random':
        np.random.seed(seed)
        idx = np.flip(np.lexsort((np.random.random(x.size), x)).argsort())
    else:
        raise ValueError
    
    Tx[idx] = x[idx]
    return Tx


def BT(A, b, s, rule='lex', rcond=1e-15):
    """
    The basic thresholding algorithm consists in determinign the support of the s-sparse vector x
    to be recovered from the measurement vector b = Ax, as the indices of s largest absolute
    entries of A' b, and then in finding the vector with this support that best fits the measurement.
    
    The intuition relies on the approximate inversion of the action on sparse vectors of the
    measurement matrix A by the action of its adjoint A'.
    
    An s-sparse vector with S := supp(x) is recovered from b = Ax via basic thresholding, if and
    only if
        min(j \in S) |(A' b)_j| > max(l \in Sc) |(A' b)_l|

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n.
    b : np.array
        Measurement vector of dimension m.
    s : int
        Sparsity level of the solution x.
    rule : str, optional
        Rule used for solving ties with the thresholding operator. Defaults to 'lex'.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.

    Returns
    -------
    np.array
        s-sparse vector approximating a solution of the linear system Ax = b.

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"

    S = np.nonzero(threshold(A.T @ b, s, rule))  # support set
    x = np.zeros(m)

    # orthogonal projection on S
    x[S]  = np.linalg.pinv(A[:, S], rcond=rcond) @ b

    return x


def IHT(A, b, s, x0=None, rule='lex', max_iter=500, tol_res=None):
    """
    Iterative hard thresholding is an iterative algorithm to solve the rectangular system
        A' Az = A'b
        
    knowing that the solution is s-sparse. This equation can be interpreted as the fixed point equation
        z = (I - A' A)x + A' b
             
    Only the s largest absolute emax_iter=500,ntries are kept at each iteration.

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n.
    b : np.array
        Measurement vector of dimension m.
    s : int
        Sparsity level of the solution x.
    x0 : np.array, optional
        Starting vector of dimension n, typically 0 or s-sparse.
    rule : str, optional
        Rule used for solving ties with the thresholding operator. Defaults to 'lex'.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.

    Returns
    -------
    np.array
        s-sparse vector approximating a solution of the linear system A' Az = A'b.

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0)
    r = b - A @ x0

    for k in range(0, max_iter):
        x = threshold(x + A.T@r, s, rule)
        r = b - A@x
        
        # termination criterion
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged

    
def HTP(A, b, s, x0=None, rule='lex', max_iter=500, rcond=1e-15, tol_res=None):
    """
    Iterative hard thresholding is a variation of IHT which computes an orthogonal projection 
    in each step.
    
    x(k+1) is defined as the vector with the same support as threshold(x(k) + A'r(k), s) that best 
    fits the measurements.

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n.
    b : np.array
        Measurement vector of dimension m.
    s : int
        Sparsity level of the solution x.
    x0 : np.array
        Starting vector of dimension n, typically 0 or s-sparse.
    rule : str, optional
        Rule used for solving ties with the thresholding operator. Defaults to 'lex'.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.
    tol_res : float, optional
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.

    Returns
    -------
    np.array
        s-sparse vector approximating a solution of the linear system A' Az = A' b.

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0)
    r = b - A @ x0

    for k in range(0, max_iter):
        S  = np.nonzero(threshold(x + A.T@r, s, rule))  # support set
        Sc = np.setdiff1d(np.arange(0, m), S)

        # orthogonal projection
        x[S]  = np.linalg.pinv(A[:, S], rcond=rcond) @ b
        x[Sc] = 0.0

        # compute residual for next step
        r = b - A@x
        
        # termination criterion
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged


def CoSaMP(A, b, s, x0=None, rule='lex', max_iter=500, rcond=1e-15, tol_res=None):
    """
    Compressive sampling matching pursuit (CoSaMP) keeps track of the active support set S(k+1),
    and adds as well as removes elements in each iteration. An estimate of the solution sparsity
    required beforehand. 
    
    At each iteration, an s-sparse approximation is used to compute the current  error. The 2s 
    columns of A that correlate best with this error are then selected and added to the support set.
    
    A least squares estimate is found over the current support. The s largest elements in magnitude
    are found, and their corresponding locations are chosen as the new support set.

    Parameters
    ----------
    A : np.array
        Coefficient matrix of dimension m x n. The algorithm operates under the assumption that
        A has normalized columns.
    b : np.array
        Right-hand side of dimension m.
    s : int
        Sparsity level of the solution x.
    x0 : np.array
        Starting vector of dimension n, typically 0 or s-sparse.
    rule : str, optional
        Rule used for solving ties with the thresholding operator. Defaults to 'lex'.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.
    tol_res : float, optional
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None
    
    Returns
    -------
    np.array

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"
    # expensive check, disable with `python -O`
    assert np.all(np.isclose(np.linalg.norm(A, axis=0), np.ones(n))), "columns of A are not normalized"
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0)
    r = b - A@x

    for k in range(0, max_iter):
        # preliminary support set
        S = np.hstack((np.nonzero(x), np.nonzero(threshold(A.T@r, 2*s, rule))))

        # orthogonal projection
        u = np.zeros(n)
        u[S] = np.linalg.pinv(A[:, S], rcond=rcond) @ b
        
        # thresholding
        x = threshold(u, s, rule)
        r = b - A@x
        
        # termination criterion
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged


def SP(A, b, s, x0=None, rule='lex', max_iter=500, rcond=1e-15, tol_res=None):
    """
    CoSaMP and subspace pursuit differ in the following:
    
    1. Subspace pursuit only adds the s columns that correlate best with the error. CoSaMP
       adds 2s columns in each iteration.
    2. CoSaMP uses the output of the thresholding step. Subspace pursuit optimizes the error
       over the new support.

    Parameters
    ----------
    A : np.array
        Measurement matrix of dimension m x n. The algorithm operates under the assumption that
        A has normalized columns.
    b : np.array
        Measurement vector of dimension m.
    s : int
        Sparsity level of the solution x.
    x0 : np.array, optional
        Starting vector of dimension n, typically 0 or s-sparse.
    rule : str, optional
        Rule used for solving ties with the thresholding operator. Defaults to 'lex'.
    max_iter : int, optional
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    rcond : float, optional
        Cutoff for small singular values when computing pseudoinverses. Defaults to 1e-15.
    tol_res : float, optional
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.

    Returns
    -------
    np.array
        s-sparse vector approximating a solution of the linear system Ax = b.
    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"
    # expensive check, disable with `python -O`
    assert np.all(np.isclose(np.linalg.norm(A, axis=0), np.ones(n))), "columns of A are not normalized"
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.copy(x0)
    r = b - A@x

    for k in range(0, max_iter):
        # preliminary support set
        U = np.hstack((np.nonzero(x), np.nonzero(threshold(A.T@r, s, rule))))

        # orthogonal projection
        u = np.zeros(n)
        u[U] = np.linalg.pinv(A[:, U], rcond=rcond) @ b
        
        # thresholding and second projection step
        S = np.nonzero(threshold(u, s))
        x = np.zeros(n)

        x[S] = np.linalg.pinv(A[:, S], rcond=rcond) @ b
        r = b - A@x
        
        # termination criterion
        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    if tol_res is None:
        converged = None
    elif k == max_iter-1:
        converged = False
    else:
        converged = True

    return x, converged