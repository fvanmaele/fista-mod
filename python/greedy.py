import numpy as np
import cvxpy as cp


def basis_pursuit(A, b, nn=False):
    """
    Solve the l1 minimization problem:  (P1)
        min ||x||_1  s.t.  Ax = b
    
    by recasting it as a linear program in the standard form:
        min(u, v) 1'u + 1'v  s.t.  Au - Av = b,  u, v >= 0
        
    where u equals the positive part of x and v the negative part. Iff A has the null-space property
    of order s \in [n], then basis pursuit recovers the sparsest solution to b = Ax.

    Parameters
    ----------
    A : np.array
        Coefficient matrix of dimension m x n.
    b : np.array
        Right-hand side of dimension m.
    nn : bool
        Add the constraint u >= 0 when True (non-negative least squares).

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
    prob.solve()
    
    return u.value
        

def OMP(A, b, max_iter=500, rcond=1e-15, tol_res=None, s=None):
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
        Coefficient matrix of dimension m x n. The algorithm operates under the assumption
        that the columns of A are normalized.
    b : np.array
        Right-hand side of dimension m.
    max_iter : int
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    tol_res : float
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.
    s : int
        The known sparsity level of the solution vector. Defaults to None.
    rcond : float
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
        
        # termination criteria
        if s is not None and len(S) == s:
            break

        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
            
    return x


def MP(A, b, max_iter=500, tol_res=None, s=None):
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
        Coefficient matrix of dimension m x n. The algorithm operates under the assumption that
        A has normalized columns.
    b : np.array
        Right-hand side of dimension m.
    max_iter : int
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    tol_res : float
        Terminate when ||b - Ax(k)||_2 is less than the given tolerance. Defaults to None.
    s : int
        The known sparsity level of the solution vector. Defaults to None.

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
        
        # termination criteria
        if s is not None and np.count_nonzero(np.isclose(x, np.zeros(n))) == s:
            break

        if tol_res is not None and np.linalg.norm(r) < tol_res:
            break
    
    return x
        

def threshold(x, s, rule='lex', seed=None):
    """
    Hard thresholding operator.

    Parameters
    ----------
    x : np.array
        Input vector of dimension n >= s.
    s : int
        Number of entries with largest magnitudes that should be kept.
    rule : str
        Rule to resolve ties when elements have the same magnitudes. Can be one 'lex', 'lex_reverse'
        or 'random'. Defaults to 'lex'.

    Returns
    -------
    np.array
        s entries of x with the largest magnitudes set to zero.

    """
    # precondition checks
    assert s > 0, "s must be a positive value"
    assert x.size <= s, "s must not exceed the length of x"
    Tx = np.zeros(x.size)

    # FIXME: actually set the values of the complement (a copy of x) to zero
    if rule == 'lex':
        return x[np.flip(np.argsort(x))[:s]]
    if rule == 'lex_reverse':
        return x[np.argsort(np.flip(x))[:s]]
    if rule == 'random':
        np.random.seed(seed)
        return x[np.flip(np.lexsort((np.random.random(x.size), x)).argsort())]


def CoSaMP(A, b, s, max_iter=500, x0=None):
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
    max_iter : int
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    x0 : np.array
        Starting vector of dimension n, typically 0 or s-sparse.
    
    Returns
    -------
    np.array

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"
    # expensive check, disable with `python -O`
    assert np.all(np.isclose(np.linalg.norm(A, axis=0), np.ones(n))), "columns of A are not normalized"
    
    # TODO


def IHT(A, b, s, max_iter=500, x0=None):
    """
    Iterative hard thresholding. An estimate of the solution sparsity is required beforehand.

    Parameters
    ----------
    A : np.array
        Coefficient matrix of dimension m x n.
    b : np.array
        Right-hand side of dimension m.
    s : int
        Sparsity level of the solution x.
    max_iter : int
        Maximum number of iterations before terminating the algorithm. Defaults to 500.
    x0 : np.array
        Starting vector of dimension n, typically 0 or s-sparse.

    Returns
    -------
    np.array

    """
    m, n = np.shape(A)
    # precondition checks
    assert m == len(b), "A and b have incompatible dimensions"

    if x0 is None:
        x0 = np.zeros(n)
    r = b - A @ x0
    
    for k in range(0, max_iter):
        pass