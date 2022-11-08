#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of performance of different greedymethods with l1-minimization (Sheet 12, Exercise 1)

@author: Ferdinand Vanmaele
"""

import numpy as np
import algorithms
from sensors import sensor_random, sensor_partial_fourier

import json
import sys
from time import process_time


# %% Helper functions
def problem_range(problem):
    """
    Convert a string such as '1, 3-5' to a set of problems we want to solve.

    Parameters
    ----------
    problem : str

    Returns
    -------
    to_solve : dict

    """
    problems = ['basis_pursuit', 'omp', 'mp', 'iht', 'cosamp', 'bt', 'htp', 'sp']
    to_solve = {}
    
    if problem != 'all':
        ranges = problem.split(',')

        for prange in ranges:
            p = prange.split('-')            

            if len(p) == 1:
                idx = int(p[0])-1 # 1-indexing
                to_solve[problems[idx]] = True
            elif len(p) == 2:
                for pi in range(int(p[0]), int(p[1])+1):
                    idx = pi-1
                    to_solve[problems[idx]] = True
            else:
                print("error: invalid --problem range")
                sys.exit(1)
    else:
        for p in problems:
            to_solve[p] = True

    return to_solve, problems


def recovery_error(x, xh, ord=None):
    """
    Compute the forward relative error ||x - xh|| / ||x||. This assumes the true solution x
    is known beforehand.

    Parameters
    ----------
    x : np.array
        Known solution x.
    xh : np.array
        Approximate solution xh.
    ord : int, optional
        The type of norm as specified in np.linalg.norm. Defaults to None (2-norm for vectors).

    Returns
    -------
    float
        Forward relative error.

    """
    return np.linalg.norm(x - xh, ord=ord) / np.linalg.norm(x)


def exact_recovery_condition(A, S):
    m, n = A.shape
    Sc = np.setdiff1d(range(0, n), S)
    
    return np.linalg.norm(np.linalg.pinv(A[:, S]) @ A[:, Sc], ord=1)  # <1 => TRUE


def coherence(A):
    m, n = np.shape(A)
    mu_max = 0

    for i in range(0, n):
        for j in range(0, i+1):
            mu = np.abs(A[:, i] @ A[:, j]) / np.linalg.norm(A[:, i]) / np.linalg.norm(A[:, j])
            if mu > mu_max:
                mu_max = mu
    return mu


def generate_problems(n, s_min, s_max, n_reps, seed=None):
    np.random.seed(seed)
    # https://stackoverflow.com/a/8713681
    xs = [[] for _ in range(s_min, s_max+1)]
    
    for i, s in enumerate(range(s_min, s_max+1)):
        for k in range(0, n_reps):
            x = np.zeros(n)
            x_nnz = np.random.randn(s)          # nonzero values from standard Gaussian distribution
            S = np.random.permutation(n)[:s]    # nonzero locations uniformly at random
            x[S] = x_nnz

            assert np.nonzero(x)[0].size == s   # sanity check
            xs[i].append((np.copy(x), np.copy(S)))

    return xs


def run_trials(trials, algorithm, label):
    avg = []
    avg_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre = []
        ts_elapsed = []
    
        for x, S in Trial:
            b  = A @ x
            ts_start = process_time()
            xh = algorithm(A, b, s)   # insert user-defined algorithm
            
            ts_stop  = process_time()
            x_fre.append(recovery_error(x, xh))
            
            ts_elapsed.append(ts_stop - ts_start)
    
        avg.append(np.mean(x_fre))
        avg_ts.append(np.mean(ts_elapsed))

        print("n: {}, sparsity: {}, {}, avg. error: {}".format(n, s, label, avg[-1]))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_ts[-1]))

    return avg, avg_ts


# %%
m = 2**6
n = 2**7
A = sensor_random(m, n)
F = sensor_partial_fourier(m, n)
Ac = coherence(A)
Fc = coherence(F)

# %%
if __name__ == "__main__":
    # retrieve arguments from the command-line
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')

    # mandatory arguments
    parser.add_argument("MType", type=str, help="Normalized column matrix, Gaussian [A] or partial Fourier [F]")

    # optional arguments
    parser.add_argument("--seed", type=int, default=100, help="value for np.random.seed()")
    parser.add_argument("--n", type=int, default=2**7, help="problem dimension (columns)")
    parser.add_argument("--m", type=int, default=2**6, help="problem dimension (rows)")
    parser.add_argument("--tol", type=float, default=1e-6, help="value for residual-based stopping criterion")
    parser.add_argument("--n-trials", type=int, default=100, help="repetitions for each sparsity value")
    parser.add_argument("--s-max", type=int, default=None, help="maximum sparsity level (defaults to m)")
    parser.add_argument("--s-min", type=int, default=None, help="minimum sparsity level (defaults to 1)")
    parser.add_argument("--problem", type=str, default='all', help="list of algorithms to test (starts from 1, comma separation)")
    args = parser.parse_args()

    # %% assign command-line arguments
    np.random.seed(args.seed)
    n, m, tol = args.n, args.m, args.tol
    n_trials  = args.n_trials
    mtx_type  = args.MType

    # define default values
    if mtx_type == 'A':
        A = sensor_random(m, n)
    elif mtx_type == 'F':
        A = sensor_partial_fourier(m, n)
    else:
        print("error: invalid matrix type", file=sys.stderr)
        sys.exit(1)

    s_max = m if args.s_max is None else args.s_max
    s_min = 1 if args.s_min is None else args.s_min
    to_solve, _ = problem_range(args.problem)

    # Recover xh from measurements b = Ax, for every problem instance of sparsity 1 <= s <= m
    trials = generate_problems(n, s_min, s_max, n_trials, seed=None)

    # %% 01
    if 'basis_pursuit' in to_solve:
        algorithm = lambda A, b, _: algorithms.basis_pursuit(A, b)
        avg_bp, avg_bp_ts = run_trials(trials, algorithm, 'BP')
        
        with open('{}_m{}_n{}_{}_trials_fre_BP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_bp, 'cputime': avg_bp_ts}, outfile)
     
    # %% 02
    if 'omp' in to_solve:
        algorithm = lambda A, b, _: algorithms.OMP(A, b, tol_res=tol)[0]
        avg_omp, avg_omp_ts = run_trials(trials, algorithm, 'OMP')
        
        with open('{}_m{}_n{}_{}_trials_fre_OMP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_omp, 'cputime': avg_omp_ts}, outfile)
    
    # %% 03
    if 'mp' in to_solve:
        algorithm = lambda A, b, _: algorithms.MP(A, b, tol_res=tol)[0]
        avg_mp, avg_mp_ts = run_trials(trials, algorithm, 'MP')
        
        with open('{}_m{}_n{}_{}_trials_fre_MP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_mp, 'cputime': avg_mp_ts}, outfile)
    
    # %% 04
    if 'iht' in to_solve:
        algorithm = lambda A, b, s: algorithms.IHT(A, b, s, tol_res=tol, adaptive=True)[0]
        avg_iht, avg_iht_ts = run_trials(trials, algorithm, 'IHT')
        
        with open('{}_m{}_n{}_{}_trials_fre_IHT.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_iht, 'cputime': avg_iht_ts}, outfile)
    
    # %% 05
    if 'cosamp' in to_solve:
        algorithm = lambda A, b, s: algorithms.CoSaMP(A, b, s, tol_res=tol)[0]
        avg_cosamp, avg_cosamp_ts = run_trials(trials, algorithm, 'CoSaMP')
        
        with open('{}_m{}_n{}_{}_trials_fre_CoSaMP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_cosamp, 'cputime': avg_cosamp_ts}, outfile)
    
    # %% 06
    if 'bt' in to_solve:
        algorithm = lambda A, b, s: algorithms.BT(A, b, s)
        avg_bt, avg_bt_ts = run_trials(trials, algorithm, 'BT')
        
        with open('{}_m{}_n{}_{}_trials_fre_BT.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_bt, 'cputime': avg_bt_ts}, outfile)
    
    # %% 07
    if 'htp' in to_solve:
        algorithm = lambda A, b, s: algorithms.HTP(A, b, s, tol_res=tol)[0]
        avg_htp, avg_htp_ts = run_trials(trials, algorithm, 'HTP')

        with open('{}_m{}_n{}_{}_trials_fre_HTP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_htp, 'cputime': avg_htp_ts}, outfile)
    
    # %% 08
    if 'sp' in to_solve:
        algorithm = lambda A, b, s: algorithms.SP(A, b, s, tol_res=tol)[0]
        avg_sp, avg_sp_ts = run_trials(trials, algorithm, 'SP')
        
        with open('{}_m{}_n{}_{}_trials_fre_SP.json'.format(mtx_type, m, n, n_trials), 'w') as outfile:
            json.dump({'error': avg_sp, 'cputime': avg_sp_ts}, outfile)