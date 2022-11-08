#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of performance of different greedymethods with l1-minimization (Sheet 12, Exercise 1)

@author: Ferdinand Vanmaele
"""

import numpy as np
import algorithms
from sensors import sensor_random, sensor_random_partial_fourier

import json
import sys
from time import process_time


# %% Helper functions
def problem_range(include):
    """
    Convert a string such as '1, 3-5' to a set of problems we want to solve.

    Parameters
    ----------
    include : str

    Returns
    -------
    to_solve : dict

    """
    problems = ['basis_pursuit', 'omp', 'mp', 'iht', 'cosamp', 'bt', 'htp', 'sp']
    to_solve = {}
    
    if include != 'all':
        ranges = include.split(',')

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
                print("error: invalid --include range")
                sys.exit(1)
    else:
        for p in problems:
            to_solve[p] = True

    return to_solve


def recovery_error(x, xh, ord=None):
    """
    Compute the forward relative error ||x - xh|| / ||x||. This assumes the true solution x
    is known a-priori.

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


# %% 1. basis pursuit (l1-minimization)
def main_basis_pursuit(trials, *args):
    avg_bp = []
    avg_bp_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_bp = []
        ts_elapsed = []
    
        for x, S in Trial:
            b  = A @ x
            ts_start = process_time()
            xh = algorithms.basis_pursuit(A, b)
            
            ts_stop  = process_time()
            x_fre_bp.append(recovery_error(x, xh))
            
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_bp.append(np.mean(x_fre_bp))
        avg_bp_ts.append(np.mean(ts_elapsed))
    
        print("n: {}, sparsity: {}, basis pursuit, avg. error: {}".format(n, s, avg_bp[-1]))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_bp_ts[-1]))

    return avg_bp, avg_bp_ts

    
# %% 2. orthogonal matching pursuit
def main_omp(trials, tol, *args):
    avg_omp = []
    avg_omp_ts = []
    avg_erc = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_omp = []
        iters_omp = []
        erc = []
        ts_elapsed = []
    
        for x, S in Trial:
            b = A @ x
            erc.append(exact_recovery_condition(A, S))  # XXX: seperate this out
            ts_start = process_time()
            xh, conv, iters = algorithms.OMP(A, b, tol_res=tol)
            
            ts_stop  = process_time()
            x_fre_omp.append(recovery_error(x, xh))
            iters_omp.append(iters)
            
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_omp.append(np.mean(x_fre_omp))
        avg_omp_ts.append(np.mean(ts_elapsed))
        avg_erc.append(np.mean(erc))
        
        print("n: {}, sparsity: {}, OMP, avg. error: {}, avg. ERC: {}, avg. iterations: {}".format(
            n, s, avg_omp[-1], avg_erc[-1], np.round(np.mean(iters_omp))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_omp_ts[-1]))

    return avg_omp, avg_omp_ts, avg_erc


# %% 3. matching pursuit
def main_mp(trials, tol, *args):
    avg_mp = []
    avg_mp_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_mp = []
        iters_mp = []
        ts_elapsed = []
    
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh, conv, iters = algorithms.MP(A, b, tol_res=tol)
    
            ts_stop  = process_time()
            x_fre_mp.append(recovery_error(x, xh))
            iters_mp.append(iters)
    
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_mp.append(np.mean(x_fre_mp))
        avg_mp_ts.append(np.mean(ts_elapsed))
        
        print("n: {}, sparsity: {}, MP, avg. error: {}, avg. iterations: {}".format(
            n, s, avg_mp[-1], np.round(np.mean(iters_mp))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_mp_ts[-1]))

    return avg_mp, avg_mp_ts


# %% 4. iterative hard thresholding
def main_iht(trials, tol, *args):
    avg_iht = []
    avg_iht_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_iht = []
        iters_iht = []
        ts_elapsed = []
    
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh, conv, iters = algorithms.IHT(A, b, s, tol_res=tol, adaptive=True)
            
            ts_stop  = process_time()
            x_fre_iht.append(recovery_error(x, xh))
            iters_iht.append(iters)
            
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_iht.append(np.mean(x_fre_iht))
        avg_iht_ts.append(np.mean(ts_elapsed))
    
        print("n: {}, sparsity: {}, IHT, avg. error: {}, avg. iterations: {}".format(
            n, s, avg_iht[-1], np.round(np.mean(iters_iht))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_iht_ts[-1]))

    return avg_iht, avg_iht_ts


# %% 5. compressive sampling matching pursuit
def main_cosamp(trials, tol, *args):
    avg_cosamp = []
    avg_cosamp_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_cosamp = []
        iters_cosamp = []
        ts_elapsed = []
        
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh, conv, iters = algorithms.CoSaMP(A, b, s, tol_res=tol)
    
            ts_stop  = process_time()
            x_fre_cosamp.append(recovery_error(x, xh))
            iters_cosamp.append(iters)
    
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_cosamp.append(np.mean(x_fre_cosamp))
        avg_cosamp_ts.append(np.mean(ts_elapsed))
        
        print("n: {}, sparsity: {}, CoSaMP, avg. error: {}, avg. iterations: {}".format(
            n, s, avg_cosamp[-1], np.round(np.mean(iters_cosamp))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_cosamp_ts[-1]))

    return avg_cosamp, avg_cosamp_ts


# %% 6. basic thresholding
def main_bt(trials, *args):
    avg_bt = []
    avg_bt_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_bt = []
        ts_elapsed = []
    
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh = algorithms.BT(A, b, s)
    
            ts_stop  = process_time()
            x_fre_bt.append(recovery_error(x, xh))
    
            ts_elapsed.append(ts_stop - ts_start)
            
        avg_bt.append(np.mean(x_fre_bt))
        avg_bt_ts.append(np.mean(ts_elapsed))
        
        print("n: {}, sparsity: {}, BT, avg. error: {}".format(n, s, avg_bt[-1]))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_bt_ts[-1]))

    return avg_bt, avg_bt_ts


# %% 7. hard thresholding pursuit
def main_htp(trials, tol, *args):
    avg_htp = []
    avg_htp_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_htp = []
        iters_htp = []
        ts_elapsed = []
    
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh, conv, iters = algorithms.HTP(A, b, s, tol_res=tol)
    
            ts_stop  = process_time()
            x_fre_htp.append(recovery_error(x, xh))
            iters_htp.append(iters)
            
            ts_elapsed.append(ts_stop - ts_start)
            
        avg_htp.append(np.mean(x_fre_htp))
        avg_htp_ts.append(np.mean(ts_elapsed))
        
        print("n: {}, sparsity: {}, HTP, avg. error: {}, avg. iterations: {}".format(
            n, s, avg_htp[-1], np.round(np.mean(iters_htp))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_htp_ts[-1]))

    return avg_htp, avg_htp_ts


# %% 8. subspace pursuit
def main_sp(trials, tol, *args):
    avg_sp = []
    avg_sp_ts = []
    
    for s, Trial in enumerate(trials, start=1):
        x_fre_sp = []
        iters_sp = []
        ts_elapsed = []
        
        for x, S in Trial:
            b = A @ x
            ts_start = process_time()
            xh, conv, iters = algorithms.SP(A, b, s, tol_res=tol)
    
            ts_stop = process_time()
            x_fre_sp.append(recovery_error(x, xh))
            iters_sp.append(iters)
            
            ts_elapsed.append(ts_stop - ts_start)
    
        avg_sp.append(np.mean(x_fre_sp))
        avg_sp_ts.append(np.mean(ts_elapsed))
        
        print("n: {}, sparsity: {}, SP, avg. error: {}, avg. iterations: {}".format(
            n, s, avg_sp[-1], np.round(np.mean(iters_sp))))
        print("avg. CPU time elapsed: {:>2.6f}s".format(avg_sp_ts[-1]))

    return avg_sp, avg_sp_ts


# %%
if __name__ == "__main__":
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

    # Assign command-line arguments
    np.random.seed(args.seed)
    n = args.n
    m = args.m
    tol = args.tol
    n_repetitions = args.n_trials
    mtx_type = args.MType

    # Define default values
    if mtx_type == 'A':
        A = sensor_random(m, n)
    elif mtx_type == 'F':
        A = sensor_random_partial_fourier(m, n)
    else:
        print("error: invalid matrix type", file=sys.stderr)
        sys.exit(1)

    s_max = m if args.s_max is None else args.s_max
    s_min = 1 if args.s_min is None else args.s_min
    to_solve = problem_range(args.problem)

    # %% Recover xh from measurements b = Ax, for every problem instance of sparsity 1 <= s <= m
    trials = generate_problems(n, s_min, s_max, n_repetitions, seed=None)

    # Solve problems
    # for label in to_solve.keys():
    #     func = eval('main_' . label)
    #     avg, avg_ts = func(trials, tol)
        
    #     with open('{}_m{}_n{}_{}_trials_fre_{}.json'.format(mtx_type, m, n, n_repetitions, label), 'w') as outfile:
    #         json.dump({'error': avg, 'cputime': avg_bp}, outfile)
     
    if 'basis_pursuit' in to_solve:
        avg_bp, avg_bp_ts = main_basis_pursuit(trials)
        
        with open('{}_m{}_n{}_{}_trials_fre_BP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_bp, 'cputime': avg_bp_ts}, outfile)
     
    if 'omp' in to_solve:
        avg_omp, avg_omp_ts = main_omp(trials, tol)
        
        with open('{}_m{}_n{}_{}_trials_fre_OMP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_omp, 'cputime': avg_omp_ts}, outfile)
    
    if 'mp' in to_solve:
        avg_mp, avg_mp_ts = main_mp(trials, tol)
        
        with open('{}_m{}_n{}_{}_trials_fre_MP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_mp, 'cputime': avg_mp_ts}, outfile)
    
    if 'iht' in to_solve:
        avg_iht, avg_iht_ts = main_iht(trials, tol)
        
        with open('{}_m{}_n{}_{}_trials_fre_IHT.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_iht, 'cputime': avg_iht_ts}, outfile)
    
    if 'cosamp' in to_solve:
        avg_cosamp, avg_cosamp_ts = main_cosamp(trials, tol)
        
        with open('{}_m{}_n{}_{}_trials_fre_CoSaMP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_cosamp, 'cputime': avg_cosamp_ts}, outfile)
    
    if 'bt' in to_solve:
        avg_bt, avg_bt_ts = main_bt(trials)
        
        with open('{}_m{}_n{}_{}_trials_fre_BT.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_bt, 'cputime': avg_bt_ts}, outfile)
    
    if 'htp' in to_solve:
        avg_htp, avg_htp_ts = main_htp(trials, tol)

        with open('{}_m{}_n{}_{}_trials_fre_HTP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_htp, 'cputime': avg_htp_ts}, outfile)
    
    if 'sp' in to_solve:
        avg_sp, avg_sp_ts = main_sp(trials, tol)
        
        with open('{}_m{}_n{}_{}_trials_fre_SP.json'.format(mtx_type, m, n, n_repetitions), 'w') as outfile:
            json.dump({'error': avg_sp, 'cputime': avg_sp_ts}, outfile)
