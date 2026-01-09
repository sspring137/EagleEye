#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:38:28 2025

@author: sspringe
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
def expected_max_binomial_exact_scipy(n, p, N):
    """
    If X1,…,XN are i.i.d. with CDF
      F_X(m) = P(Xi ≤ m),
    then the CDF of their maximum X^\dag = max_i Xi is
      F_M(m) = P(M ≤ m) = [F_X(m)]^N.

    Tail‑sum for expectation:
      E[X^\dag] = sum_{m=0}^{n-1} [1 - F_{X^\dag}(m)]
           = sum_{m=0}^{n-1} [1 - (F_X(m))^N].

    Exact computation using scipy.stats.binom:
      F_X(m) = binom.cdf(m, n, p)
      E[X^\dag] = sum_{m=0}^{n-1} (1 - F_X(m)^N)
    """
    m = np.arange(n)
    F = binom.cdf(m, n, p)
    Exp_Max = np.sum(1 - F**N)
    return Exp_Max
def get_trashold(K_M, p,N):
    """
    Compute the critical trahold -log(binom.pmf(Exp_Max,K_M,p)
    Using the expected maximum number of successes over
    1e6 repetitions of binomial trials 
    """
    # Approximate extreme number of trials
    Exp_max_succ = expected_max_binomial_exact_scipy(K_M, p, N)
    # interpolate between the two integers
    A = int(Exp_max_succ)
    A1 = -np.log(binom.pmf(A,K_M,p  ))
    B = int(Exp_max_succ)+1
    B1 = -np.log(binom.pmf(B,K_M,p  ))
    C = Exp_max_succ-A
    NLP_star = A1*(1-C) + B1*C
    
    return NLP_star



def expected_quantile_binomial_exact_scipy(n, p, N, q):
    """
    Expected q-quantile of N i.i.d. Binomial(n, p) counts.

    Let Xi ~ Binomial(n, p)   (i = 1,…,N)
    Let r = ceil(q N)         be the order-statistic index (1 ≤ r ≤ N)
    Let F_X(m) = P(Xi ≤ m)    be the single-trial CDF at m ∈ {0,…,n}

    For each threshold m, the probability that AT LEAST r of the N samples
    fall ≤ m is
        P(X_(r) ≤ m) = 1 – BinomCDF(k = r-1 ; N trials, success prob = F_X(m))

    Using the tail-sum identity,
        E[X_(r)] = Σ_{m=0}^{n-1} P(X_(r) > m)
                  = Σ_{m=0}^{n-1} BinomCDF(k = r-1 ; N, p = F_X(m))
    """
    # order-statistic index corresponding to the desired quantile
    r = int(np.ceil(q * N))
    if r < 1 or r > N:
        raise ValueError("q must satisfy 0 < q ≤ 1")

    # m = 0 … n-1
    m = np.arange(n, dtype=int)

    # single-trial CDF F_X(m) for each m
    F = binom.cdf(m, n, p)        # shape (n,)

    # Binomial CDF with array p broadcasts over m:
    #   P{ ≤ r-1 successes } when success-probability = F(m)
    tail = binom.cdf(r - 1, N, F) # shape (n,)

    return float(np.sum(tail))