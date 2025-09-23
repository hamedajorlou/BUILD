import numpy as np
from numpy.linalg import pinv, LinAlgError
from typing import List, Dict
from scipy.linalg import norm
from scipy.stats import pearsonr

def greedy(X: np.ndarray, k: int, i: int) -> List[int]:
    n, p = X.shape
    k = min(k, p - 1)
    S = []
    X_S = X.copy()

    for _ in range(k):
        dots = X_S[:, i] @ X_S
        norms = np.sqrt(np.sum(X_S ** 2, axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = np.abs(dots) / norms
        scores[i] = np.nan
        for s in S:
            scores[s] = np.nan

        if np.all(np.isnan(scores)):
            break
        pivot = np.nanargmax(scores)
        dots_pivot = X_S[:, pivot] @ X_S
        X_S = X_S - np.outer(X_S[:, pivot], dots_pivot / (norms[pivot] ** 2))
        S.append(pivot)

    return S

def greedy_old(emp_cov: np.ndarray, k: int, i: int) -> List[int]:
    p = emp_cov.shape[0]
    k = min(k, p - 1)
    S = []
    cov = emp_cov.copy()

    for _ in range(k):
        scores = np.abs(cov[i, :]) / np.sqrt(np.abs(np.diag(cov)))
        scores[i] = np.nan
        for s in S:
            scores[s] = np.nan

        if np.all(np.isnan(scores)):
            break
        pivot = np.nanargmax(scores)
        cov = cov - np.outer(cov[:, pivot], cov[:, pivot]) / cov[pivot, pivot]
        S.append(pivot)

    return S

def prune(X: np.ndarray, S: List[int], i: int, threshold: float) -> List[int]:
    if not S:
        return []
    X_S = X[:, S]
    x_i = X[:, i]
    M = X_S.T @ X_S
    v = X_S.T @ x_i
    try:
        w = np.linalg.solve(M, v)
    except LinAlgError:
        w = pinv(M) @ v
    err = norm(x_i - X_S @ w) ** 2
    S_final = []
    for idx, j in enumerate(S):
        S_noj = S[:idx] + S[idx+1:]
        if not S_noj:
            continue
        X_S_noj = X[:, S_noj]
        M_noj = X_S_noj.T @ X_S_noj
        v_noj = X_S_noj.T @ x_i
        try:
            w_noj = np.linalg.solve(M_noj, v_noj)
        except LinAlgError:
            w_noj = pinv(M_noj) @ v_noj
        err_noj = norm(x_i - X_S_noj @ w_noj) ** 2
        if err_noj > (1 + threshold) * err:
            S_final.append(j)
    return S_final

def greedy_and_prune(X: np.ndarray, emp_cov: np.ndarray, i: int, k: int, threshold: float) -> List[int]:
    if X.shape[0] < X.shape[1] * 0.7:
        S = greedy(X, k, i)
    else:
        S = greedy_old(emp_cov, k, i)
    return prune(X, S, i, threshold)

def full_greedy_and_prune(X: np.ndarray, emp_cov: np.ndarray, k: int, threshold: float) -> np.ndarray:
    n = X.shape[1]
    S = [greedy_and_prune(X, emp_cov, i, k, threshold) for i in range(n)]
    for i in range(n):
        S[i] = [j for j in S[i] if i in S[j]]
    prec = np.zeros((n, n))
    for i in range(n):
        U = S[i]
        if not U:
            prec[i, i] = 1 / (emp_cov[i, i] + 1e-12)
            continue
        X_U = X[:, U]
        x_i = X[:, i]
        try:
            w = pinv(X_U) @ x_i
        except LinAlgError:
            w = pinv(X_U) @ x_i
        var_i = norm(x_i - X_U @ w) ** 2 / max(1, X.shape[0]-len(U))
        var_i = max(var_i, 1e-12)
        prec[i, i] = 1 / var_i
        for idx, j in enumerate(U):
            w_j = w[idx]
            new_val = -w_j * prec[i, i]
            if prec[i, j] == 0 or abs(prec[i, j]) > abs(new_val):
                prec[i, j] = new_val
            prec[j, i] = prec[i, j]
    return prec

def best_greedy(X: np.ndarray,
                emp_cov: np.ndarray,
                k_list: List[int],
                threshold_list: List[float],
                metric: str = "frobenius") -> Dict[str, object]:
    Theta = full_greedy_and_prune(X, emp_cov, k_list[0], threshold_list[0])
    best_err = 0
    best_k = k_list[0]
    best_threshold = threshold_list[0]
    return {
        "prec": Theta,
        "normed_err": best_err,
        "k": best_k,
        "threshold": best_threshold
    }
