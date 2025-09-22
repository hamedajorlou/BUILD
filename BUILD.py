import numpy as np
from Greedy_prune import best_greedy, full_greedy_and_prune
from utils import *


def BUILDER(
    Theta0,
    X,
    emp_cov,
    best_k,
    best_thr,
    *,
    edge_threshold=0.3,
    refresh_every=0.5,   
    var=1.0,   
    verbose=False           
):

    p = X.shape[1]
    if not (0.0 < refresh_every <= 1.0):
        raise ValueError("refresh_every must be in (0, 1].")

    A_est = np.zeros((p, p), dtype=float)

    Theta_aux = Theta0.astype(float).copy()

    processed = np.zeros(p, dtype=bool)

    refresh_points = set()
    k = 1
    while True:
        target = int(np.floor(p * (k * refresh_every)))
        if target <= 0 or target >= p:  # don't refresh at 0 or after all removed
            break
        refresh_points.add(target)
        k += 1


    removed_count = 0
    step = 0

    def find_next_leaf(theta_aux):
        diag_vals = np.diag(theta_aux)
        diag_vals_masked = np.where(processed | (diag_vals < edge_threshold), np.inf, diag_vals)
        i = np.argmin(diag_vals_masked)
        if np.isinf(diag_vals_masked[i]):
            return None
        return i

    while True:
        if removed_count in refresh_points:
            active_idx = np.where(~processed)[0]
            if active_idx.size > 1:
                X_sub = X[:, active_idx]
                emp_cov_sub = emp_cov[np.ix_(active_idx, active_idx)]
                Theta_refresh = full_greedy_and_prune(X_sub, emp_cov_sub, best_k, best_thr)
                Theta_aux[np.ix_(active_idx, active_idx)] = Theta_refresh
        i = find_next_leaf(Theta_aux)
        if i is None:
            break

        processed[i] = True
        removed_count += 1

        a_i = - var * Theta_aux[i, :].astype(float)
        a_i[i] = 0.0
        a_i[np.abs(a_i) < edge_threshold] = 0.0
        A_est[:, i] = a_i
        parents = np.where(np.abs(a_i) >= edge_threshold)[0]

        for j in parents:
            Theta_aux[j, :] -= (1/var) * a_i[j] * a_i
        Theta_aux[i, :] = 0.0
        Theta_aux[:, i] = 0.0
        step += 1
    
    return A_est


def BUILD(X, emp_cov, k_list, threshold_list, topo_thr, refresh_every, var=1.0, metric="frobenius"):
    best = best_greedy(X, emp_cov, k_list, threshold_list, metric)
    Theta0 = best['prec']
    A_est = BUILDER(
        Theta0=Theta0, X=X, emp_cov=emp_cov,
        best_k=best['k'], best_thr=best['threshold'],
        edge_threshold=topo_thr, refresh_every=refresh_every, var=var
    )
    A_est_bin = to_bin(A_est, thr=topo_thr)
    return {"prec": Theta0, "A_est": A_est, "A_est_bin": A_est_bin, "k": best['k'], "thr": best['threshold']}
