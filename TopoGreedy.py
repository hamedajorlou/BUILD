import numpy as np
from Greedy_prune import best_greedy, full_greedy_and_prune
from utils import *

# def estimate_adjacency_matrix_sequential(theta_matrix, p, edge_threshold=0.2):
#     A_est = np.zeros((p, p))
#     Theta_aux = theta_matrix.copy()
#     processed = np.zeros(p, dtype=bool)

#     def find_next_leaf(theta_aux):
#         diag_vals = np.diag(theta_aux)
#         diag_vals_masked = np.where(processed | (diag_vals < edge_threshold), np.inf, diag_vals)
#         i = np.argmin(diag_vals_masked)
#         if np.isinf(diag_vals_masked[i]):
#             return None
#         return i

#     while True:
#         i = find_next_leaf(Theta_aux)
#         if i is None:
#             break
#         processed[i] = True
#         a_i = -Theta_aux[i, :]
#         a_i[np.abs(a_i) < edge_threshold] = 0
#         A_est[i, :] = a_i
#         parents = np.where(np.abs(a_i) >= edge_threshold)[0]
#         Theta_aux[:, i] += a_i
#         Theta_aux[i, :] += a_i
#         Theta_aux[i, i] = 0
#         for j in parents:
#             Theta_aux[j, :] -= a_i[j] * a_i
#         Theta_aux[i, :] = 0
#         Theta_aux[:, i] = 0

#     A_est[np.abs(A_est) < edge_threshold] = 0
#     np.fill_diagonal(A_est, 0)
#     # return np.triu(A_est.T, k=-1)
#     return A_est.T

def BUILDER1(
    Theta0,
    X,
    emp_cov,
    best_k,
    best_thr,
    *,
    edge_threshold=0.3,
    refresh_every=0.5,      
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

        a_i = -Theta_aux[i, :].astype(float)
        a_i[i] = 0.0
        a_i[np.abs(a_i) < edge_threshold] = 0.0
        A_est[:, i] = a_i
        parents = np.where(np.abs(a_i) >= edge_threshold)[0]

        for j in parents:
            Theta_aux[j, :] -= a_i[j] * a_i
        Theta_aux[i, :] = 0.0
        Theta_aux[:, i] = 0.0
        step += 1
    
    return A_est



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





# def BUILD(
#     Theta0,
#     X,
#     emp_cov,
#     best_k,
#     best_thr,
#     *,
#     edge_threshold=0.3,
#     refresh_every=0.5,
#     max_refreshes=None
# ):
#     p = X.shape[1]
#     if not (0.0 < refresh_every <= 1.0):
#         raise ValueError("refresh_every must be in (0, 1].")

#     A_est = np.zeros((p, p), dtype=float)
#     Theta_aux = Theta0.astype(float).copy()
#     processed = np.zeros(p, dtype=bool)

#     refresh_points = set()
#     k = 1
#     while True:
#         target = int(np.floor(p * (k * refresh_every)))
#         if target <= 0 or target >= p:
#             break
#         refresh_points.add(target)
#         k += 1
#         if max_refreshes is not None and len(refresh_points) >= max_refreshes:
#             break

#     removed_count = 0
#     step = 0

#     def find_next_leaf(theta_aux):
#         diag_vals = np.diag(theta_aux)
#         diag_vals_masked = np.where(processed | (diag_vals < edge_threshold), np.inf, diag_vals)
#         i = np.argmin(diag_vals_masked)
#         if np.isinf(diag_vals_masked[i]):
#             return None
#         return i

#     while True:
#         if removed_count in refresh_points:
#             active_idx = np.where(~processed)[0]
#             if active_idx.size > 1:
#                 X_sub = X[:, active_idx]
#                 emp_cov_sub = emp_cov[np.ix_(active_idx, active_idx)]
#                 Theta_refresh = full_greedy_and_prune(X_sub, emp_cov_sub, best_k, best_thr)
#                 Theta_aux[np.ix_(active_idx, active_idx)] = Theta_refresh

#         i = find_next_leaf(Theta_aux)
#         if i is None:
#             break

#         processed[i] = True
#         removed_count += 1

#         a_i = -Theta_aux[i, :].astype(float)
#         a_i[i] = 0.0
#         a_i[np.abs(a_i) < edge_threshold] = 0.0
#         A_est[i, :] = a_i
#         parents = np.where(np.abs(a_i) >= edge_threshold)[0]

#         for j in parents:
#             Theta_aux[j, :] -= a_i[j] * a_i

#         Theta_aux[i, :] = 0.0
#         Theta_aux[:, i] = 0.0

#         step += 1

#     np.fill_diagonal(A_est, 0.0)
#     return A_est.T


# # ---------------------------------------------
# # Refreshed variant: midpoint precision refresh
# # ---------------------------------------------
# def estimate_adjacency_matrix_sequential_with_refresh(
#     Theta0, X, emp_cov, best_k, best_thr,
#     edge_threshold=0.2, refresh_at=0.5
# ):
#     """
#     Same as before, but now prints Θ and A at every step.
#     This will get very verbose for large p — best for small toy runs (p ≤ 8–12).
#     """

#     p = X.shape[1]
#     A_est = np.zeros((p, p))
#     Theta_aux = Theta0.copy()
#     processed = np.zeros(p, dtype=bool)

#     refresh_after = int(np.floor(p * float(refresh_at)))
#     did_refresh = False
#     removed_count = 0
#     step = 0

#     def find_next_leaf(theta_aux):
#         diag_vals = np.diag(theta_aux)
#         diag_vals_masked = np.where(processed | (diag_vals < edge_threshold), np.inf, diag_vals)
#         i = np.argmin(diag_vals_masked)
#         if np.isinf(diag_vals_masked[i]):
#             return None
#         return i

#     while True:
#         # ----- REFRESH -----
#         if (not did_refresh) and (removed_count >= refresh_after):
#             active_idx = np.where(~processed)[0]
#             print(f"\n=== REFRESH at step {step}, {len(active_idx)} active nodes ===")
#             if len(active_idx) > 1:
#                 X_sub = X[:, active_idx]
#                 emp_cov_sub = emp_cov[np.ix_(active_idx, active_idx)]
#                 Theta_refresh = full_greedy_and_prune(X_sub, emp_cov_sub, best_k, best_thr)
#                 Theta_aux[np.ix_(active_idx, active_idx)] = Theta_refresh
#                 print("Re-estimated Θ block:\n", Theta_refresh)
#             did_refresh = True

#         # ----- CHOOSE LEAF -----
#         i = find_next_leaf(Theta_aux)
#         if i is None:
#             break

#         print(f"\n--- Step {step}: removing leaf {i} ---")
#         print("Θ diagonal:", np.diag(Theta_aux))
#         print("Row before elimination (−Θ[i,:]):", -Theta_aux[i, :])

#         processed[i] = True
#         removed_count += 1

#         # ----- INFER PARENTS / UPDATE A -----
#         a_i = -Theta_aux[i, :].copy()
#         a_i[np.abs(a_i) < edge_threshold] = 0.0
#         A_est[i, :] = a_i
#         parents = np.where(np.abs(a_i) >= edge_threshold)[0]
#         print("Parents inferred:", parents)
#         print("Adjacency row A[i,:]:", A_est[i, :])

#         # ----- UPDATE Θ -----
#         Theta_aux[:, i] += a_i
#         Theta_aux[i, :] += a_i
#         Theta_aux[i, i] = 0
#         for j in parents:
#             Theta_aux[j, :] -= a_i[j] * a_i
#         Theta_aux[i, :] = 0
#         Theta_aux[:, i] = 0

#         print("Θ after elimination:\n", Theta_aux)

#         step += 1

#     # ----- FINAL ADJACENCY -----
#     A_est[np.abs(A_est) < edge_threshold] = 0
#     np.fill_diagonal(A_est, 0)
#     # A_tri = np.triu(A_est.T, k=-1)
#     print("\n=== Final Adjacency ===\n", A_est.T)

#     return A_est.T


# import heapq
# import numpy as np

# def estimate_adjacency_matrix_sequential_with_multi_refresh_no_diag_heap(
#     Theta0,
#     X,
#     emp_cov,
#     best_k,
#     best_thr,
#     edge_threshold=0.3,
#     refresh_every=0.5,
#     verbose=False
# ):
#     """
#     Min-heap variant: selects the next leaf by popping the smallest valid diagonal from a heap.
#     """
#     p = X.shape[1]
#     if not (0.0 < refresh_every <= 1.0):
#         raise ValueError("refresh_every must be in (0, 1].")

#     A_est = np.zeros((p, p), dtype=float)
#     Theta_aux = Theta0.astype(float).copy()
#     processed = np.zeros(p, dtype=bool)

#     # Build set of refresh points
#     refresh_points = set()
#     k = 1
#     while True:
#         target = int(np.floor(p * (k * refresh_every)))
#         if target <= 0 or target >= p:
#             break
#         refresh_points.add(target)
#         k += 1

#     # Initialize diagonal cache and min-heap (lazy deletion strategy)
#     diag_vals = np.diag(Theta_aux).astype(float).copy()
#     heap = [(diag_vals[i], i) for i in range(p) if diag_vals[i] >= edge_threshold]
#     heapq.heapify(heap)

#     removed_count = 0

#     def pop_next_leaf():
#         # Pop until we find an unprocessed index with up-to-date diagonal >= edge_threshold
#         while heap:
#             val, i = heapq.heappop(heap)
#             if processed[i]:
#                 continue
#             if not np.isfinite(diag_vals[i]) or diag_vals[i] < edge_threshold:
#                 continue
#             # Lazy delete: skip stale entries whose value no longer matches current diag
#             if abs(val - diag_vals[i]) > 0:
#                 continue
#             return i
#         return None

#     while True:
#         # Refresh precision on active set at scheduled removal counts
#         if removed_count in refresh_points:
#             active_idx = np.where(~processed)[0]
#             if active_idx.size > 1:
#                 X_sub = X[:, active_idx]
#                 emp_cov_sub = emp_cov[np.ix_(active_idx, active_idx)]
#                 Theta_refresh = full_greedy_and_prune(X_sub, emp_cov_sub, best_k, best_thr)
#                 Theta_aux[np.ix_(active_idx, active_idx)] = Theta_refresh

#                 # Update diagonals for active nodes and push fresh entries to heap
#                 diag_vals[active_idx] = np.diag(Theta_refresh)
#                 for j in active_idx:
#                     if diag_vals[j] >= edge_threshold:
#                         heapq.heappush(heap, (diag_vals[j], j))

#         i = pop_next_leaf()
#         if i is None:
#             break

#         processed[i] = True
#         removed_count += 1

#         # Parent extraction from row i
#         a_i = -Theta_aux[i, :].astype(float)
#         a_i[np.abs(a_i) < edge_threshold] = 0.0
#         A_est[i, :] = a_i
#         parents = np.where(np.abs(a_i) >= edge_threshold)[0]

#         # Rank-1 style elimination updates only for parent rows
#         # Theta_aux[j,:] -= a_i[j] * a_i ; update their diagonals accordingly
#         for j in parents:
#             Theta_aux[j, :] -= a_i[j] * a_i
#             diag_vals[j] -= (a_i[j] * a_i[j])
#             if not processed[j] and diag_vals[j] >= edge_threshold:
#                 heapq.heappush(heap, (diag_vals[j], j))

#         # Zero out removed node in Theta
#         Theta_aux[i, :] = 0.0
#         Theta_aux[:, i] = 0.0
#         diag_vals[i] = np.inf  # keep it out of future consideration

#     return A_est.T



# # ---------------------------------------------
# # TopoGreedy DAG
# # ---------------------------------------------



# def TopoGreedy(X, emp_cov, k_list, threshold_list, topo_thr, metric="frobenius"):
#     best = best_greedy(X, emp_cov, k_list, threshold_list, metric)
#     Theta = best['prec']
#     A_est = estimate_adjacency_matrix_sequential(Theta, X.shape[1], edge_threshold=topo_thr)
#     A_est_bin = to_bin(A_est, thr=topo_thr)
#     return {"prec": Theta, "A_est": A_est, "A_est_bin": A_est_bin, "k": best['k'], "thr": best['threshold']}



# def TopoGreedy_refresh_BUILD(X, emp_cov, k_list, threshold_list, topo_thr, refresh_every, metric="frobenius"):
#     best = best_greedy(X, emp_cov, k_list, threshold_list, metric)
#     Theta0 = best['prec']
#     A_est = BUILD(
#         Theta0=Theta0, X=X, emp_cov=emp_cov,
#         best_k=best['k'], best_thr=best['threshold'],
#         edge_threshold=topo_thr, refresh_every=refresh_every
#     )
#     A_est_bin = to_bin(A_est, thr=topo_thr)
#     return {"prec": Theta0, "A_est": A_est, "A_est_bin": A_est_bin, "k": best['k'], "thr": best['threshold']}


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


# def TopoGreedy_refresh_3(X, emp_cov, k_list, threshold_list, topo_thr, refresh_every, metric="frobenius"):
#     best = best_greedy(X, emp_cov, k_list, threshold_list, metric)
#     Theta0 = best['prec']
#     A_est = estimate_adjacency_matrix_sequential_with_multi_refresh_no_diag_heap(
#         Theta0=Theta0, X=X, emp_cov=emp_cov,
#         best_k=best['k'], best_thr=best['threshold'],
#         edge_threshold=topo_thr, refresh_every=refresh_every
#     )
#     A_est_bin = to_bin(A_est, thr=topo_thr)
#     return {"prec": Theta0, "A_est": A_est, "A_est_bin": A_est_bin, "k": best['k'], "thr": best['threshold']}


# def estimate_adjacency_matrix_sequential_with_multi_refresh_2(
#     Theta0,
#     X,
#     emp_cov,
#     best_k,
#     best_thr,
#     *,
#     edge_threshold=0.3,      # used ONLY for edge extraction / sparsifying a_i
#     diag_threshold=None,     # NEW: used ONLY for leaf eligibility via diag(Θ) >= diag_threshold
#     refresh_every=0.5,       # e.g., 0.1 => refresh 10 times along the way
#     max_refreshes=None,      # optional cap on number of refreshes
#     verbose=False            # print progress, Θ rows, and adjacency rows as we go
# ):
#     """
#     Sequential leaf removal with *periodic precision refreshes*.

#     Behavior:
#       - Start from Theta0 (precision estimated on the full set).
#       - Remove one "leaf" at a time: pick the smallest diagonal entry among nodes whose
#         diag(Θ) >= diag_threshold (and not processed).
#       - At multiples of `refresh_every` (as a fraction of p nodes removed), re-estimate precision
#         on the REMAINING active variables and splice that block back into the working Θ.
#       - Continue until all nodes are removed.

#     Notes:
#       - edge_threshold is ONLY for sparsifying/deciding edges from a_i = -Θ[i,:].
#       - diag_threshold is ONLY for leaf eligibility (separate knob from edge_threshold).
#       - For large p, consider setting verbose=False (it prints a lot).
#     """

#     import numpy as np

#     p = X.shape[1]
#     if not (0.0 < refresh_every <= 1.0):
#         raise ValueError("refresh_every must be in (0, 1].")

#     # Back-compat: if diag_threshold not provided, use the previous behavior
#     # (eligibility gated by edge_threshold).
#     diag_threshold_eff = edge_threshold if (diag_threshold is None) else float(diag_threshold)

#     # --- adjacency we build during elimination ---
#     A_est = np.zeros((p, p), dtype=float)

#     # --- working precision (mutated in-place) ---
#     Theta_aux = Theta0.astype(float).copy()

#     # --- processed flags ---
#     processed = np.zeros(p, dtype=bool)

#     # --- build refresh schedule in terms of removal counts ---
#     refresh_points = set()
#     k = 1
#     while True:
#         target = int(np.floor(p * (k * refresh_every)))
#         if target <= 0 or target >= p:  # don't refresh at 0 or after all removed
#             break
#         refresh_points.add(target)
#         k += 1
#         if max_refreshes is not None and len(refresh_points) >= max_refreshes:
#             break

#     if verbose:
#         pts = sorted(list(refresh_points))
#         print(f"[init] p={p}, refresh_every={refresh_every} -> refresh at removals: {pts}")
#         print(f"[init] edge_threshold={edge_threshold}, diag_threshold={diag_threshold_eff}")

#     removed_count = 0
#     step = 0

#     def find_next_leaf(theta_aux):
#         # Pick the smallest diagonal among unprocessed nodes with diag >= diag_threshold_eff
#         diag_vals = np.diag(theta_aux)
#         mask_bad = processed | (diag_vals < diag_threshold_eff)
#         diag_vals_masked = np.where(mask_bad, np.inf, diag_vals)
#         i = np.argmin(diag_vals_masked)
#         if np.isinf(diag_vals_masked[i]):
#             return None
#         return i

#     while True:
#         # --------------------------
#         # refresh if we hit a point
#         # --------------------------
#         if removed_count in refresh_points:
#             active_idx = np.where(~processed)[0]
#             if verbose:
#                 print(f"\n=== REFRESH at removal #{removed_count} (active={len(active_idx)}) ===")
#             if active_idx.size > 1:
#                 X_sub = X[:, active_idx]
#                 emp_cov_sub = emp_cov[np.ix_(active_idx, active_idx)]
#                 Theta_refresh = full_greedy_and_prune(X_sub, emp_cov_sub, best_k, best_thr)
#                 Theta_aux[np.ix_(active_idx, active_idx)] = Theta_refresh
#                 if verbose:
#                     fro = np.linalg.norm(Theta_refresh, ord='fro')
#                     print(f"[refresh] ||Theta_refresh||_F = {fro:.4f}")
#             else:
#                 if verbose:
#                     print("[refresh] skipped (<=1 active node).")

#         # --------------------------
#         # choose next leaf
#         # --------------------------
#         i = find_next_leaf(Theta_aux)
#         if i is None:
#             if verbose:
#                 print("\n[done] No eligible leaf remains (diag gate).")
#             break

#         if verbose:
#             print(f"\n--- step {step} | remove leaf i={i} ---")
#             print("[diag Θ]:", np.diag(Theta_aux))
#             print("[row -Θ[i,:]]:", -Theta_aux[i, :])

#         processed[i] = True
#         removed_count += 1

#         # --------------------------
#         # infer parents & write A row
#         # --------------------------
#         a_i = -Theta_aux[i, :].astype(float)
#         a_i[np.abs(a_i) < edge_threshold] = 0.0
#         A_est[i, :] = a_i
#         parents = np.where(np.abs(a_i) >= edge_threshold)[0]
#         if verbose:
#             print(f"[parents] idx={parents.tolist()} (|a_i|>= {edge_threshold})")
#             print("[A[i,:]]:", A_est[i, :])

#         # --------------------------
#         # elimination-style updates (kept as-is)
#         # --------------------------
#         Theta_aux[:, i] += a_i
#         Theta_aux[i, :] += a_i
#         Theta_aux[i, i] = 0.0
#         for j in parents:
#             Theta_aux[j, :] -= a_i[j] * a_i
#         Theta_aux[i, :] = 0.0
#         Theta_aux[:, i] = 0.0

#         if verbose:
#             active_idx = np.where(~processed)[0]
#             fro_active = np.linalg.norm(Theta_aux[np.ix_(active_idx, active_idx)], ord='fro') if active_idx.size else 0.0
#             print(f"[post] active={len(active_idx)}, ||Θ_active||_F={fro_active:.4f}")

#         step += 1

#     # finalize adjacency
#     A_est[np.abs(A_est) < edge_threshold] = 0.0
#     np.fill_diagonal(A_est, 0.0)
#     return A_est.T


# # def TopoGreedy_refresh_2(
# #     X, emp_cov, k_list, threshold_list,
# #     topo_thr, refresh_every,
# #     *, diag_threshold=None, metric="frobenius"
# # ):
# #     """
# #     Wrapper that:
# #       1) Finds a good starting precision Θ0 via best_greedy,
# #       2) Runs sequential elimination with periodic refresh,
# #       3) Returns weighted & binary adjacency plus the best hyperparams.

# #     Set diag_threshold to control leaf eligibility separately from topo_thr.
# #     Typical choices: 0.0, 1e-6, or a small percentile of diag(Θ0).
# #     """
# #     best = best_greedy(X, emp_cov, k_list, threshold_list, metric)
# #     Theta0 = best['prec']
# #     A_est = estimate_adjacency_matrix_sequential_with_multi_refresh_2(
# #         Theta0=Theta0, X=X, emp_cov=emp_cov,
# #         best_k=best['k'], best_thr=best['threshold'],
# #         edge_threshold=topo_thr,
# #         diag_threshold=diag_threshold,       # NEW: decoupled from topo_thr
# #         refresh_every=refresh_every
# #     )
# #     A_est_bin = to_bin(A_est, thr=topo_thr)
# #     return {"prec": Theta0, "A_est": A_est, "A_est_bin": A_est_bin, "k": best['k'], "thr": best['threshold']}
