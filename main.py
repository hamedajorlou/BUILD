# -*- coding: utf-8 -*-
"""
Unified SEM experiment runner for multiple DAG-learning baselines.
- Simulates data
- Runs baselines (CoLiDE-EV/NV, DAGMA(l2), Nonneg-DAGMA (PGD/FISTA/ADAM), MetMulDagma,
  NOTEARS(l2), TopoGreedy, TopoGreedy_refresh)
- Computes metrics (SHD/TPR/FDR/F1, Frobenius error, acyclicity proxy, runtime, DAG rate)
- Saves results to .npz + JSON sidecar

Assumptions:
- The following are already importable in your environment:
    utils, colide_ev, colide_nv, DAGMA_linear, Nonneg_dagma, MetMulDagma, notears_linear,
    TopoGreedy, TopoGreedy_refresh, best_greedy, full_greedy_and_prune, etc.
- N_CPUS, n_dags, N_samples, Exps from your older snippets are NOT required; this script defines its own.
"""

import os
import json
import time
import numpy as np
from time import perf_counter
from typing import Dict, Any, Tuple
from sklearn.metrics import f1_score
from Baselines import *
from dag_utils import *
from utils import *


# ------------------------
# your helper deps (assumed)
# ------------------------
# import utils  # must provide: simulate_sem, standarize, to_bin, count_accuracy, compute_norm_sq_err, is_dag

# ------------------------
# Baselines (assumed pre-imported)
# ------------------------
# from your pasted code they should already be available:
# colide_ev, colide_nv, DAGMA_linear, Nonneg_dagma, MetMulDagma, notears_linear
# TopoGreedy, TopoGreedy_refresh, best_greedy, full_greedy_and_prune


# =======================
# Common helpers
# =======================

def get_lamb_value(n_nodes, n_samples, times=1.0):
    """Adaptive lambda (same formula you used)."""
    return np.sqrt(np.log(n_nodes) / max(1, n_samples)) * times

def safe_to_bin(A, thr):
    B = A.copy().astype(float)
    B[np.abs(B) < thr] = 0.0
    B[B != 0.0] = 1.0
    np.fill_diagonal(B, 0.0)
    return B

def acyclicity_proxy(W: np.ndarray) -> float:
    """
    Cheap acyclicity proxy: run utils.is_dag on binarized adjacency (symmetrize small residuals).
    Return 0.0 if DAG, 1.0 otherwise (lower is better).
    If you have each model's .dagness, you can swap this for a real scalar.
    """
    try:
        A_bin = to_bin(W, 1e-12)
        return 0.0 if is_dag(A_bin) else 1.0
    except Exception:
        return float('nan')


# =======================
# Adapters (unify .fit -> .W_est/.Theta_est)
# =======================

class BaseAdapter:
    name = "base"
    def fit(self, X: np.ndarray, **kwargs):
        raise NotImplementedError
    def dagness(self, W: np.ndarray) -> float:
        return acyclicity_proxy(W)

class NOTEARSAdapter(BaseAdapter):
    name = "NOTEARS(l2)"
    def fit(self, X: np.ndarray, **kwargs):
        # kwargs must include: lambda1
        lam = kwargs.get("lambda1", 0.02)
        wthr = kwargs.get("w_threshold", 0.3)
        W = notears_linear(X, lambda1=lam, loss_type="l2",
                           max_iter=kwargs.get("max_iter", 100),
                           h_tol=kwargs.get("h_tol", 1e-8),
                           rho_max=kwargs.get("rho_max", 1e+16),
                           w_threshold=wthr)
        self.W_est = W
        self.Theta_est = None
        return self

class DAGMAAdapter(BaseAdapter):
    name = "DAGMA(l2)"
    def __init__(self, **init):
        self.model = DAGMA_linear(loss_type='l2', **init)
    def fit(self, X: np.ndarray, **kwargs):
        W = self.model.fit(X, **kwargs)  # returns W
        self.W_est = W
        self.Theta_est = None
        return self
    def dagness(self, W: np.ndarray) -> float:
        try:
            return float(self.model.dagness(W))
        except Exception:
            return super().dagness(W)

class CoLiDEEVAdapter(BaseAdapter):
    name = "CoLiDE-EV"
    def __init__(self, **init):
        self.model = colide_ev(**init)
    def fit(self, X: np.ndarray, **kwargs):
        W, sig = self.model.fit(X, **kwargs)
        self.W_est = W
        self.Theta_est = None
        return self
    def dagness(self, W: np.ndarray) -> float:
        try:
            return float(self.model.dagness(W))
        except Exception:
            return super().dagness(W)

class CoLiDENVAdapter(BaseAdapter):
    name = "CoLiDE-NV"
    def __init__(self, **init):
        self.model = colide_nv(**init)
    def fit(self, X: np.ndarray, **kwargs):
        W, sig = self.model.fit(X, **kwargs)
        self.W_est = W
        self.Theta_est = None
        return self
    def dagness(self, W: np.ndarray) -> float:
        try:
            return float(self.model.dagness(W))
        except Exception:
            return super().dagness(W)

class NonnegDAGMAAdapter(BaseAdapter):
    name = "Nonneg-DAGMA"
    def __init__(self, primal_opt='pgd', acyclicity='logdet'):
        self.model = Nonneg_dagma(primal_opt=primal_opt, acyclicity=acyclicity)
    def fit(self, X: np.ndarray, **kwargs):
        W = self.model.fit(X, **kwargs)
        self.W_est = W
        self.Theta_est = None
        return self
    def dagness(self, W: np.ndarray) -> float:
        try:
            return float(self.model.dagness(W))
        except Exception:
            return super().dagness(W)

class MetMulDAGMAAdapter(BaseAdapter):
    name = "MetMul-DAGMA"
    def __init__(self):
        self.model = MetMulDagma()
    def fit(self, X: np.ndarray, **kwargs):
        W = self.model.fit(X, **kwargs)
        self.W_est = W
        self.Theta_est = None
        return self
    def dagness(self, W: np.ndarray) -> float:
        try:
            return float(self.model.dagness(W))
        except Exception:
            return super().dagness(W)

class TopoGreedyAdapter(BaseAdapter):
    name = "TopoGreedy"
    def fit(self, X: np.ndarray, **kwargs):
        # needs emp_cov, k_list, threshold_list, topo_thr
        emp_cov = kwargs["emp_cov"]
        k_list = kwargs["k_list"]
        threshold_list = kwargs["threshold_list"]
        topo_thr = kwargs["topo_thr"]
        out = TopoGreedy(X, emp_cov, k_list, threshold_list, topo_thr)
        # Align orientation with other baselines: use weighted adjacency W_est
        # Your earlier pipeline transposed in the aggregator; here we set W_est to A_est.T directly.
        self.W_est = out["A_est"].T
        self.Theta_est = out["prec"]
        return self

class TopoGreedyRefreshAdapter(BaseAdapter):
    name = "TopoGreedy_refresh"
    def fit(self, X: np.ndarray, **kwargs):
        emp_cov = kwargs["emp_cov"]
        k_list = kwargs["k_list"]
        threshold_list = kwargs["threshold_list"]
        topo_thr = kwargs["topo_thr"]
        refresh_every = kwargs["refresh_every"]
        out = TopoGreedy_refresh(X, emp_cov, k_list, threshold_list, topo_thr, refresh_every)
        self.W_est = out["A_est"].T
        self.Theta_est = out["prec"]
        return self


# =======================
# Experiment core
# =======================

def run_one_setting(
    rng_seed: int,
    data_p: Dict[str, Any],
    n_samples: int,
    baselines: Dict[str, Tuple[BaseAdapter, Dict[str, Any], Dict[str, Any]]],
    edge_thr_eval: float = 0.2,
    standarize_X: bool = True
) -> Dict[str, Any]:
    """
    Simulate one SEM instance (fixed seed), run all baselines, return metrics per baseline.
    baselines: {name: (adapter_instance, init_kwargs, fit_kwargs_template)}
      - init_kwargs: used to re-instantiate/fresh adapter per run
      - fit_kwargs_template: copied per run; may update adaptively (e.e., lambda)
    """
    np.random.seed(rng_seed)

    # --- simulate data ---
    dp = data_p.copy()
    dp['n_samples'] = n_samples
    W_true, _, X, Theta_true = simulate_sem(**dp)
    # X_std = standarize(X) if standarize_X else X
    W_true_bin = to_bin(W_true, edge_thr_eval)
    norm_W_true = np.linalg.norm(W_true)

    emp_cov = (X.T @ X) / float(X.shape[0])

    # --- run baselines ---
    results = {}
    for name, (adapter_proto, init_kwargs, fit_template) in baselines.items():
        # fresh adapter each time to avoid state bleed
        adapter = type(adapter_proto)(**init_kwargs) if init_kwargs is not None else type(adapter_proto)()

        # make per-run fit kwargs
        fit_kwargs = dict(fit_template) if fit_template is not None else {}

        # optional adaptive lambda logic
        if "lambda1" in fit_kwargs and fit_kwargs.get("_adapt_lambda", False):
            lam_times = fit_kwargs.pop("lambda1")
            fit_kwargs["lambda1"] = get_lamb_value(data_p['n_nodes'], n_samples, lam_times)
        # Topo variants need emp_cov etc.
        if isinstance(adapter, (TopoGreedyAdapter, TopoGreedyRefreshAdapter)):
            fit_kwargs["emp_cov"] = emp_cov

        t0 = perf_counter()
        try:
            adapter.fit(X, **fit_kwargs)
            ok = True
        except Exception as e:
            ok = False
            adapter.W_est = np.zeros_like(W_true)
            adapter.Theta_est = np.zeros_like(Theta_true)
            err_msg = str(e)
        t1 = perf_counter()

        W_est = adapter.W_est if adapter.W_est is not None else np.zeros_like(W_true)
        W_est_bin = to_bin(W_est, edge_thr_eval)

        # metrics
        if ok:
            metrics_dict = count_accuracy(W_true_bin, W_est_bin)
            shd = metrics_dict['shd']
            tpr = metrics_dict['tpr']
            fdr = metrics_dict['fdr']
            fscore = f1_score(W_true_bin.flatten(), W_est_bin.flatten())
            err = compute_norm_sq_err(W_true, W_est, norm_W_true)
            dagness_val = adapter.dagness(W_est)
            theta_rel = (np.linalg.norm((adapter.Theta_est if adapter.Theta_est is not None else 0) - Theta_true, 'fro') /
                         (np.linalg.norm(Theta_true, 'fro') + 1e-8))
            runtime = t1 - t0
            dag_ok = 1 if is_dag(W_est_bin) else 0
            results[name] = dict(
                ok=True, shd=shd, tpr=tpr, fdr=fdr, f1=fscore, rel_err=err,
                acyc=dagness_val, runtime=runtime, dag_ok=dag_ok, theta_rel=theta_rel
            )
        else:
            results[name] = dict(
                ok=False, error=err_msg, shd=np.nan, tpr=np.nan, fdr=np.nan,
                f1=np.nan, rel_err=np.nan, acyc=np.nan, runtime=t1 - t0,
                dag_ok=0, theta_rel=np.nan
            )

    return dict(
        seed=rng_seed,
        n_samples=n_samples,
        results=results
    )


def run_experiment_suite(
    out_dir: str,
    seeds: np.ndarray,
    N_samples: np.ndarray,
    data_p: Dict[str, Any],
    baselines: Dict[str, Tuple[BaseAdapter, Dict[str, Any], Dict[str, Any]]],
    edge_thr_eval: float = 0.2,
    standarize_X: bool = True
):
    os.makedirs(out_dir, exist_ok=True)

    # storage arrays
    B = list(baselines.keys())
    R = len(seeds)
    S = len(N_samples)

    shd   = np.full((R, S, len(B)), np.nan)
    tpr   = np.full_like(shd, np.nan)
    fdr   = np.full_like(shd, np.nan)
    f1    = np.full_like(shd, np.nan)
    err   = np.full_like(shd, np.nan)
    acyc  = np.full_like(shd, np.nan)
    timea = np.full_like(shd, np.nan)
    dagok = np.full_like(shd, np.nan)
    thetadiff = np.full_like(shd, np.nan)

    for ri, seed in enumerate(seeds):
        for si, n in enumerate(N_samples):
            print(f"[run] seed={seed} n={n}")
            pack = run_one_setting(
                rng_seed=seed,
                data_p=data_p,
                n_samples=int(n),
                baselines=baselines,
                edge_thr_eval=edge_thr_eval,
                standarize_X=standarize_X
            )
            res = pack["results"]
            for bi, bname in enumerate(B):
                r = res[bname]
                # Check if the run was successful before accessing metric values
                if r["ok"]:
                    shd[ri, si, bi]   = r["shd"]
                    tpr[ri, si, bi]   = r["tpr"]
                    fdr[ri, si, bi]   = r["fdr"]
                    f1[ri, si, bi]    = r["f1"]
                    err[ri, si, bi]   = r["rel_err"]
                    acyc[ri, si, bi]  = r["acyc"]
                    timea[ri, si, bi] = r["runtime"]
                    dagok[ri, si, bi] = r["dag_ok"]
                    thetadiff[ri, si, bi] = r["theta_rel"]


    # save arrays
    stamp = time.strftime("%Y%m%d-%H%M%S")
    npz_path = os.path.join(out_dir, f"sem_experiment_{stamp}.npz")
    np.savez_compressed(
        npz_path,
        baselines=np.array(B),
        seeds=np.array(seeds),
        N_samples=np.array(N_samples),
        shd=shd, tpr=tpr, fdr=fdr, f1=f1, rel_err=err, acyc=acyc, runtime=timea, dag_ok=dagok, theta_rel=thetadiff
    )

    # save config/meta
    meta = dict(
        out_npz=os.path.basename(npz_path),
        data_p=data_p,
        baselines=B,
        edge_thr_eval=edge_thr_eval,
        standarize_X=standarize_X
    )
    with open(os.path.join(out_dir, f"sem_experiment_{stamp}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] saved results to: {npz_path}")
    return npz_path


# =======================
# Define & launch
# =======================

if __name__ == "__main__":
    # ----- data model -----
    N = 50
    data_p = dict(
        n_nodes=N,
        graph_type='er',      # as in your earlier code
        edges=4*N,
        edge_type='positive',
        w_range = (0.5,1),
        var=1
        # n_samples will be filled per run
    )

    # ----- sampling schedule -----
    seeds = np.arange(1, 6)                 # 5 random DAGs (adjust)
    N_samples = np.array([200, 400, 800, 1000])   # sample sizes (adjust)
    edge_thr_eval = 0.5
    standarize_X = False

    # ----- baseline registry -----
    # Each entry: "name": (adapter_instance, init_kwargs, fit_kwargs_template)
    # You can toggle/adapt lambda with _adapt_lambda=True and set lambda1 as a multiplier for get_lamb_value.
    baselines: Dict[str, Tuple[BaseAdapter, Dict[str, Any], Dict[str, Any]]] = {
        "NOTEARS(l2)": (NOTEARSAdapter(), {},
                        dict(lambda1=0.02, w_threshold=0.3, max_iter=100)),

        "DAGMA(l2)": (DAGMAAdapter(), {},
                      dict(lambda1=0.02, T=5, mu_init=1.0, mu_factor=0.1,
                           s=[1.0, .9, .8, .7, .6], warm_iter=3e4, max_iter=6e4,
                           lr=3e-4, checkpoint=1000, beta_1=0.99, beta_2=0.999)),

        "CoLiDE-EV": (CoLiDEEVAdapter(), {},
                      dict(lambda1=0.02, T=5, mu_init=1.0, mu_factor=0.1,
                           s=[1.0, .9, .8, .7, .6], warm_iter=3e4, max_iter=6e4,
                           lr=3e-4, checkpoint=1000, beta_1=0.99, beta_2=0.999)),

        #"CoLiDE-NV": (CoLiDENVAdapter(), {},
        #              dict(lambda1=0.02, T=5, mu_init=1.0, mu_factor=0.1,
        #                   s=[1.0, .9, .8, .7, .6], warm_iter=3e4, max_iter=6e4,
        #                   lr=3e-4, checkpoint=1000, beta_1=0.99, beta_2=0.999)),

        #"Nonneg-DAGMA(pgd)": (NonnegDAGMAAdapter(primal_opt='pgd', acyclicity='logdet'), {},
        #                     dict(alpha=0.1, lamb=0.02, stepsize=1e-2, s=1.0, max_iters=2000,
        #                          checkpoint=250, tol=1e-6, Sigma=1, track_seq=False, verb=False)),

        #"MetMul-DAGMA": (MetMulDAGMAAdapter(), {},
        #                 dict(lamb=0.02, stepsize=1e-2, s=1.0, iters_in=1000, iters_out=8,
        #                      checkpoint=250, tol=1e-6, beta=5, gamma=0.25, rho_0=1, alpha_0=0.1,
        #                      track_seq=False, dec_step=None, Sigma=1, verb=False)),

         "TopoGreedy": (TopoGreedyAdapter(), {},
                        dict(k_list=[50], threshold_list=[5e-4], topo_thr=edge_thr_eval)),

         "TopoGreedy_refresh(0.2)": (TopoGreedyRefreshAdapter(), {},
                                     dict(k_list=[50], threshold_list=[5e-4],
                                          topo_thr=edge_thr_eval, refresh_every=0.2)),
    }

    # Optional: enable adaptive lambda for some methods (uses get_lamb_value)
    # baselines["NOTEARS(l2)"][2]["_adapt_lambda"] = True
    # baselines["NOTEARS(l2)"][2]["lambda1"] = 0.5   # multiplier for get_lamb_value

    out_dir = "./sem_results"
    run_experiment_suite(
        out_dir=out_dir,
        seeds=seeds,
        N_samples=N_samples,
        data_p=data_p,
        baselines=baselines,
        edge_thr_eval=edge_thr_eval,
        standarize_X=standarize_X
    )