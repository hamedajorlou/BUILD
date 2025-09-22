from __future__ import annotations
from typing import List, Dict
import numpy.linalg as la
import dag_utils as dagu
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
import numpy as np
import os, json, math, uuid, shutil, datetime as dt
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score
# from Baselines import GOLEM_Torch
import matplotlib.pyplot as plt



def to_bin(A, thr):
    B = A.copy().astype(float)
    B[np.abs(B) < thr] = 0.0
    B[B != 0.0] = 1.0
    np.fill_diagonal(B, 0.0)
    return B

def get_lambda_value(n_nodes: int, n_samples: int, times: float = 1.0) -> float:
    """Common λ heuristic: sqrt(log p / n) scaled by `times`."""
    return math.sqrt(max(1e-12, np.log(max(2, n_nodes))) / max(2, n_samples)) * times


def count_accuracy(W_bin_true, W_bin_est):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1},
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    pred_und = np.flatnonzero(W_bin_est == -1)
    pred = np.flatnonzero(W_bin_est == 1)
    cond = np.flatnonzero(W_bin_true)
    cond_reversed = np.flatnonzero(W_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # Compute SHD
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(W_bin_est + W_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(W_bin_true + W_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    # Compute TPR
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    tpr = float(len(true_pos)) / max(len(cond), 1)

    # Compute FDR
    pred_size = len(pred) + len(pred_und)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)

    return shd, tpr, fdr
    # return {"shd": shd, "tpr": tpr, "fdr": fdr}



import numpy.linalg as la

def compute_norm_sq_err(W_true, W_est, norm_W_true):
    """
    Compute Normalized Mean Squared Error (NMSE) between two matrices:
        NMSE = ||W_true - W_est||_F^2 / ||W_true||_F^2
    """

    return la.norm(W_true - W_est, 'fro')**2 / norm_W_true**2


def compute_norm_sq_err_old(W_true, W_est, norm_W_true=None):
    norm_W_true = norm_W_true if norm_W_true is not None else la.norm(W_true)
    norm_W_est = la.norm(W_est) if la.norm(W_est) > 0 else 1
    return (la.norm(W_true/norm_W_true - W_est/norm_W_est))**2


def f1_score(W_true, W_est):
    return 2 * (W_true * W_est).sum() / (W_true.sum() + W_est.sum())




def plot_and_save_results(out_dir: str, npz_filename: str = None):
    """
    Load experiment results and create performance plots and summary tables.
    
    Args:
        out_dir: Directory containing the results
        npz_filename: Specific npz file to load. If None, uses the most recent one.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os, sys
    import time
    import pandas as pd
    
    # Find the results file
    if npz_filename is None:
        npz_files = [f for f in os.listdir(out_dir) if f.startswith("sem_experiment_") and f.endswith(".npz")]
        if not npz_files:
            print("No results files found!")
            return
        latest_npz = max(npz_files, key=lambda x: os.path.getctime(os.path.join(out_dir, x)))
        npz_path = os.path.join(out_dir, latest_npz)
    else:
        npz_path = os.path.join(out_dir, npz_filename)
    
    # Load results
    data = np.load(npz_path)
    baselines = data['baselines']
    seeds = data['seeds']
    N_samples = data['N_samples']
    shd = data['shd']
    tpr = data['tpr']
    fdr = data['fdr']
    rel_err = data['rel_err']  # This is the normalized frobenius norm
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SEM Experiment Results', fontsize=16, fontweight='bold')
    
    metrics = [
        (shd, 'Structural Hamming Distance (SHD)', 'lower is better'),
        (tpr, 'True Positive Rate (TPR)', 'higher is better'),
        (fdr, 'False Discovery Rate (FDR)', 'lower is better'),
        (rel_err, 'Normalized Frobenius Norm Error', 'lower is better')
    ]
    
    for idx, (metric_data, title, direction) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Compute mean and std across seeds for each method and sample size
        mean_metric = np.nanmean(metric_data, axis=0)  # Average over seeds
        std_metric = np.nanstd(metric_data, axis=0)    # Std over seeds
        
        # Plot lines for each baseline
        for bi, baseline in enumerate(baselines):
            baseline_str = baseline.decode('utf-8') if isinstance(baseline, bytes) else str(baseline)
            ax.errorbar(N_samples, mean_metric[:, bi], yerr=std_metric[:, bi], 
                        marker='o', label=baseline_str, linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n({direction})')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Log scale for x-axis if sample sizes vary significantly
        if np.max(N_samples) / np.min(N_samples) > 10:
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(out_dir, f"performance_plots_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance plots saved to: {plot_path}")
    
    # Create a summary table
    summary_data = []
    for bi, baseline in enumerate(baselines):
        baseline_str = baseline.decode('utf-8') if isinstance(baseline, bytes) else str(baseline)
        
        # Get metrics for the largest sample size (last column)
        last_idx = -1
        summary_data.append({
            'Method': baseline_str,
            'SHD (mean±std)': f"{np.nanmean(shd[:, last_idx, bi]):.3f}±{np.nanstd(shd[:, last_idx, bi]):.3f}",
            'TPR (mean±std)': f"{np.nanmean(tpr[:, last_idx, bi]):.3f}±{np.nanstd(tpr[:, last_idx, bi]):.3f}",
            'FDR (mean±std)': f"{np.nanmean(fdr[:, last_idx, bi]):.3f}±{np.nanstd(fdr[:, last_idx, bi]):.3f}",
            'Norm Error (mean±std)': f"{np.nanmean(rel_err[:, last_idx, bi]):.3f}±{np.nanstd(rel_err[:, last_idx, bi]):.3f}"
        })
    
    # Save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(out_dir, f"summary_table_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    # Display summary
    print("\nSummary Results (for largest sample size):")
    print(summary_df.to_string(index=False))
    
    plt.show()


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
        return 0.0 if dagu.is_dag(A_bin) else 1.0
    except Exception:
        return float('nan')


def standarize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)






import numpy as np
from typing import Dict, Set, Any, Optional

def parents_dict_to_adj(
    parents: Dict[Any, Set[Any]],
    n: Optional[int] = None,
    dtype: type = float,
    check_bounds: bool = True
) -> np.ndarray:
    """
    Convert a mapping {child: {parent_i, ...}} to an adjacency matrix A
    with convention A[parent, child] = 1.

    Args:
        parents: dict mapping child -> set of parents (ints or np.integer).
        n: optional number of nodes; if None inferred from keys and parent entries.
        dtype: numpy dtype for the returned matrix (default float).
        check_bounds: if True, raises if any index is outside [0, n-1].

    Returns:
        A: (n x n) numpy array adjacency (binary 0/1).
    """
    # collect all node ids appearing as child or parent
    child_ids = list(parents.keys())
    parent_ids = [p for s in parents.values() for p in s]
    all_ids = set(int(x) for x in child_ids) | set(int(x) for x in parent_ids)

    if n is None:
        # infer n as max id + 1 (assumes nodes are 0..n-1)
        n = (max(all_ids) + 1) if all_ids else 0

    A = np.zeros((n, n), dtype=dtype)

    for child, prts in parents.items():
        c = int(child)
        if check_bounds and not (0 <= c < n):
            raise ValueError(f"Child index {c} out of bounds for n={n}.")
        for p in prts:
            pi = int(p)
            if check_bounds and not (0 <= pi < n):
                raise ValueError(f"Parent index {pi} out of bounds for n={n}.")
            A[pi, c] = 1

    return A



def _to_jsonable(obj):
    """Recursively convert objects to JSON-serializable forms."""
    import numpy as np
    from pathlib import Path

    # basic types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        # cast numpy scalars to Python
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    # numpy arrays -> lists (careful for huge arrays: we won't dump arrays here)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # tuples -> lists
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]

    # lists
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]

    # dicts
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # dataclasses
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(obj):
            return _to_jsonable(asdict(obj))
    except Exception:
        pass

    # callables (functions/classes)
    if callable(obj):
        # try to capture module + name; fall back to repr
        name = getattr(obj, "__name__", obj.__class__.__name__)
        mod  = getattr(obj, "__module__", None)
        return f"{mod+'.' if mod else ''}{name}"

    # objects with __dict__
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))

    # fallback
    return repr(obj)


@dataclass

# @dataclass
class BaselineSpec:
    """
    Unified baseline configuration.

    - model: either a *callable estimator* with .fit() method (class) OR a function (e.g., notears_linear)
    - init: kwargs passed to class constructor (ignored if model is a function)
    - args: kwargs passed to .fit(...) or function call
    - name: label for legends / saving
    - standardize: if True, pass standardized X to the model; else pass raw X
    - adapt_lambda: if True, rescale 'lamb' or 'lambda1' in args per (p, n)
    - topo_transpose: if True, transpose W_est and binarized variant to align with your downstream expectations
                      (e.g., TopoGreedy variants that output transposed)
    - is_topogreedy_refresh: if True, expects dict output with keys {"prec","A_est","A_est_bin"}; handled specially
    """
    model: Any
    init: Dict[str, Any] = field(default_factory=dict)
    args: Dict[str, Any] = field(default_factory=dict)
    name: str = "baseline"
    standardize: bool = False
    adapt_lambda: bool = False
    topo_transpose: bool = False
    is_topogreedy_refresh: bool = False



class ExperimentConfig:
    n_graphs: int
    n_nodes: int
    n_samples_list: List[int]
    edge_threshold: float
    data_params: Dict[str, Any]
    baselines: List[BaselineSpec]
    out_dir: str = "./exp_results"
    run_tag: Optional[str] = None
    save_intermediate: bool = True
    seed_offset: int = 0  # to vary graph seeds across runs



# ---------------------------------------------
# Plotting helpers
# ---------------------------------------------
def plot_summary_curves(final: Dict[str, np.ndarray],
                        cfg: ExperimentConfig,
                        out_dir: Path,
                        baselines: List[BaselineSpec]):
    """
    final: output of runner.run(); contains arrays (G,S,B,...)
    Plots mean ± s.e.m. across G graphs, vs n_samples, for each baseline.
    Also adds shaded regions for 10th-90th percentiles.
    Saves PNGs to out_dir.
    """
    n_samples = np.array(cfg.n_samples_list)
    G = final["shd"].shape[0]
    B = final["shd"].shape[2]

    # --- means & standard errors over graphs (axis=0) ---
    shd_mean = final["shd"].mean(axis=0)                        # (S, B)
    shd_sem  = final["shd"].std(axis=0, ddof=1) / np.sqrt(G)    # (S, B)
    shd_p10  = np.percentile(final["shd"], 10, axis=0)          # (S, B)
    shd_p90  = np.percentile(final["shd"], 90, axis=0)          # (S, B)

    err_mean = final["err"].mean(axis=0)                        # (S, B)
    err_sem  = final["err"].std(axis=0, ddof=1) / np.sqrt(G)    # (S, B)
    err_p10  = np.percentile(final["err"], 10, axis=0)          # (S, B)
    err_p90  = np.percentile(final["err"], 90, axis=0)          # (S, B)

    # --- SHD vs #samples ---
    plt.figure()
    for bi in range(B):
        plt.errorbar(
            n_samples, shd_mean[:, bi], yerr=shd_sem[:, bi],
            marker='o', capsize=4, linewidth=2, label=baselines[bi].name
        )
        plt.fill_between(
            n_samples, shd_p10[:, bi], shd_p90[:, bi],
            alpha=0.2
        )
    plt.xlabel("Number of samples (n)")
    plt.ylabel("SHD (mean ± s.e.m. over graphs)")
    plt.title("SHD vs #Samples")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "shd_vs_samples.png", dpi=200)

    # --- Normalized MSE (weights) vs #samples ---
    plt.figure()
    for bi in range(B):
        plt.errorbar(
            n_samples, err_mean[:, bi], yerr=err_sem[:, bi],
            marker='o', capsize=4, linewidth=2, label=baselines[bi].name
        )
        plt.fill_between(
            n_samples, err_p10[:, bi], err_p90[:, bi],
            alpha=0.2
        )
    plt.xlabel("Number of samples (n)")
    plt.ylabel("Normalized MSE of weights (mean ± s.e.m.)")
    plt.title("Normalized MSE (weights) vs #Samples")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nmse_vs_samples.png", dpi=200)

    print(f"Saved plots to: {out_dir}")


import re

def _sanitize_baseline_name_for_csv(name: str) -> str:
    """Turn 'TopoGreedy-0.02' -> 'TopoGreedy_0_02' (safe CSV column header)."""
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')

def save_summary_csvs(final: Dict[str, np.ndarray],
                      cfg: ExperimentConfig,
                      out_dir: Path,
                      baselines: List[BaselineSpec]) -> Tuple[Path, Path, Path]:
    """
    Writes three CSVs into out_dir:
      - shd_vs_samples.csv:   n_samples + (mean, sem, p10, p90) per baseline
      - nmse_vs_samples.csv:  n_samples + (mean, sem, p10, p90) per baseline
      - summary_vs_samples.csv: wide table with both metrics interleaved
    Returns the three paths.
    """
    import csv
    n_samples = np.asarray(cfg.n_samples_list)
    G = int(final["shd"].shape[0])
    B = int(final["shd"].shape[2])

    # Means, SEM, and percentiles across graphs
    def _sem(arr):
        return (arr.std(axis=0, ddof=1) / np.sqrt(G)) if G > 1 else np.zeros_like(arr.mean(axis=0))

    shd_mean = final["shd"].mean(axis=0)   # (S,B)
    shd_sem  = _sem(final["shd"])          # (S,B)
    shd_p10  = np.percentile(final["shd"], 10, axis=0)  # (S,B)
    shd_p90  = np.percentile(final["shd"], 90, axis=0)  # (S,B)
    
    nmse_mean = final["err"].mean(axis=0)  # (S,B) normalized MSE of weights
    nmse_sem  = _sem(final["err"])         # (S,B)
    nmse_p10  = np.percentile(final["err"], 10, axis=0)  # (S,B)
    nmse_p90  = np.percentile(final["err"], 90, axis=0)  # (S,B)

    # Make header labels safe for CSV/pgfplots
    safe_names = [_sanitize_baseline_name_for_csv(b.name) for b in baselines]
    orig_names = [b.name for b in baselines]  # for reference/legends

    # 1) SHD CSV
    shd_csv = out_dir / "shd_vs_samples.csv"
    with open(shd_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["n_samples"]
        for sn in safe_names:
            header += [f"{sn}_mean", f"{sn}_sem", f"{sn}_p10", f"{sn}_p90"]
        w.writerow(header)
        for si, n in enumerate(n_samples):
            row = [int(n)]
            for bi in range(B):
                row += [float(shd_mean[si, bi]), float(shd_sem[si, bi]),
                        float(shd_p10[si, bi]), float(shd_p90[si, bi])]
            w.writerow(row)

    # 2) NMSE CSV
    nmse_csv = out_dir / "nmse_vs_samples.csv"
    with open(nmse_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["n_samples"]
        for sn in safe_names:
            header += [f"{sn}_mean", f"{sn}_sem", f"{sn}_p10", f"{sn}_p90"]
        w.writerow(header)
        for si, n in enumerate(n_samples):
            row = [int(n)]
            for bi in range(B):
                row += [float(nmse_mean[si, bi]), float(nmse_sem[si, bi]),
                        float(nmse_p10[si, bi]), float(nmse_p90[si, bi])]
            w.writerow(row)

    # 3) Wide summary CSV (both metrics interleaved)
    summary_csv = out_dir / "summary_vs_samples.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["n_samples"]
        for sn in safe_names:
            header += [f"{sn}_SHD_mean", f"{sn}_SHD_sem", f"{sn}_SHD_p10", f"{sn}_SHD_p90",
                       f"{sn}_NMSE_mean", f"{sn}_NMSE_sem", f"{sn}_NMSE_p10", f"{sn}_NMSE_p90"]
        w.writerow(header)
        for si, n in enumerate(n_samples):
            row = [int(n)]
            for bi in range(B):
                row += [float(shd_mean[si, bi]),  float(shd_sem[si, bi]),
                        float(shd_p10[si, bi]),   float(shd_p90[si, bi]),
                        float(nmse_mean[si, bi]), float(nmse_sem[si, bi]),
                        float(nmse_p10[si, bi]),  float(nmse_p90[si, bi])]
            w.writerow(row)

    print("Saved CSVs:", shd_csv, nmse_csv, summary_csv, sep="\n- ")
    return shd_csv, nmse_csv, summary_csv


import pandas as pd

def reshape_for_overleaf(input_csv: str, output_csv: str):
    # Load the original CSV
    df = pd.read_csv(input_csv)

    # Pivot the data: group by Samples, expand Algorithm into columns
    error_pivot = df.pivot(index="Samples", columns="Algorithm", values="Error_Mean")
    error_p10 = df.pivot(index="Samples", columns="Algorithm", values="Error_P10")
    error_p90 = df.pivot(index="Samples", columns="Algorithm", values="Error_P90")

    time_pivot = df.pivot(index="Samples", columns="Algorithm", values="Time_Mean")
    time_p10 = df.pivot(index="Samples", columns="Algorithm", values="Time_P10")
    time_p90 = df.pivot(index="Samples", columns="Algorithm", values="Time_P90")

    # Flatten column names to match pgfplots style, e.g., "GAO_ErrorMean"
    error_pivot.columns = [f"{alg}_ErrorMean" for alg in error_pivot.columns]
    error_p10.columns   = [f"{alg}_ErrorP10" for alg in error_p10.columns]
    error_p90.columns   = [f"{alg}_ErrorP90" for alg in error_p90.columns]

    time_pivot.columns  = [f"{alg}_TimeMean" for alg in time_pivot.columns]
    time_p10.columns    = [f"{alg}_TimeP10" for alg in time_p10.columns]
    time_p90.columns    = [f"{alg}_TimeP90" for alg in time_p90.columns]

    # Merge all into one wide dataframe
    out_df = pd.concat([error_pivot, error_p10, error_p90,
                        time_pivot, time_p10, time_p90], axis=1).reset_index()

    # Save to CSV
    out_df.to_csv(output_csv, index=False)


import numpy as np
from typing import Dict, Set, List, Tuple

def extract_config_from_adj(
    adj: np.ndarray,
    *,
    convention: str = "row->col",  
    tol: float = 0.0,             
) -> Tuple[List[int], Dict[int, Dict[int, float]], Dict[int, Set[int]], np.ndarray]:
    """
    From a DAG adjacency (weighted or binary), return:
      order, coefficient, parents, B
    in the same format as generate_config.

    - convention="row->col": adj[i, j] != 0 means i -> j (PARENT i, CHILD j).
    - convention="col->row": adj[i, j] != 0 means j -> i.
    - tol: treat |adj| <= tol as zero.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix")

    A = np.array(adj, dtype=float, copy=True)
    if convention == "col->row":
        A = A.T  # make it behave like row->col internally

    n = A.shape[0]
    # Topological order via Kahn (validates acyclicity)
    indeg = np.sum(np.abs(A) > tol, axis=0).astype(int)
    order: List[int] = [i for i in range(n) if indeg[i] == 0]
    idx = 0
    while idx < len(order):
        u = order[idx]; idx += 1
        for v in np.where(np.abs(A[u, :]) > tol)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(int(v))
    if len(order) != n:
        raise ValueError("Input adjacency has a cycle (not a DAG).")

    # Parents and coefficient dicts
    parents: Dict[int, Set[int]] = {}
    coefficient: Dict[int, Dict[int, float]] = {}

    for child in range(n):
        prt_idx = np.where(np.abs(A[:, child]) > tol)[0]
        # Cast to plain int for clean sets (no np.int64 in repr)
        prt_set = {int(i) for i in prt_idx.tolist()}
        parents[child] = prt_set
        coefficient[child] = {int(i): float(A[i, child]) for i in prt_set}

    B = A.copy()
    return order, coefficient, parents, B
