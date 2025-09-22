import numpy as np
import networkx as nx
from numpy.linalg import norm, LinAlgError
from scipy.linalg import pinv
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------
# DAG generation
# ---------------------------------------------


def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def create_dag(n_nodes, graph_type, edges, permute=False, edge_type='positive', w_range=(0, 0.5),
               rew_prob=.1):
    """
    edge_type can be binary, positive, or weighted
    """
    if 'er' in graph_type:
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.tril(nx.to_numpy_array(G), k=-1)

    elif graph_type == 'sf' or graph_type == 'sf_t':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        adj = nx.to_numpy_array(G)
        W = np.tril(adj, k=-1)

    else:
        raise ValueError('Unknown graph type')

    assert nx.is_weighted(G) == False
    assert nx.is_empty(G) == False

    if permute:
        P = np.eye(n_nodes)
        P = P[:, np.random.permutation(n_nodes)]
        W = P @ W @ P.T

    if edge_type == 'binary':
        W_weighted = W.copy()
    elif edge_type == 'positive':
        weights = np.random.uniform(w_range[0], w_range[1], size=W.shape)
        W_weighted = weights * W
    elif edge_type == 'weighted':
        # Default range: w_range=((-2.0, -0.5), (0.5, 2.0))
        W_weighted = np.zeros(W.shape)
        S = np.random.randint(len(w_range), size=W.shape)
        for i, (low, high) in enumerate(w_range):
            weights = np.random.uniform(low=low, high=high, size=W.shape)
            W_weighted += W * (S == i) * weights
    else:
        raise ValueError('Unknown edge type')

    dag = nx.DiGraph(W_weighted)
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"

    return W_weighted, nx.DiGraph(dag)

def create_sem_signals(n_nodes, n_samples, G, noise_type='normal', var=1):
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == n_nodes

    X = np.zeros((n_samples, n_nodes))

    W_weighted = nx.to_numpy_array(G)

    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W_weighted[parents, j])
        if noise_type == 'normal':
            scale = np.sqrt(var)
            X[:, j] = eta + np.random.normal(scale=scale, size=(n_samples))
        elif noise_type == 'exp':
            scale = np.sqrt(var)
            X[:, j] = eta + np.random.exponential(scale=scale, size=(n_samples))
        elif noise_type == 'laplace':
            scale = np.sqrt(var / 2.0)
            X[:, j] = eta + np.random.laplace(loc=0.0, scale=scale, size=(n_samples))
        elif noise_type == 'gumbel':
            scale = np.sqrt(6.0 * var) / np.pi
            X[:, j] = eta + np.random.gumbel(loc=0.0, scale=scale, size=(n_samples))
        else:
            raise ValueError('Noise type error!')

    return X

def simulate_sem(n_nodes, n_samples, graph_type, edges, permute=False, edge_type='positive',
                 w_range=(0, 0.5), noise_type='normal', var=1):
    A, dag = create_dag(n_nodes, graph_type, edges, permute, edge_type, w_range)
    X = create_sem_signals(n_nodes, n_samples, dag, noise_type, var)
    I = np.eye(n_nodes)
    Theta = 1/var * (I - A) @ (I - A).T
    return A, dag, X, Theta



def standarize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)


def to_bin(A, thr):
    B = A.copy().astype(float)
    B[np.abs(B) < thr] = 0.0
    B[B != 0.0] = 1.0
    np.fill_diagonal(B, 0.0)
    return B





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

def compute_norm_sq_err(W_true, W_est, norm_W_true):
    return np.linalg.norm(W_est - W_true, 'fro') / (norm_W_true + 1e-8)