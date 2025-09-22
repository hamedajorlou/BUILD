import numpy as np
from numpy import linalg as la
import networkx as nx
import time
from pandas import DataFrame
from IPython.display import display
from scipy.special import expit as sigmoid
import igraph as ig
import random
import typing
import matplotlib.pyplot as plt



def is_dag(W):
    return nx.is_directed_acyclic_graph(nx.DiGraph(W))

def create_dag(n_nodes, graph_type, edges, permute=True, edge_type='positive', w_range=(.5, 1.5),
               rew_prob=.1):
    """
    edge_type cana be binary, positive, or negative 
    """    
    if 'er' in graph_type:
        prob = float(edges*2)/float(n_nodes**2 - n_nodes)
        G = nx.erdos_renyi_graph(n_nodes, prob)
        W = np.triu(nx.to_numpy_array(G), k=1)

    elif graph_type == 'sf' or graph_type == 'sf_t':
        sf_m = int(round(edges / n_nodes))
        G = nx.barabasi_albert_graph(n_nodes, sf_m)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sf' else np.tril(adj, k=-1)

    elif graph_type == 'sw' or graph_type == 'sw_t':
        G = nx.watts_strogatz_graph(n_nodes, int(2*round(edges/n_nodes)), rew_prob)
        adj = nx.to_numpy_array(G)
        W = np.triu(adj, k=1) if graph_type == 'sw' else np.tril(adj, k=-1)

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

    # Perform X = Z(I-A)^-1 sequentially to increase speed
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


def simulate_sem(n_nodes, n_samples, graph_type, edges, permute=True, edge_type='positive',
                 w_range=(.5, 1.5), noise_type='normal', var=1):
    A, dag = create_dag(n_nodes, graph_type, edges, permute, edge_type, w_range)
    X = create_sem_signals(n_nodes, n_samples, dag, noise_type, var)
    return A, dag, X

def to_bin(W, thr=0.1):
    W_bin = np.copy(W)
    W_bin[np.abs(W_bin) < thr] = 0
    W_bin[np.abs(W_bin) >= thr] = 1

    return W_bin

def compute_norm_sq_err(W_true, W_est, norm_W_true=None):
    norm_W_true = norm_W_true if norm_W_true is not None else la.norm(W_true)
    norm_W_est = la.norm(W_est) if la.norm(W_est) > 0 else 1
    return (la.norm(W_true/norm_W_true - W_est/norm_W_est))**2

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

def display_results(exps_leg, metrics, agg='mean', file_name=None):
    
    metric_str = {'leg': exps_leg}
    for key, value in metrics.items():
        metric_str[key] = []
        
        agg_metric = np.median(value, axis=0) if agg == 'median' else np.mean(value, axis=0)
        std_metric = np.std(value, axis=0)
        for i, _ in enumerate(exps_leg):
            text = f'{agg_metric[i]:.4f}  \u00B1 {std_metric[i]:.4f}'
            metric_str[key].append(text)
        
    df = DataFrame(metric_str)
    display(df)

    if file_name:
        df.to_csv(f'{file_name}.csv', index=False)
        print(f'DataFrame saved to {file_name}.csv')

def standarize(X):
    return (X - X.mean(axis=0))/X.std(axis=0)

def plot_data(axes, data, exps, x_vals, xlabel, ylabel, skip_idx=[], agg='mean', deviation=None,
              alpha=.25, plot_func='semilogx'):
    if agg == 'median':
        agg_data = np.median(data, axis=0)
    else:
        agg_data = np.mean(data, axis=0)

    std = np.std(data, axis=0)
    prctile25 = np.percentile(data, 25, axis=0)
    prctile75 = np.percentile(data, 75, axis=0)

    for i, exp in enumerate(exps):
        if i in skip_idx:
            continue
        getattr(axes, plot_func)(x_vals, agg_data[:,i], exp['fmt'], label=exp['leg'])

        if deviation == 'prctile':
            up_ci = prctile25[:,i]
            low_ci = prctile75[:,i]
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)
        elif deviation == 'std':
            up_ci = agg_data[:,i] + std[:,i]
            low_ci = np.maximum(agg_data[:,i] - std[:,i], 0)
            axes.fill_between(x_vals, low_ci, up_ci, alpha=alpha)

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True)
    axes.legend()

def plot_all_metrics(shd, tpr, fdr, fscore, err, acyc, runtime, dag_count, x_vals, exps, 
                     agg='mean', skip_idx=[], dev=False, alpha=.25, xlabel='Number of samples'):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], shd, exps, x_vals, xlabel, 'SDH', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[1], tpr, exps, x_vals, xlabel, 'TPR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[2], fdr, exps, x_vals, xlabel, 'FDR', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plot_data(axes[3], fscore, exps, x_vals, xlabel, 'F1', skip_idx,
              agg=agg, deviation=dev, alpha=alpha)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_data(axes[0], err, exps, x_vals, xlabel, 'Fro Error', skip_idx, agg=agg,
              deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[1], acyc, exps, x_vals, xlabel, 'Acyclity', skip_idx, agg=agg,
              deviation=dev)
    plot_data(axes[2], runtime, exps, x_vals, xlabel, 'Running time (seconds)',
              skip_idx, agg=agg, deviation=dev, alpha=alpha, plot_func='loglog')
    plot_data(axes[3], dag_count, exps, x_vals, xlabel, 'Graph is DAG', skip_idx,
              agg=agg)
    plt.tight_layout()


def data_to_csv(fname, models, xaxis, error, agg='mean', dev='std'):
    data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error), axis=1)
    header = 'xaxis; '  

    for i, model in enumerate(models):
        header += model['leg']
        if i < len(models)-1:
            header += '; '

    np.savetxt(fname, data, delimiter=';', header=header, comments='')
    print('SAVED as:', fname)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d: int, s0: int, graph_type: str) -> np.ndarray:
    r"""
    Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    d : int
        num of nodes
    s0 : int
        expected num of edges
    graph_type : str
        One of ``["ER", "SF", "BP"]``
    
    Returns
    -------
    numpy.ndarray
        :math:`(d, d)` binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == 'Fully':
        B = np.triu(np.ones((d,d)), 1)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B: np.ndarray, 
                       w_ranges: typing.List[typing.Tuple[float,float]]=((-2.0, -0.5), (0.5, 2.0)),
                       ) -> np.ndarray:
    r"""
    Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    w_ranges : typing.List[typing.Tuple[float,float]], optional
        List of tuples specifying disjoint weight ranges. Each tuple contains (min, max) values.
        For example, ((-2.0, -0.5), (0.5, 2.0)) means weights will be sampled from either
        [-2.0, -0.5] or [0.5, 2.0]. Default is ((-2.0, -0.5), (0.5, 2.0)).

    Returns
    -------
    np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    
    I = np.eye(B.shape[0])
    Theta = (I - W.T).T @ (I - W.T)

    return W, Theta

def simulate_linear_sem(W: np.ndarray, 
                        n: int, 
                        sem_type: str, 
                        noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                        ) -> np.ndarray:
    r"""
    Simulate samples from linear SEM with specified type of noise.
    For ``uniform``, noise :math:`z \sim \mathrm{uniform}(-a, a)`, where :math:`a` is the ``noise_scale``.
    
    Parameters
    ----------
    W : np.ndarray
        :math:`[d, d]` weighted adj matrix of DAG.
    n : int
        num of samples. When ``n=inf`` mimics the population risk, only for Gaussian noise.
    sem_type : str
        ``gauss``, ``exp``, ``gumbel``, ``uniform``, ``logistic``, ``poisson``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix, :math:`[d, d]` if ``n=inf``.
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or have length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B:np.ndarray, 
                           n: int, 
                           sem_type: str, 
                           noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                           ) -> np.ndarray:
    r"""
    Simulate samples from nonlinear SEM.

    Parameters
    ----------
    B : np.ndarray
        :math:`[d, d]` binary adj matrix of DAG.
    n : int
        num of samples
    sem_type : str
        ``mlp``, ``mim``, ``gp``, ``gp-add``
    noise_scale : typing.Optional[typing.Union[float,typing.List[float]]], optional
        scale parameter of the additive noises. If ``None``, all noises have scale 1. Default: ``None``.

    Returns
    -------
    np.ndarray
        :math:`[n, d]` sample matrix.
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    r"""
    Compute various accuracy metrics for B_est.

    | true positive = predicted association exists in condition in correct direction
    | reverse = predicted association exists in condition in opposite direction
    | false positive = predicted association does not exist in condition
    
    Parameters
    ----------
    B_true : np.ndarray
        :math:`[d, d]` ground truth graph, :math:`\{0, 1\}`.
    B_est : np.ndarray
        :math:`[d, d]` estimate, :math:`\{0, 1, -1\}`, -1 is undirected edge in CPDAG.

    Returns
    -------
    dict
        | fdr: (reverse + false positive) / prediction positive
        | tpr: (true positive) / condition positive
        | fpr: (reverse + false positive) / condition negative
        | shd: undirected extra + undirected missing + reverse
        | nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}



