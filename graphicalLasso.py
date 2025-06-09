
import numpy as np
from numpy import linalg as la
import networkx as nx
import time

def soft_threshold(x, threshold):
    """Apply soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def graphical_lasso(S, alpha, max_iter=1000, tol=1e-6, verbose=False):
    """
    Robust Graphical Lasso algorithm for sparse precision matrix estimation.
    
    Parameters:
    -----------
    S : array-like, shape (p, p)
        Sample covariance matrix
    alpha : float
        Regularization parameter (sparsity control)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print convergence information
        
    Returns:
    --------
    Theta : array, shape (p, p)
        Estimated precision matrix
    Sigma : array, shape (p, p)
        Estimated covariance matrix
    """
    p = S.shape[0]
    
    # Initialize with regularized sample covariance inverse
    # Add small regularization to ensure positive definiteness
    reg_S = S + alpha * np.eye(p)
    Theta = np.linalg.inv(reg_S)
    
    # Ensure initial matrix is well-conditioned
    eigenvals = np.linalg.eigvals(Theta)
    if np.min(eigenvals) <= 0:
        Theta = Theta + (1e-6 - np.min(eigenvals)) * np.eye(p)
    
    for iteration in range(max_iter):
        Theta_old = Theta.copy()
        
        # Block coordinate descent over each variable
        for j in range(p):
            # Partition matrices
            idx = np.arange(p) != j
            
            # S partitions
            S_11 = S[np.ix_(idx, idx)]
            s_12 = S[idx, j]
            s_22 = S[j, j]
            
            # Current Theta partitions  
            Theta_11 = Theta[np.ix_(idx, idx)]
            
            # Solve the regularized linear system for the j-th column/row
            # We solve: (S_11 + alpha * I) * w = s_12
            # where w will be related to the off-diagonal elements
            
            try:
                # Add regularization for numerical stability
                reg_S_11 = S_11 + alpha * np.eye(p-1)
                
                # Solve the lasso problem using coordinate descent
                w = np.zeros(p-1)
                
                # Initialize with least squares solution
                try:
                    w = np.linalg.solve(reg_S_11, s_12)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(reg_S_11, s_12, rcond=None)[0]
                
                # Coordinate descent for lasso regularization
                for lasso_iter in range(200):
                    w_old = w.copy()
                    
                    for k in range(p-1):
                        # Compute partial residual
                        residual = s_12[k] - np.dot(reg_S_11[k, :], w) + reg_S_11[k, k] * w[k]
                        
                        # Soft thresholding update
                        if reg_S_11[k, k] > 1e-12:
                            w[k] = soft_threshold(residual / reg_S_11[k, k], alpha / reg_S_11[k, k])
                        else:
                            w[k] = 0.0
                    
                    # Check inner convergence
                    if np.linalg.norm(w - w_old) < tol * 0.1:
                        break
                
                # Update the precision matrix
                # Compute diagonal element
                theta_jj = 1.0 / (s_22 - np.dot(s_12, w))
                
                # Ensure positive diagonal
                if theta_jj <= 0 or not np.isfinite(theta_jj):
                    theta_jj = 1.0 / (s_22 + alpha)
                
                # Update off-diagonal elements
                theta_j = -theta_jj * w
                
                # Set the j-th row and column
                Theta[j, j] = theta_jj
                Theta[j, idx] = theta_j
                Theta[idx, j] = theta_j
                
            except (np.linalg.LinAlgError, ValueError):
                # Fallback: keep previous values or use simple update
                if verbose:
                    print(f"Numerical issues at iteration {iteration}, variable {j}")
                continue
        
        # Ensure symmetry
        Theta = (Theta + Theta.T) / 2
        
        # Check for positive definiteness and fix if needed
        try:
            eigenvals = np.linalg.eigvals(Theta)
            min_eigval = np.min(eigenvals)
            if min_eigval <= 1e-12:
                Theta = Theta + (1e-6 - min_eigval) * np.eye(p)
        except:
            pass
        
        # Check convergence
        diff = np.linalg.norm(Theta - Theta_old, 'fro')
        if diff < tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations (diff: {diff:.2e})")
            break
        
        if verbose and (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}, diff: {diff:.2e}")
    
    else:
        if verbose:
            print(f"Maximum iterations ({max_iter}) reached")
    
    # Final symmetry enforcement
    Theta = (Theta + Theta.T) / 2
    
    # Compute covariance matrix
    try:
        # Check condition number before inversion
        cond_num = np.linalg.cond(Theta)
        if cond_num > 1e12:
            if verbose:
                print(f"Warning: Precision matrix is ill-conditioned (cond={cond_num:.2e})")
            Sigma = np.linalg.pinv(Theta)
        else:
            Sigma = np.linalg.inv(Theta)
    except np.linalg.LinAlgError:
        if verbose:
            print("Warning: Using pseudo-inverse for covariance matrix")
        Sigma = np.linalg.pinv(Theta)
    
    return Theta



def compute_Dq(dag: nx.DiGraph, target_node: str, only_diag: bool = True,
               verbose: bool = False, ordered: bool = False) -> np.ndarray:
    """
    Compute Dq, the frequency response matrix of the GSO associated with node q, based on the
    existence of paths from each node to the target node.

    Args:
        dag (nx.DiGraph): Directed acyclic graph (DAG).
        target_node (str): Target node identifier.
        only_diag (bool, optional): Whether to return only the diagonal of the matrix. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        np.ndarray: Frequency response matrix Dq.
    """
    N = dag.number_of_nodes()
    target_idx = ord(target_node) - ord('a') if isinstance(target_node, str) else target_node
    
    path_exists = np.zeros(N)
    max_node = target_idx + 1 if ordered else N 
    for i in range(max_node):
        path_exists[i] = nx.has_path(dag, i, target_idx)
    
    if verbose:
        for i, exists in enumerate(path_exists):
            print(f'Has path from node {i} to node {target_node}: {exists}')

    if only_diag:
        return path_exists
    else:
        return np.diag(path_exists)

def compute_GSOs(W, dag):
    N = W.shape[0]
    GSOs = np.array([W @ compute_Dq(dag, i, N) @ la.inv(W) for i in range(N)])
    return GSOs


def create_dag(N, p, weighted=True, weakly_conn=True, max_tries=25):
    """
    Create a random directed acyclic graph (DAG) with independent edge probability.

    Args:
        N (int): Number of nodes.
        p (float): Probability of edge creation.
        weighted (bool, optional): Whether to generate a weighted DAG. Defaults to True.
        weakly_conn (bool, optional): Whether to ensure weak connectivity. Defaults to True.
        max_tries (int, optional): Maximum number of attempts to generate a valid DAG. Defaults to 25.

    Returns:
        tuple[np.ndarray, nx.DiGraph]: Tuple containing the adjacency matrix and the DAG.
    """
    attempt = 0
    while attempt < max_tries:
        # Generate random directed graph
        graph = nx.erdos_renyi_graph(N, p, directed=True)
        adjacency_matrix = nx.to_numpy_array(graph)
        
        # Make it lower triangular to ensure DAG property
        adjacency_matrix = np.tril(adjacency_matrix, k=-1) 

        if weighted:
            # Create weight matrix with values in [0.5, 2]
            weight_values = np.random.uniform(low=0.5, high=2, size=(N, N))
            
            # Randomly assign positive or negative signs
            sign_choices = np.random.choice([-1, 1], size=(N, N), p=[0.5, 0.5])
            
            # Apply signs to weights (negative values become [-2, -0.5])
            weight_values = weight_values * sign_choices
            
            # Apply weights to adjacency matrix
            adjacency_matrix = adjacency_matrix * weight_values
            
            # Normalize columns by their sum
            column_sums = adjacency_matrix.sum(axis=0)
            nonzero_columns = column_sums != 0
            adjacency_matrix[:, nonzero_columns] /= column_sums[nonzero_columns]

        # Create DAG from adjacency matrix
        dag = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())

        # Check if DAG meets connectivity requirement
        if not weakly_conn or nx.is_weakly_connected(dag):
            assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
            return adjacency_matrix.T, dag
        
        attempt += 1
    
    # If we reach here, we couldn't generate a weakly connected DAG
    print('Generated Adjacency Matrix:', adjacency_matrix.T)
    print('WARNING: DAG is not weakly connected')
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
    return adjacency_matrix.T, dag





def create_dag_sf(N, m, weighted=True, weakly_conn=True, max_tries=25):
    """
    Create a random directed acyclic graph (DAG) with independent edge probability.

    Args:
        N (int): Number of nodes.
        p (float): Probability of edge creation.
        weighted (bool, optional): Whether to generate a weighted DAG. Defaults to True.

    Returns:
        tuple[np.ndarray, nx.DiGraph]: Tuple containing the adjacency matrix and the DAG.
    """
    for _ in range (max_tries):
        print('bar')
        graph = nx.barabasi_albert_graph(N, m)
        adj = nx.to_numpy_array(graph)
        Adj = np.tril(adj, k=-1)

        if weighted:
            Weights = np.random.uniform(low=0.5, high=2, size=(N, N))
            Signs = np.random.choice([-1, 1], size=(N, N), p=[0.5, 0.5])
            Weights = Weights * Signs
            Adj = Adj * Weights
            colums_sum = Adj.sum(axis=0)
            col_sums_nonzero = colums_sum[colums_sum != 0]
            Adj[:, colums_sum != 0] /= col_sums_nonzero

        dag = nx.from_numpy_array(Adj, create_using=nx.DiGraph())

        if not weakly_conn or nx.is_weakly_connected(dag):
            assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
            return Adj, dag
    
    print('WARING: dag is not weakly connected')
    assert nx.is_directed_acyclic_graph(dag), "Graph is not a DAG"
    return Adj.T, dag





def get_graph_data(d_dat_p, get_Psi=False, sf=False):
    if not sf:
        Adj, dag = create_dag(d_dat_p['N'], d_dat_p['p'])
    else:
        m = float(d_dat_p['p'] * ((d_dat_p['N']*(d_dat_p['N']-1))/2))
        sf_m = int(round(m / d_dat_p['N']))
        Adj, dag = create_dag_sf(d_dat_p['N'], sf_m)

    W = la.inv(np.eye(d_dat_p['N']) - Adj)
    W_inv = la.inv(W)

    if get_Psi:
        Psi = np.array([compute_Dq(dag, i) for i in range(d_dat_p['N'])]).T
        GSOs = np.array([(W * Psi[:,i]) @ W_inv for i in range(d_dat_p['N'])])
        return Adj, W, GSOs, Psi
    
    GSOs = np.array([(W * compute_Dq(dag, i)) @ W_inv for i in range(d_dat_p['N'])])
    return Adj, W, GSOs



def generate_sparse_precision_matrix(N, p, seed=42):
    """Generate a sparse positive definite precision matrix for testing using DAG structure."""

    np.random.seed(seed)
    
    # Generate a weighted DAG level by level with children first, roots last
    A = np.zeros((N, N))
    
    # Create levels - children nodes have lower indices, root nodes have higher indices
    # This means edges go from higher indices to lower indices (opposite of typical topological order)
    for i in range(N):
        for j in range(i):  # j < i, so edges go from higher index to lower index
            if np.random.random() < p:
                # Generate random weight in [0.5, 2] with random sign
                weight = np.random.uniform(0.5, 2.0)
                sign = np.random.choice([-1, 1])
                A[i, j] = weight * sign
    
    # Normalize columns to ensure stability
    for j in range(N):
        col_sum = np.sum(np.abs(A[:, j]))
        if col_sum > 0:
            A[:, j] = A[:, j] / col_sum
    
    A = A.T
    # Create DAG from adjacency matrix for verification
    dag = nx.from_numpy_array(A, create_using=nx.DiGraph())
    
    # Verify it's a DAG
    assert nx.is_directed_acyclic_graph(dag), "Generated graph is not a DAG"
    
    # Generate precision matrix from the adjacency matrix using DAG structure
    # Use (I - A)^T (I - A) for DAG structure
    I = np.eye(N)
    Theta = (I - A).T @ (I - A)
    
    # Add small regularization to ensure positive definiteness
    min_eigenval = np.min(np.linalg.eigvals(Theta))
    if min_eigenval <= 1e-8:
        Theta += (1e-6 - min_eigenval) * I
    
    # Verify positive definiteness
    try:
        np.linalg.cholesky(Theta)
        print("Generated precision matrix is positive definite")
    except np.linalg.LinAlgError:
        print("Warning: Generated precision matrix may not be positive definite")
    

    return Theta, A

