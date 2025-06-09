import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import numpy.linalg as la
from utils import compute_lambda
from Graphical_Lasso_impl import org_graphical_lasso, graphical_lasso_TB, soft_threshold, prox_grad_step_, FairGLASSO_fista


def estimate_adjacency_matrix(theta_matrix, p, edge_threshold=0.2):
    """
    Estimate adjacency matrix A from precision matrix using iterative algorithm.
    
    Args:
        theta_matrix: Precision matrix to process
        p: Number of nodes
        edge_threshold: Threshold for removing weak edges
    
    Returns:
        A_est: Estimated adjacency matrix
    """
    A_est = np.zeros((p, p))
    Theta_aux = theta_matrix.copy()
    
    def process_leaf_nodes(theta_aux, a_est):
        """Process one iteration of leaf node identification and removal."""
        diag_vals = np.diag(theta_aux)
        idx_leaf_nodes = np.where(np.isclose(diag_vals, 1.0, atol=2e-1))[0]
        
        if len(idx_leaf_nodes) == 0:
            return theta_aux, a_est, False  # No more leaf nodes found
        
        for i in idx_leaf_nodes:
            theta_aux[i, i] = 0
            a_i = -theta_aux[i, :]
            a_est[i, :] = a_i
            
            parents = np.where(np.abs(a_i) > 1e-6)[0]
            
            theta_aux[:, i] += a_i
            theta_aux[i, :] += a_i
            for j in parents:
                theta_aux[j, :] -= a_i[j] * a_i
        
        return theta_aux, a_est, True  # Continue processing
    
    # Iteratively process leaf nodes
    for _ in range(p):
        Theta_aux, A_est, should_continue = process_leaf_nodes(Theta_aux, A_est)
        if not should_continue:
            break
    
    # Remove weak edges
    A_est[np.abs(A_est) < edge_threshold] = 0
    return A_est.T


def soft_threshold(x, threshold):
    """Apply soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)



# def graphical_lasso(S, alpha, max_iter=1000, tol=1e-6, verbose=False):
#     """
#     Robust Graphical Lasso algorithm for sparse precision matrix estimation.
    
#     Parameters:
#     -----------
#     S : array-like, shape (p, p)
#         Sample covariance matrix
#     alpha : float
#         Regularization parameter (sparsity control)
#     max_iter : int
#         Maximum number of iterations
#     tol : float
#         Convergence tolerance
#     verbose : bool
#         Print convergence information
        
#     Returns:
#     --------
#     Theta : array, shape (p, p)
#         Estimated precision matrix
#     Sigma : array, shape (p, p)
#         Estimated covariance matrix
#     """
#     p = S.shape[0]
    
#     # Initialize with regularized sample covariance inverse
#     # Add small regularization to ensure positive definiteness
#     reg_S = S + alpha * np.eye(p)
#     Theta = np.linalg.inv(reg_S)
    
#     # Ensure initial matrix is well-conditioned
#     eigenvals = np.linalg.eigvals(Theta)
#     if np.min(eigenvals) <= 0:
#         Theta = Theta + (1e-6 - np.min(eigenvals)) * np.eye(p)
    
#     for iteration in range(max_iter):
#         Theta_old = Theta.copy()
        
#         # Block coordinate descent over each variable
#         for j in range(p):
#             # Partition matrices
#             idx = np.arange(p) != j
            
#             # S partitions
#             S_11 = S[np.ix_(idx, idx)]
#             s_12 = S[idx, j]
#             s_22 = S[j, j]
            
#             # Current Theta partitions  
#             Theta_11 = Theta[np.ix_(idx, idx)]
            
#             # Solve the regularized linear system for the j-th column/row
#             # We solve: (S_11 + alpha * I) * w = s_12
#             # where w will be related to the off-diagonal elements
            
#             try:
#                 # Add regularization for numerical stability
#                 reg_S_11 = S_11 + alpha * np.eye(p-1)
                
#                 # Solve the lasso problem using coordinate descent
#                 w = np.zeros(p-1)
                
#                 # Initialize with least squares solution
#                 try:
#                     w = np.linalg.solve(reg_S_11, s_12)
#                 except np.linalg.LinAlgError:
#                     w = np.linalg.lstsq(reg_S_11, s_12, rcond=None)[0]
                
#                 # Coordinate descent for lasso regularization
#                 for lasso_iter in range(400):
#                     w_old = w.copy()
                    
#                     for k in range(p-1):
#                         # Compute partial residual
#                         residual = s_12[k] - np.dot(reg_S_11[k, :], w) + reg_S_11[k, k] * w[k]
#                         # print(f"Largest eigenvalue of reg_S_11: {np.max(reg_S_11):.6f}")
#                         # print(f"Smallest eigenvalue of reg_S_11: {np.min(reg_S_11):.6f}")
#                         # Soft thresholding update
#                         if reg_S_11[k, k] > 1e-12:
#                             w[k] = soft_threshold(residual / reg_S_11[k, k], alpha / reg_S_11[k, k])
#                         else:
#                             w[k] = 0.0
                    
#                     # Check inner convergence
#                     if np.linalg.norm(w - w_old) < tol * 0.1:
#                         break
                
#                 # Update the precision matrix
#                 # Compute diagonal element
#                 theta_jj = 1.0 / (s_22 - np.dot(s_12, w))
                
#                 # Ensure positive diagonal
#                 if theta_jj <= 0 or not np.isfinite(theta_jj):
#                     theta_jj = 1.0 / (s_22 + alpha)
                
#                 # Update off-diagonal elements
#                 theta_j = -theta_jj * w
                
#                 # Set the j-th row and column
#                 Theta[j, j] = theta_jj
#                 Theta[j, idx] = theta_j
#                 Theta[idx, j] = theta_j
                
#             except (np.linalg.LinAlgError, ValueError):
#                 # Fallback: keep previous values or use simple update
#                 if verbose:
#                     print(f"Numerical issues at iteration {iteration}, variable {j}")
#                 continue
        
#         # Ensure symmetry
#         Theta = (Theta + Theta.T) / 2
        
#         # Check for positive definiteness and fix if needed
#         try:
#             eigenvals = np.linalg.eigvals(Theta)
#             min_eigval = np.min(eigenvals)
#             if min_eigval <= 1e-12:
#                 Theta = Theta + (1e-6 - min_eigval) * np.eye(p)
#         except:
#             pass
        
#         # Check convergence
#         diff = np.linalg.norm(Theta - Theta_old, 'fro')
#         if diff < tol:
#             if verbose:
#                 print(f"Converged after {iteration + 1} iterations (diff: {diff:.2e})")
#             break
        
#         if verbose and (iteration + 1) % 50 == 0:
#             print(f"Iteration {iteration + 1}, diff: {diff:.2e}")
    
#     else:
#         if verbose:
#             print(f"Maximum iterations ({max_iter}) reached")
    
#     # Final symmetry enforcement
#     Theta = (Theta + Theta.T) / 2
    
#     # Compute covariance matrix
#     try:
#         # Check condition number before inversion
#         cond_num = np.linalg.cond(Theta)
#         if cond_num > 1e12:
#             if verbose:
#                 print(f"Warning: Precision matrix is ill-conditioned (cond={cond_num:.2e})")
#             Sigma = np.linalg.pinv(Theta)
#         else:
#             Sigma = np.linalg.inv(Theta)
#     except np.linalg.LinAlgError:
#         if verbose:
#             print("Warning: Using pseudo-inverse for covariance matrix")
#         Sigma = np.linalg.pinv(Theta)
    
#     return Theta, Sigma



# def graphical_lasso_verbose(S, alpha, max_iter=100, tol=1e-6, verbose=True):
#     p = S.shape[0]
#     Theta = np.linalg.inv(S + alpha * np.eye(p))  # Initial regularized inverse
#     convergence_history = []
    
#     if verbose:
#         print(f"[Init] Problem size: {p} variables")
#         print(f"[Init] Alpha: {alpha}, Max Iter: {max_iter}, Tolerance: {tol}")
#         print("=" * 60)
    
#     for iteration in range(1, max_iter + 1):
#         Theta_old = Theta.copy()
        
#         for j in range(p):
#             idx = np.arange(p) != j
#             S_11 = S[np.ix_(idx, idx)]
#             s_12 = S[idx, j]
#             s_22 = S[j, j]
            
#             # Solve Lasso: minimize ||s_12 - S_11 @ w||² + alpha * ||w||₁
#             w = np.zeros(p - 1)
            
#             # Coordinate Descent for Lasso
#             for lasso_iter in range(200):
#                 w_old = w.copy()
                
#                 for k in range(p - 1):
#                     # Compute residual excluding k-th component
#                     mask = np.arange(p - 1) != k
#                     residual = s_12[k] - np.dot(S_11[k, mask], w[mask])
#                     # print(f"Largest eigenvalue of S_11: {np.max(S_11):.6f}")
#                     # print(f"Smallest eigenvalue of S_11: {np.min(S_11):.6f}")
#                     # Coordinate descent update with soft thresholding
#                     if S_11[k, k] > 1e-12:
#                         eig_vals = np.linalg.eigvals(Theta)

#                         w[k] = soft_threshold(residual / S_11[k, k], alpha / S_11[k, k])
#                     else:
#                         w[k] = 0.0
                
#                 # Check convergence of inner Lasso loop
#                 if np.linalg.norm(w - w_old) < tol * 0.1:
#                     if verbose and lasso_iter % 50 == 0:
#                         print(f"[Iter {iteration}] Var {j}, Lasso iter {lasso_iter}, ||w - w_old|| = {np.linalg.norm(w - w_old):.4e}")
#                     break
            
#             # Update Theta entries
#             denominator = s_22 - np.dot(s_12, w)
#             if denominator > 1e-6:
#                 theta_jj = 1.0 / denominator
#             else:
#                 theta_jj = 1.0 / 1e-6
                
#             theta_j = -theta_jj * w
            
#             # Update both row and column to maintain symmetry
#             Theta[j, j] = theta_jj
#             Theta[j, idx] = theta_j
#             Theta[idx, j] = theta_j
            
#             if verbose and j % 20 == 0:
#                 print(f"[Iter {iteration}] Updated variable {j}, theta_jj = {theta_jj:.4e}, ||theta_j|| = {np.linalg.norm(theta_j):.4e}")
        
#         # Check convergence (no need to symmetrize since we maintain symmetry)
#         diff = np.linalg.norm(Theta - Theta_old, 'fro')
#         convergence_history.append(diff)
        
#         if verbose:
#             print(f"[Iter {iteration}] Frobenius norm diff = {diff:.10e}")
        
#         if diff < tol:
#             if verbose:
#                 print(f"[Converged] after {iteration} iterations.")
#             break
    
#     if verbose:
#         print("=" * 60)
#         print("Final convergence history:")
#         for i, d in enumerate(convergence_history):
#             print(f"Iteration {i + 1}: {d:.10e}")
    
#     return Theta, np.linalg.pinv(Theta), convergence_history


# def prox_grad_step_(Sigma, Theta, mu1, eta, epsilon, bias_type, prec_type):
#     Soft_thresh = lambda R, alpha: np.maximum( np.abs(R)-alpha, 0 ) * np.sign(R)
#     p = Sigma.shape[0]

#     # Gradient step + soft-thresholding
#     # fairness_term = grad_fairness_penalty_(Theta, Z, bias_type) if mu2 != 0 else 0
#     Gradient = Sigma - la.inv( Theta + epsilon*np.eye(p) )
#     Theta_aux = Theta - eta*Gradient
#     Theta_aux[np.eye(p)==0] = Soft_thresh( Theta_aux[np.eye(p)==0], eta*mu1 )
#     Theta_aux = (Theta_aux + Theta_aux.T)/2

#     # Projection
#     if prec_type == 'non-negative':
#         # Projection onto non-negative matrices
#         Theta_aux[(Theta_aux <= 0)*(np.eye(p) == 0)] = 0
#     elif prec_type == 'non-positive':
#         # Projection onto non-positive matrices
#         Theta_aux[(Theta_aux >= 0)*(np.eye(p)==0)] = 0

#     # Second projection onto PSD set
#     eigenvals, eigenvecs = np.linalg.eigh( Theta_aux )
#     eigenvals[eigenvals < 0] = 0
#     Theta_next = eigenvecs @ np.diag( eigenvals ) @ eigenvecs.T

#     return Theta_next
# # ----------------------------------------------------




# def FairGLASSO_fista(Sigma, mu1, eta, bias_type, epsilon=.1, iters=1000,
#                      prec_type=None, tol=1e-3, EARLY_STOP=False, RETURN_ITERS=False):
#     """
#     Solve a graphical lasso problem with fairness regularization using the FISTA algorithm.

#     Parameters:
#     -----------
#     Sigma : numpy.ndarray
#         Sample covariance matrix.
#     mu1 : float
#         Weight for the l1 norm.
#     eta : float
#         Step size.
#     mu2 : float
#         Weight for the fairness penalty.
#     Z : numpy.ndarray
#         Matrix of sensitive attributes for the fairness penalty.
#     epsilon : float, optional
#         Small constant to load the diagonal of the estimated Theta to ensure strict positivity (default is 0.1).
#     iters : int, optional
#         Number of iterations (default is 1000).
#     EARLY_STOP: bool, optional
#         If True, end iterations when difference small enough.
#     A_true : numpy.ndarray or None, optional
#         True precision matrix to keep track of the error (default is None).

#     Returns:
#     --------
#     Theta_k : numpy.ndarray
#         Estimated precision matrix.
#     errs_A : numpy.ndarray
#         Array of errors in precision matrix estimation over iterations (if A_true is provided).

#     Notes:
#     ------
#     FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) implementation with the second demographic parity penalty.
#     """
#     print(Sigma.shape)
#     p = Sigma.shape[0]
#     # Ensure Theta_current is initialized to an invertible matrix
#     Theta_prev = np.eye(p)
#     Theta_fista = np.eye(p)
#     t_k = 1

#     # Initialize array to store iterations to check convergence
#     if RETURN_ITERS:
#         norm_iters = []

#     for _ in range(iters):
#         Theta_k = prox_grad_step_( Sigma, Theta_fista, mu1, eta, epsilon, bias_type, prec_type )
#         t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
#         Theta_fista = Theta_k + (t_k - 1)/t_next*(Theta_k - Theta_prev)
        
#         if EARLY_STOP and np.linalg.norm(Theta_prev-Theta_k,'fro') < tol:
#             break

#         # Update values for next iteration
#         if RETURN_ITERS:
#             norm_iters.append( np.linalg.norm(Theta_prev-Theta_k,'fro') )

#         Theta_prev = Theta_k
#         t_k = t_next

#     if RETURN_ITERS:
#         return Theta_k, norm_iters
    
#     return Theta_k


# def graphical_la(S, alpha, max_iter=1000, tol=1e-6, verbose=False):
#     """
#     Robust Graphical Lasso algorithm for sparse precision matrix estimation.
    
#     Parameters:
#     -----------
#     S : array-like, shape (p, p)
#         Sample covariance matrix
#     alpha : float
#         Regularization parameter (sparsity control)
#     max_iter : int
#         Maximum number of iterations
#     tol : float
#         Convergence tolerance
#     verbose : bool
#         Print convergence information
        
#     Returns:
#     --------
#     Theta : array, shape (p, p)
#         Estimated precision matrix
#     Sigma : array, shape (p, p)
#         Estimated covariance matrix
#     """
#     p = S.shape[0]
    
#     # Initialize with regularized sample covariance inverse
#     # Add small regularization to ensure positive definiteness
#     reg_S = S + alpha * np.eye(p)
#     Theta = np.linalg.inv(reg_S)
    
#     # Ensure initial matrix is well-conditioned
#     eigenvals = np.linalg.eigvals(Theta)
#     if np.min(eigenvals) <= 0:
#         Theta = Theta + (1e-6 - np.min(eigenvals)) * np.eye(p)
    
#     for iteration in range(max_iter):
#         Theta_old = Theta.copy()
        
#         # Block coordinate descent over each variable
#         for j in range(p):
#             # Partition matrices
#             idx = np.arange(p) != j
            
#             # S partitions
#             S_11 = S[np.ix_(idx, idx)]
#             s_12 = S[idx, j]
#             s_22 = S[j, j]
            
#             # Current Theta partitions  
#             Theta_11 = Theta[np.ix_(idx, idx)]
            
#             # Solve the regularized linear system for the j-th column/row
#             # We solve: (S_11 + alpha * I) * w = s_12
#             # where w will be related to the off-diagonal elements
            
#             try:
#                 # Add regularization for numerical stability
#                 reg_S_11 = S_11 + alpha * np.eye(p-1)
                
#                 # Solve the lasso problem using coordinate descent
#                 w = np.zeros(p-1)
                
#                 # Initialize with least squares solution
#                 try:
#                     w = np.linalg.solve(reg_S_11, s_12)
#                 except np.linalg.LinAlgError:
#                     w = np.linalg.lstsq(reg_S_11, s_12, rcond=None)[0]
                
#                 # Coordinate descent for lasso regularization
#                 for lasso_iter in range(400):
#                     w_old = w.copy()
                    
#                     for k in range(p-1):
#                         # Compute partial residual
#                         residual = s_12[k] - np.dot(reg_S_11[k, :], w) + reg_S_11[k, k] * w[k]

#                         # Soft thresholding update
#                         if reg_S_11[k, k] > 1e-12:
#                             w[k] = soft_threshold(residual / reg_S_11[k, k], alpha / reg_S_11[k, k])
#                         else:
#                             w[k] = 0.0
                    
#                     # Check inner convergence
#                     if np.linalg.norm(w - w_old) < tol * 0.1:
#                         break
                
#                 # Update the precision matrix
#                 # Compute diagonal element
#                 theta_jj = 1.0 / (s_22 - np.dot(s_12, w))
                
#                 # Ensure positive diagonal
#                 if theta_jj <= 0 or not np.isfinite(theta_jj):
#                     theta_jj = 1.0 / (s_22 + alpha)
                
#                 # Update off-diagonal elements
#                 theta_j = -theta_jj * w
                
#                 # Set the j-th row and column
#                 Theta[j, j] = theta_jj
#                 Theta[j, idx] = theta_j
#                 Theta[idx, j] = theta_j
                
#             except (np.linalg.LinAlgError, ValueError):
#                 # Fallback: keep previous values or use simple update
#                 if verbose:
#                     print(f"Numerical issues at iteration {iteration}, variable {j}")
#                 continue
        
#         # Ensure symmetry
#         Theta = (Theta + Theta.T) / 2
        
#         # Check for positive definiteness and fix if needed
#         try:
#             eigenvals = np.linalg.eigvals(Theta)
#             min_eigval = np.min(eigenvals)
#             if min_eigval <= 1e-12:
#                 Theta = Theta + (1e-6 - min_eigval) * np.eye(p)
#         except:
#             pass
        
#         # Check convergence
#         diff = np.linalg.norm(Theta - Theta_old, 'fro')
#         if diff < tol:
#             if verbose:
#                 print(f"Converged after {iteration + 1} iterations (diff: {diff:.2e})")
#             break
        
#         if verbose and (iteration + 1) % 50 == 0:
#             print(f"Iteration {iteration + 1}, diff: {diff:.2e}")
    
#     else:
#         if verbose:
#             print(f"Maximum iterations ({max_iter}) reached")
    
#     # Final symmetry enforcement
#     Theta = (Theta + Theta.T) / 2
    
#     # Compute covariance matrix
#     try:
#         # Check condition number before inversion
#         cond_num = np.linalg.cond(Theta)
#         if cond_num > 1e12:
#             if verbose:
#                 print(f"Warning: Precision matrix is ill-conditioned (cond={cond_num:.2e})")
#             Sigma = np.linalg.pinv(Theta)
#         else:
#             Sigma = np.linalg.inv(Theta)
#     except np.linalg.LinAlgError:
#         if verbose:
#             print("Warning: Using pseudo-inverse for covariance matrix")
#         Sigma = np.linalg.pinv(Theta)
    
#     return Theta, Sigma


class TopoGreedyBase:
    def __init__(self, verbose=False, dtype=np.float64, seed=0):
        np.random.seed(seed)
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
        self.W_est = None
        self.Theta_est = None
        self.convergence_history = None

    def _preprocess(self, X, max_iter=2000, tol=1e-4):
        """Common preprocessing steps for all variants"""
        self.X = X
        self.n, self.d = X.shape
        S = np.cov((X - X.mean(axis=0)) / X.std(axis=0).clip(min=1e-10).reshape(1,-1), rowvar=False)
        lambda_max = compute_lambda(self.n, self.d)
        return S, lambda_max

    def _postprocess(self, Theta_est):
        """Common postprocessing steps for all variants"""
        self.Theta_est = Theta_est
        self.W_est = estimate_adjacency_matrix(Theta_est, self.d)
        return self.W_est, self.Theta_est

class TopoGreedy_org(TopoGreedyBase):
    def fit(self, X, max_iter=2000, tol=1e-4):
        """Original TopoGreedy with standard graphical lasso"""
        S, lambda_max = self._preprocess(X, max_iter, tol)
        Theta_est = org_graphical_lasso(S, lambda_max, max_iter=max_iter, tol=tol, verbose=False)
        return self._postprocess(Theta_est)

class TopoGreedy_Fista(TopoGreedyBase):
    def fit(self, X, max_iter=2000, tol=1e-4):
        """TopoGreedy with FISTA optimization"""
        S, lambda_max = self._preprocess(X, max_iter, tol)
        Theta_est = FairGLASSO_fista(S, mu1=lambda_max, eta=0.01, bias_type='dp', 
                                    epsilon=0.1, iters=max_iter, prec_type='non-negative', 
                                    tol=tol, EARLY_STOP=False, RETURN_ITERS=False)
        return self._postprocess(Theta_est)

class TopoGreedy_TB(TopoGreedyBase):
    def fit(self, X, max_iter=2000, tol=1e-4):
        """TopoGreedy with Trench-Block optimization"""
        S, lambda_max = self._preprocess(X, max_iter, tol)
        Theta_est = graphical_lasso_TB(S, lambda_max, max_iter=max_iter, tol=tol, verbose=False)
        return self._postprocess(Theta_est)





class TopoGreedy:
    
    def __init__(self, verbose=False, dtype=np.float64, seed=0):
        super().__init__()
        np.random.seed(seed)
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None
        self.W_est = None
    
    def fit(self, X, max_iter=2000, tol=1e-4):
        """
        TopoGreedy algorithm for causal discovery.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            max_iter: Maximum iterations for graphical lasso
            tol: Tolerance for convergence
            edge_threshold: Threshold for removing weak edges
        
        Returns:
            A_est: Estimated adjacency matrix
        """
        self.X = X
        self.n, self.d = X.shape
        S = np.cov(X.T, bias=True)
        lambda_max = compute_lambda(self.n, self.d)
        estimated_Theta = org_graphical_lasso(S, lambda_max, max_iter=max_iter, tol=tol, verbose=False)
        self.W_est = estimate_adjacency_matrix(estimated_Theta, self.d)
        self.Theta_est = estimated_Theta
        return self.W_est, self.Theta_est
