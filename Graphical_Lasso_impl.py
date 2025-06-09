import numpy as np
import scipy.linalg as la
from scipy.linalg import inv, LinAlgError
import warnings


def soft_threshold(x, threshold):
    """Apply soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def org_graphical_lasso(S, alpha, max_iter=1000, tol=1e-6, verbose=False):
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
                for lasso_iter in range(400):
                    w_old = w.copy()
                    
                    for k in range(p-1):
                        # Compute partial residual
                        residual = s_12[k] - np.dot(reg_S_11[k, :], w) + reg_S_11[k, k] * w[k]
                        # print(f"Largest eigenvalue of reg_S_11: {np.max(reg_S_11):.6f}")
                        # print(f"Smallest eigenvalue of reg_S_11: {np.min(reg_S_11):.6f}")
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


def prox_grad_step_(Sigma, Theta, mu1, eta, epsilon, bias_type, prec_type):
    Soft_thresh = lambda R, alpha: np.maximum( np.abs(R)-alpha, 0 ) * np.sign(R)
    p = Sigma.shape[0]

    # Gradient step + soft-thresholding
    # fairness_term = grad_fairness_penalty_(Theta, Z, bias_type) if mu2 != 0 else 0
    Gradient = Sigma - la.inv( Theta + epsilon*np.eye(p) )
    Theta_aux = Theta - eta*Gradient
    Theta_aux[np.eye(p)==0] = Soft_thresh( Theta_aux[np.eye(p)==0], eta*mu1 )
    Theta_aux = (Theta_aux + Theta_aux.T)/2

    # Projection
    if prec_type == 'non-negative':
        # Projection onto non-negative matrices
        Theta_aux[(Theta_aux <= 0)*(np.eye(p) == 0)] = 0
    elif prec_type == 'non-positive':
        # Projection onto non-positive matrices
        Theta_aux[(Theta_aux >= 0)*(np.eye(p)==0)] = 0

    # Second projection onto PSD set
    eigenvals, eigenvecs = np.linalg.eigh( Theta_aux )
    eigenvals[eigenvals < 0] = 0
    Theta_next = eigenvecs @ np.diag( eigenvals ) @ eigenvecs.T

    return Theta_next
# ----------------------------------------------------



def FairGLASSO_fista(Sigma, mu1, eta, bias_type, epsilon=.1, iters=1000,
                     prec_type=None, tol=1e-3, EARLY_STOP=False, RETURN_ITERS=False):
    """
    Solve a graphical lasso problem with fairness regularization using the FISTA algorithm.

    Parameters:
    -----------
    Sigma : numpy.ndarray
        Sample covariance matrix.
    mu1 : float
        Weight for the l1 norm.
    eta : float
        Step size.
    mu2 : float
        Weight for the fairness penalty.
    Z : numpy.ndarray
        Matrix of sensitive attributes for the fairness penalty.
    epsilon : float, optional
        Small constant to load the diagonal of the estimated Theta to ensure strict positivity (default is 0.1).
    iters : int, optional
        Number of iterations (default is 1000).
    EARLY_STOP: bool, optional
        If True, end iterations when difference small enough.
    A_true : numpy.ndarray or None, optional
        True precision matrix to keep track of the error (default is None).

    Returns:
    --------
    Theta_k : numpy.ndarray
        Estimated precision matrix.
    errs_A : numpy.ndarray
        Array of errors in precision matrix estimation over iterations (if A_true is provided).

    Notes:
    ------
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) implementation with the second demographic parity penalty.
    """
    p = Sigma.shape[0]
    # Ensure Theta_current is initialized to an invertible matrix
    Theta_prev = np.eye(p)
    Theta_fista = np.eye(p)
    t_k = 1

    # Initialize array to store iterations to check convergence
    if RETURN_ITERS:
        norm_iters = []

    for _ in range(iters):
        Theta_k = prox_grad_step_( Sigma, Theta_fista, mu1, eta, epsilon, bias_type, prec_type )
        t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
        Theta_fista = Theta_k + (t_k - 1)/t_next*(Theta_k - Theta_prev)
        
        if EARLY_STOP and np.linalg.norm(Theta_prev-Theta_k,'fro') < tol:
            break

        # Update values for next iteration
        if RETURN_ITERS:
            norm_iters.append( np.linalg.norm(Theta_prev-Theta_k,'fro') )

        Theta_prev = Theta_k
        t_k = t_next

    if RETURN_ITERS:
        return Theta_k, norm_iters
    
    return Theta_k

def coordinate_descent_lasso(V, u, rho, max_iter=1000, tol=1e-6):
    """
    Solve lasso problem using coordinate descent:
    min_beta { 1/2 ||V^{1/2} beta - b||^2 + rho ||beta||_1 }
    
    where b = V^{-1/2} u
    
    Parameters:
    -----------
    V : ndarray, shape (p-1, p-1)
        Covariance-like matrix (W_11 in the paper)
    u : ndarray, shape (p-1,)
        Cross-covariance vector (s_12 in the paper)
    rho : float
        Regularization parameter
    max_iter : int
        Maximum iterations for coordinate descent
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    beta : ndarray, shape (p-1,)
        Solution vector
    """
    p_minus_1 = V.shape[0]
    beta = np.zeros(p_minus_1)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(p_minus_1):
            # Compute partial residual
            partial_residual = u[j] - np.dot(V[j, :], beta) + V[j, j] * beta[j]
            
            # Coordinate descent update with soft thresholding
            beta[j] = soft_threshold(partial_residual, rho) / V[j, j]
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break
            
    return beta

def graphical_lasso_TB(S, rho, max_iter=100, tol=1e-3, verbose=False):
    """
    Graphical Lasso algorithm for sparse inverse covariance estimation.
    
    Solves: max log det(Theta) - tr(S * Theta) - rho * ||Theta||_1
    
    Parameters:
    -----------
    S : ndarray, shape (p, p)
        Empirical covariance matrix
    rho : float
        Regularization parameter (penalty weight)
    max_iter : int
        Maximum number of outer iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print convergence information
        
    Returns:
    --------
    W : ndarray, shape (p, p)
        Estimated covariance matrix
    Theta : ndarray, shape (p, p)
        Estimated inverse covariance (precision) matrix
    n_iter : int
        Number of iterations until convergence
    """
    p = S.shape[0]
    
    # Initialize W = S + rho * I (Algorithm step 1)
    W = S + rho * np.eye(p)
    
    # Store beta coefficients for computing Theta later
    B = np.zeros((p, p))
    
    for outer_iter in range(max_iter):
        W_old = W.copy()
        
        # Cycle through each variable (Algorithm step 2)
        for j in range(p):
            # Create indices for current partition
            idx = np.arange(p) != j
            
            # Partition matrices: W_11, w_12, s_12
            W_11 = W[np.ix_(idx, idx)]
            s_12 = S[idx, j]
            
            # Solve lasso problem (equation 2.4 in paper)
            try:
                beta = coordinate_descent_lasso(W_11, s_12, rho)
                B[idx, j] = beta
                
                # Update W: w_12 = W_11 * beta
                W[idx, j] = np.dot(W_11, beta)
                W[j, idx] = W[idx, j]  # Maintain symmetry
                
            except LinAlgError:
                warnings.warn("Numerical issues encountered. Results may be inaccurate.")
                continue
        
        # Check convergence
        avg_change = np.mean(np.abs(W - W_old))
        avg_S_offdiag = np.mean(np.abs(S[np.triu_indices(p, k=1)]))
        
        if avg_change < tol * avg_S_offdiag:
            if verbose:
                print(f"Converged after {outer_iter + 1} iterations")
            break
            
        if verbose and (outer_iter + 1) % 10 == 0:
            print(f"Iteration {outer_iter + 1}: avg change = {avg_change:.6f}")
    
    # Compute inverse covariance matrix Theta = W^{-1}
    # Using the efficient formulas from equations (2.13) and (2.14)
    Theta = np.zeros_like(W)
    
    for j in range(p):
        idx = np.arange(p) != j
        
        W_11 = W[np.ix_(idx, idx)]
        w_12 = W[idx, j]
        w_22 = W[j, j]
        
        beta = B[idx, j]
        
        # Compute theta_22 and theta_12
        try:
            theta_22 = 1.0 / (w_22 - np.dot(w_12, beta))
            theta_12 = -beta * theta_22
            
            Theta[j, j] = theta_22
            Theta[idx, j] = theta_12
            Theta[j, idx] = theta_12
            
        except (ZeroDivisionError, LinAlgError):
            # Fallback to direct matrix inversion
            try:
                Theta = inv(W)
                break
            except LinAlgError:
                warnings.warn("Could not compute inverse. Returning covariance estimate only.")
                Theta = None
                break
    
    return Theta
