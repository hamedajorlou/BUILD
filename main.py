import numpy as np
import matplotlib.pyplot as plt 
from TopoGreedy import estimate_adjacency_matrix
from graphicalLasso import generate_sparse_precision_matrix, graphical_lasso
from utils import simulate_sem, calculate_shd, compute_lambda
import warnings
import networkx as nx
import time
from timeit import default_timer as timer
from baselines import DAGMA_linear
import dagu
from baselines import colide_ev
from utils import to_dag, count_accuracy


def demo_graphical_lasso(Exps):
    """Demonstrate the Graphical Lasso algorithm using our own implementation."""
    print("Graphical Lasso Demo (using our own implementation)")
    print("=" * 50)
    
    # Parameters for both experiments   

    # Generate a random DAG with 20 nodes and 100 edges
    M = 10000 # Number of samples
    N = 50
    k = 0.5
    graph_type = 'er'  # 'er' for Erdős-Rényi, 'sf' for scale-free
    edge_type = 'weighted'  # 'binary' for binary edges, 'weighted' for weighted edges
    var_type = 'ev'  # 'ev' for equal variances, 'unif' for uniform variances
    noise = 'normal'  # 'normal' for normal noise, 't' for t-distribution noise
    var = 1.0  # Variance of noise
    w_range = ((-2.0, -0.5), (0.5, 2.0))  # Weight range for weighted edges
    seed = 123  # Random seed for reproducibility

    Iterations = 10  
    shd_values = []
    for exp in Exps:
        for run in range(Iterations):
            print(f"Run {run + 1}/{Iterations}")
            run_seed = seed + run
            X, W_weighted, var_nv, Theta = simulate_sem(N, M, k, graph_type, edge_type, var_type, noise, var, w_range, run_seed)
            # Print the number of edges in W_weighted
            num_edges = np.sum(W_weighted != 0)
            print(f"Number of edges in W_weighted: {num_edges}")
            if exp['models']['model'] == "TopoGreedy":

                S = np.cov(X.T, bias=True)
                lambda_max = compute_lambda(N, M)
                t_start = time.time()
                
                estimated_Theta = graphical_lasso(S, lambda_max, max_iter=2000, tol=1e-4)
                A_est = estimate_adjacency_matrix(estimated_Theta, N, edge_threshold=0.2)

                shd = calculate_shd(W_weighted, A_est)
                print(shd)

    
            elif exp['models']['model'] == "Colide":
                seed = 123
                model1 = colide_ev(seed=seed)
                t_start = time.time()
                W_hat_ev, sigma_est_ev = model1.fit(X, lambda1=0.05, T=4, s=[1.0, .9, .8, .7], warm_iter=2e4, max_iter=7e4, lr=0.0003)
                t_end = time.time()
                print(f'convergence time for CoLiDE-EV: {t_end-t_start:.4f}s')
                W_hat_post_ev = to_dag(W_hat_ev, thr=0.3)
                fdr_ev, tpr_ev, fpr_ev, shd_ev, pred_size_ev = count_accuracy(W_weighted.T!=0, W_hat_post_ev!=0)
                # Plot W_hat_ev and W_weighted in heatmap fashion
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot W_weighted (ground truth weighted adjacency matrix)
                vmax_W = max(np.max(np.abs(W_weighted)), np.max(np.abs(W_hat_ev)))
                im1 = axes[0].imshow(W_weighted.T, cmap='RdBu_r', vmin=-vmax_W, vmax=vmax_W)
                axes[0].set_title(f'Ground Truth Weighted Adjacency Matrix ({N} nodes)', fontsize=12)
                axes[0].set_xlabel('Variable', fontsize=10)
                axes[0].set_ylabel('Variable', fontsize=10)
                plt.colorbar(im1, ax=axes[0], shrink=0.8)
                
                # Plot W_hat_ev (estimated weighted adjacency matrix)
                im2 = axes[1].imshow(W_hat_ev, cmap='RdBu_r', vmin=-vmax_W, vmax=vmax_W)
                axes[1].set_title(f'Estimated Weighted Adjacency Matrix ({N} nodes)', fontsize=12)
                axes[1].set_xlabel('Variable', fontsize=10)
                axes[1].set_ylabel('Variable', fontsize=10)
                plt.colorbar(im2, ax=axes[1], shrink=0.8)
                
                plt.tight_layout()
                plt.show()
                print(f"  SHD for run {run + 1}: {shd_ev}") 



            elif exp['models']['model'] == "DAGMA":   


                dagu.set_random_seed(1)
                
                # n, d, s0 = 500, 20, 20 # the ground truth is a DAG of 20 nodes and 20 edges in expectation
                # graph_type, sem_type = 'ER', 'gauss'
                d = 20
                # X = dagu.simulate_linear_sem(W_true, n, sem_type)
                
                model = DAGMA_linear(loss_type='l2')
                start = timer()
                W_est = model.fit(X, lambda1=0.02)
                end = timer()
                shd = calculate_shd(W_weighted.T, W_est)
                # acc = dagu.count_accuracy(W_weighted.T, W_est != 0)
                # print(acc)
                print(f'time: {end-start:.4f}s')
                # B_true = dagu.simulate_dag(d, s0, graph_type)
                # W_true = dagu.simulate_parameter(B_true)
                # Plot B_true and W_true in heatmap fashion
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot B_true (binary adjacency matrix)
                im1 = axes[0].imshow(W_weighted.T, cmap='Blues', vmin=0, vmax=1)
                axes[0].set_title(f'Ground Truth Binary Adjacency Matrix B ({d} nodes)', fontsize=12)
                axes[0].set_xlabel('Variable', fontsize=10)
                axes[0].set_ylabel('Variable', fontsize=10)
                plt.colorbar(im1, ax=axes[0], shrink=0.8)
                
                # Plot W_true (weighted adjacency matrix)
                vmax_W = np.max(np.abs(W_est))
                im2 = axes[1].imshow(W_est, cmap='RdBu_r', vmin=-vmax_W, vmax=vmax_W)
                axes[1].set_title(f'Ground Truth Weighted Adjacency Matrix W ({d} nodes)', fontsize=12)
                axes[1].set_xlabel('Variable', fontsize=10)
                axes[1].set_ylabel('Variable', fontsize=10)
                plt.colorbar(im2, ax=axes[1], shrink=0.8)
                
                plt.tight_layout()
                plt.show()
                # Calculate SHD (Structural Hamming Distance) from scratch
                # SHD counts the number of edge differences between two graphs
                # Convert to binary adjacency matrices for comparison
                B_true = (np.abs(W_weighted.T) > 1e-6).astype(int)
                B_est = (np.abs(W_est) > 1e-6).astype(int)
                
                # Calculate SHD: sum of all differing entries
                shd = np.sum(B_true != B_est)
                print("SHD for DAGMA: ", shd)

        # diff = estimated_Theta - Theta

        # print(f"Normalized Frobenius norm of true Theta: {np.linalg.norm(Theta, 'fro') / np.sqrt(Theta.size):.4f}")
        # print(f"Normalized Frobenius norm of estimated Theta: {np.linalg.norm(estimated_Theta, 'fro') / np.sqrt(estimated_Theta.size):.4f}")
        # print(f"Normalized Frobenius norm of difference: {np.linalg.norm(diff, 'fro') / np.sqrt(diff.size):.4f}")
        # print(f"Relative error: {np.linalg.norm(diff, 'fro') / np.linalg.norm(Theta, 'fro'):.4f}")
        # gt_edges = np.sum(np.abs(W_weighted) > 1e-6)
        # est_edges = np.sum(np.abs(A_est) > 1e-6)
        # print(f"Ground truth edges: {gt_edges}")
        # print(f"Estimated edges: {est_edges}")
        # t_end = time.time()
        # t_elapsed = t_end - t_start
        # print(f"Time taken: {t_elapsed:.2f} seconds")

        

# ================================  Graphical Lasso with different alpha values  ================================


    # # Plot adjacency matrices from the last run
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # # Plot true adjacency matrix
    # im1 = ax1.imshow(W_weighted, cmap='RdBu_r', vmin=-2, vmax=2)
    # ax1.set_title('True Adjacency Matrix (W_weighted)')
    # ax1.set_xlabel('Node')
    # ax1.set_ylabel('Node')
    # plt.colorbar(im1, ax=ax1)
    
    # # Plot estimated adjacency matrix
    # im2 = ax2.imshow(A_est, cmap='RdBu_r', vmin=-2, vmax=2)
    # ax2.set_title('Estimated Adjacency Matrix (A_est)')
    # ax2.set_xlabel('Node')
    # ax2.set_ylabel('Node')
    # plt.colorbar(im2, ax=ax2)
    
    # plt.tight_layout()
    # plt.show()

    # for N in node_sizes:
    #     print(f"\n{'='*20} Running experiment with {N} nodes {'='*20}")
        
    #     # Generate true sparse precision matrix using the function
    #     true_Theta, A_true = generate_sparse_precision_matrix(N, p)

    #     print('A_true',A_true)

    #     true_Sigma = np.linalg.inv(true_Theta)

    #     # Print normalized Frobenius norm of true precision matrix
    #     true_Theta_norm = np.linalg.norm(true_Theta, 'fro') / np.sqrt(N * N-1)
    #     print(f"Normalized Frobenius norm of true precision matrix: {true_Theta_norm:.4f}")
        
    #     # Generate sample data
    #     np.random.seed(123)
    #     X = np.random.multivariate_normal(np.zeros(N), true_Sigma, M)
        
    #     # Compute sample covariance matrix
    #     S = np.cov(X.T)
    #     print(f"Sample covariance condition number: {np.linalg.cond(S):.2e}")
        
    #     # Try different alpha values and pick the best one
    #     best_alpha = None
    #     best_score = float('inf')
    #     best_results = None
        
    #     print(f"Trying different alpha values: {alphas}")
        
    #     for alpha in alphas:
    #         print(f"  Testing alpha={alpha}...")
            
    #         # Apply Graphical Lasso using our own implementation
    #         try:
    #             estimated_Theta_raw, estimated_Sigma = graphical_lasso(S, alpha, max_iter=2000, tol=1e-4, verbose=False)
                
    #             # Check if the result is valid (no NaN or infinite values)
    #             if np.any(np.isnan(estimated_Theta_raw)) or np.any(np.isinf(estimated_Theta_raw)):
    #                 print(f"    Skipping alpha={alpha} due to invalid results (NaN/Inf)")
    #                 continue
                
    #             # Apply threshold to estimated precision matrix
    #             estimated_Theta = estimated_Theta_raw.copy()
    #             estimated_Theta[np.abs(estimated_Theta) < threshold] = 0
                
    #             # Estimate A using the algorithm on estimated Theta
    #             A_est = np.zeros((N, N))
    #             Theta_aux = estimated_Theta.copy()
                
    #             A_est = estimate_adjacency_matrix(estimated_Theta, N)
    #             A_est[np.abs(A_est) < 0.25] = 0

    #             # Calculate score based on multiple metrics
    #             theta_error = np.linalg.norm(true_Theta - estimated_Theta, 'fro') / np.sqrt(N * N-1)
    #             a_error = np.linalg.norm(A_true - A_est, 'fro') / np.sqrt(N * N-1)
                
    #             # Calculate Structural Hamming Distance (SHD)
    #             true_adj = (np.abs(A_true) > 1e-6).astype(int)
    #             est_adj = (np.abs(A_est) > 1e-3).astype(int)
    #             shd = np.sum(true_adj != est_adj)
    #             normalized_shd = shd / (N * N-1)
                
    #             # Composite score (lower is better)
    #             score = theta_error + a_error + normalized_shd
                
    #             print(f"    Theta error: {theta_error:.4f}, A error: {a_error:.4f}, Normalized SHD: {normalized_shd:.4f}, Score: {score:.4f}")
                
    #             if score < best_score:
    #                 best_score = score
    #                 best_alpha = alpha
    #                 best_results = {
    #                     'estimated_Theta': estimated_Theta,
    #                     'A_est': A_est,
    #                     'theta_error': theta_error,
    #                     'a_error': a_error,
    #                     'normalized_shd': normalized_shd
    #                 }
                    
    #         except Exception as e:
    #             print(f"    Error with alpha={alpha}: {str(e)}")
    #             continue
        
    #     if best_results is None:
    #         print(f"Warning: No valid results found for {N} nodes. Skipping...")
    #         continue
            
    #     print(f"\nBest alpha: {best_alpha} with score: {best_score:.4f}")
        
    #     # Use best results
    #     estimated_Theta = best_results['estimated_Theta']
    #     A_est = best_results['A_est']
        
    #     # Print normalized Frobenius norm of estimated precision matrix
    #     estimated_Theta_norm = np.linalg.norm(estimated_Theta, 'fro') / np.sqrt(N * N)
    #     print(f"Normalized Frobenius norm of estimated precision matrix: {estimated_Theta_norm:.4f}")

    #     print(f"Estimated A: {A_est}")
    #     print(f"True A: {A_true}")

    #     # Store results
    #     results[N] = {
    #         'true_Theta': true_Theta,
    #         'estimated_Theta': estimated_Theta,
    #         'A_true': A_true,
    #         'A_est': A_est,
    #         'threshold': threshold,
    #         'best_alpha': best_alpha
    #     }
        
    #     # Print comparison metrics
    #     print(f"\nComparison Results for {N} nodes (alpha={best_alpha}):")
    #     print(f"Theta normalized Frobenius norm error: {best_results['theta_error']:.4f}")
    #     print(f"A normalized Frobenius norm error: {best_results['a_error']:.4f}")
    #     print(f"A sparsity recovery: {np.mean((A_true != 0) == (np.abs(A_est) > 1e-3)):.4f}")
        
    #     # Calculate Structural Hamming Distance (SHD)
    #     true_adj = (np.abs(A_true) > 1e-6).astype(int)
    #     est_adj = (np.abs(A_est) > 1e-3).astype(int)
    #     total_true_edges = np.sum(true_adj)
    #     total_est_edges = np.sum(est_adj)
    #     print(f"Total edges in true graph: {total_true_edges}")
    #     print(f"Total edges in estimated graph: {total_est_edges}")
    #     shd = np.sum(true_adj != est_adj)
    #     normalized_shd = shd / (N * N-1)
    #     print(f"Structural Hamming Distance (SHD): {shd}")
    #     print(f"Normalized SHD: {normalized_shd:.4f}")

    # # Filter results to only include successful experiments
    # valid_results = {k: v for k, v in results.items() if k in results}
    
    # if len(valid_results) < 1:
    #     print("Warning: No valid results for visualization")
    #     return results

    # # Create individual plots for each N value
    # for N in sorted(valid_results.keys()):
    #     true_Theta = results[N]['true_Theta']
    #     estimated_Theta = results[N]['estimated_Theta']
    #     A_true = results[N]['A_true']
    #     A_est = results[N]['A_est']
    #     threshold = results[N]['threshold']
    #     best_alpha = results[N]['best_alpha']
        
    #     # Calculate difference matrices
    #     theta_diff = true_Theta - estimated_Theta
    #     A_diff = A_true - A_est
        
    #     # Create a figure with 6 subplots (2 rows, 3 columns)
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
    #     # True precision matrix
    #     vmax_theta = max(np.max(np.abs(true_Theta)), np.max(np.abs(estimated_Theta)))
    #     im1 = axes[0, 0].imshow(true_Theta, cmap='RdBu_r', vmin=-vmax_theta, vmax=vmax_theta)
    #     axes[0, 0].set_title(f'True Precision Matrix ({N} nodes)', fontsize=12)
    #     axes[0, 0].set_xlabel('Variable', fontsize=10)
    #     axes[0, 0].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
    #     # Estimated precision matrix
    #     im2 = axes[0, 1].imshow(estimated_Theta, cmap='RdBu_r', vmin=-vmax_theta, vmax=vmax_theta)
    #     axes[0, 1].set_title(f'Estimated Precision Matrix ({N} nodes, α={best_alpha})', fontsize=12)
    #     axes[0, 1].set_xlabel('Variable', fontsize=10)
    #     axes[0, 1].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
    #     # Difference in precision matrices
    #     vmax_theta_diff = np.max(np.abs(theta_diff))
    #     im3 = axes[0, 2].imshow(theta_diff, cmap='RdBu_r', vmin=-vmax_theta_diff, vmax=vmax_theta_diff)
    #     axes[0, 2].set_title(f'Precision Matrix Difference ({N} nodes)', fontsize=12)
    #     axes[0, 2].set_xlabel('Variable', fontsize=10)
    #     axes[0, 2].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        
    #     # True A matrix
    #     vmax_A = max(np.max(np.abs(A_true)), np.max(np.abs(A_est)))
    #     im4 = axes[1, 0].imshow(A_true, cmap='viridis', vmin=0, vmax=vmax_A)
    #     axes[1, 0].set_title(f'Ground Truth A Matrix ({N} nodes)', fontsize=12)
    #     axes[1, 0].set_xlabel('Variable', fontsize=10)
    #     axes[1, 0].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        
    #     # Estimated A matrix
    #     im5 = axes[1, 1].imshow(A_est, cmap='viridis', vmin=0, vmax=vmax_A)
    #     axes[1, 1].set_title(f'Estimated A Matrix ({N} nodes, α={best_alpha})', fontsize=12)
    #     axes[1, 1].set_xlabel('Variable', fontsize=10)
    #     axes[1, 1].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        
    #     # Difference in A matrices
    #     vmax_A_diff = np.max(np.abs(A_diff))
    #     im6 = axes[1, 2].imshow(A_diff, cmap='RdBu_r', vmin=-vmax_A_diff, vmax=vmax_A_diff)
    #     axes[1, 2].set_title(f'A Matrix Difference ({N} nodes)', fontsize=12)
    #     axes[1, 2].set_xlabel('Variable', fontsize=10)
    #     axes[1, 2].set_ylabel('Variable', fontsize=10)
    #     plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        
    #     plt.tight_layout(pad=2.0)
    #     plt.savefig(f'graphical_lasso_comparison_{N}_nodes.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    
    # return results

# Run the demo
if __name__ == "__main__":

    Exps = [
        # Our Models

        # {'models': {'model': "TopoGreedy", 'max_iter': 2000, 'tol': 1e-4, 'c': 0.01, 'edge_threshold': 0.2}},
        # {'models': {'model': "Colide"}},
        {'models': {'model': "DAGMA"}},

    ]

    demo_graphical_lasso(Exps)
