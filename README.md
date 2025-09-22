# BUILD: Bayesian Undirected Inference for Learning DAGs

BUILD is a novel algorithm for causal discovery that combines Bayesian inference with undirected graph learning to efficiently discover Directed Acyclic Graphs (DAGs) from observational data. This implementation provides state-of-the-art performance in learning causal structures with improved scalability and accuracy.

## Overview

The BUILD algorithm works by:
1. Learning an undirected graph structure using sparse precision matrix estimation
2. Applying Bayesian inference to determine edge orientations
3. Iteratively refining the DAG structure through topological constraints
4. Combining multiple baselines for robust causal discovery

## Repository Structure

### Core Algorithm Files
- `BUILD.py` - Main BUILD algorithm implementation
- `utils.py` - Utility functions for data processing and evaluation
- `dag_utils.py` - DAG-specific utility functions
- `Greedy_prune.py` - Greedy pruning algorithms for graph refinement

### Baseline Implementations
- `Baselines.py` - Comprehensive collection of baseline algorithms including:
  - CoLiDE-EV (Equal Variance)
  - CoLiDE-NV (Non-equal Variance) 
  - DAGMA-linear
  - Non-negative DAGMA
  - MetMulDagma
  - NOTEARS-linear

### Notebooks and Examples
- `Continous_relax.ipynb` - Continuous relaxation experiments and analysis
- `ICMLpaper.ipynb` - Experiments and results for ICML paper

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules for local development files

## Features

### BUILD Algorithm
- **Bayesian Inference**: Uses Bayesian principles for robust edge orientation
- **Undirected Learning**: Leverages sparse precision matrix estimation
- **Topological Constraints**: Enforces DAG properties through iterative refinement
- **Scalable Implementation**: Efficient algorithms for large-scale causal discovery

### Comprehensive Baselines
- **CoLiDE Variants**: Both equal and non-equal variance implementations
- **DAGMA Methods**: Linear and non-negative variants with multiple optimization strategies
- **NOTEARS**: Linear structural equation modeling approach
- **MetMulDagma**: Multiplicative DAGMA with advanced optimization

### Key Features
- Multiple optimization strategies (FISTA, coordinate descent, block operations)
- Automatic hyperparameter selection
- Robust handling of numerical instabilities
- Comprehensive evaluation metrics
- Support for different graph types and noise models

## Installation

```bash
# Clone the repository
git clone https://github.com/hamedajorlou/BUILD.git
cd BUILD

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic BUILD Algorithm

```python
from BUILD import BUILD
import numpy as np

# Generate or load your data
X = np.random.randn(100, 10)  # 100 samples, 10 variables

# Initialize and run BUILD
model = BUILD()
W_est, Theta_est = model.fit(X, lambda1=0.1)

print(f"Estimated adjacency matrix shape: {W_est.shape}")
```

### Using Baselines

```python
from Baselines import DAGMA_linear, colide_ev, notears_linear

# DAGMA-linear
dagma_model = DAGMA_linear(loss_type='l2')
W_dagma = dagma_model.fit(X, lambda1=0.1)

# CoLiDE-EV
colide_model = colide_ev()
W_colide, sigma = colide_model.fit(X, lambda1=0.1)

# NOTEARS
W_notears = notears_linear(X, lambda1=0.1)
```

### Advanced Usage with Custom Parameters

```python
from BUILD import BUILD

# Initialize with custom parameters
model = BUILD(
    max_iter=1000,
    tol=1e-6,
    verbose=True
)

# Fit with custom regularization
W_est, Theta_est = model.fit(
    X, 
    lambda1=0.05,
    edge_threshold=0.1
)
```

## API Reference

### Class: BUILD

#### Parameters
- `max_iter` (int): Maximum number of iterations
- `tol` (float): Convergence tolerance
- `verbose` (bool): Enable/disable verbose output

#### Methods
- `fit(X, lambda1, edge_threshold=0.1)`: Fit the model to data
  - `X`: Input data matrix of shape (n_samples, n_features)
  - `lambda1`: Regularization parameter for sparsity
  - `edge_threshold`: Threshold for edge selection
  - Returns: `(W_est, Theta_est)` - Estimated adjacency matrix and precision matrix

### Baseline Classes

#### DAGMA_linear
- `fit(X, lambda1, w_threshold=0.3, T=5, ...)`: Fit DAGMA model

#### colide_ev / colide_nv
- `fit(X, lambda1, T=5, ...)`: Fit CoLiDE model

#### Nonneg_dagma
- `fit(X, alpha, lamb, stepsize, ...)`: Fit non-negative DAGMA

## Evaluation and Metrics

The repository includes comprehensive evaluation tools:

```python
import utils

# Load ground truth and estimated adjacency matrices
W_true = np.load('true_adjacency.npy')
W_est = np.load('estimated_adjacency.npy')

# Compute accuracy metrics
shd, tpr, fdr = utils.count_accuracy(W_true, W_est)
f1_score = utils.compute_f1_score(W_true, W_est)
norm_error = utils.compute_norm_sq_err(W_true, W_est)

print(f"SHD: {shd}, TPR: {tpr:.3f}, FDR: {fdr:.3f}")
print(f"F1 Score: {f1_score:.3f}, Normalized Error: {norm_error:.3f}")
```

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0

## Local Development Files

The following files are kept locally for development but not tracked in the repository:
- `Samu__run.ipynb` - Samuel's experimental notebook
- `dagu.py` - DAG utilities (local development)
- `TopoGreedy.py` - TopoGreedy implementation (local development)
- `notebook.ipynb` - General experimental notebook
- `main.py` - Main execution script (local development)
- `samu_run.ipynb` - Samuel's run notebook
- `baselines.py` - Alternative baselines implementation (local development)
- `continous_relaxation.ipynb` - Continuous relaxation experiments (local development)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For local development, you can work with the ignored files without affecting the repository.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{build2024,
  author = {Ajorlou, Hamed and Rey, Samuel and García Marques, Antonio and Mateos, Gonzalo},
  title = {BUILD: Bayesian Undirected Inference for Learning DAGs},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/hamedajorlou/BUILD}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
