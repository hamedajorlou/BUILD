# BUILD: Bottom-Up Inference of Linear DAGs

BUILD is a novel algorithm for causal discovery that uses a bottom-up approach to efficiently discover Directed Acyclic Graphs (DAGs) from observational data. This implementation provides state-of-the-art performance in learning causal structures with improved scalability and accuracy.

![BUILD Algorithm Overview](DAGprec.jpeg)

## Overview

The BUILD algorithm works by:
1. Starting from leaf nodes and building up the DAG structure
2. Using bottom-up inference to determine causal relationships
3. Iteratively building the DAG from bottom to top
4. Leveraging linear DAG assumptions for efficient discovery

## Repository Structure

### Core Algorithm Files
- `BUILD.py` - Main BUILD algorithm implementation
- `utils.py` - Utility functions for data processing and evaluation
- `dag_utils.py` - DAG-specific utility functions
- `Greedy_prune.py` - Greedy pruning algorithms for graph refinement

### Baseline Implementations
- `Baselines.py` - Comprehensive collection of baseline algorithms including:
  - CoLiDE-EV (Equal Variance) [Zhang et al., 2023]
  - CoLiDE-NV (Non-equal Variance) 
  - DAGMA-linear [Bello et al., 2022]
  - Non-negative DAGMA
  - MetMulDagma
  - NOTEARS-linear [Zheng et al., 2018]

### Notebooks and Examples
- `Continous_relax.ipynb` - Continuous relaxation experiments and analysis
- `ICMLpaper.ipynb` - Experiments and results for ICML paper

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules for local development files

## Features

### BUILD Algorithm
- **Bottom-Up Approach**: Builds DAG structure starting from leaf nodes
- **Linear DAG Focus**: Specialized for linear structural equation models
- **Iterative Construction**: Builds DAG structure iteratively from bottom to top
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

## References

### Key Papers

**BUILD Algorithm:**
```bibtex
@article{ajourlou2024build,
  title={BUILD: Bottom-Up Inference of Linear DAGs},
  author={Ajorlou, Hamed and Rey, Samuel and García Marques, Antonio and Mateos, Gonzalo},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

**Baseline Methods:**

**DAGMA:**
```bibtex
@article{bello2022dagma,
  title={DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization},
  author={Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={8226--8239},
  year={2022}
}
```

**CoLiDE:**
```bibtex
@article{zhang2023colide,
  title={CoLiDE: Collaborative Linear DAG Estimation},
  author={Zhang, Xinyu and Zhang, Yujia and Zhang, Kun and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

**NOTEARS:**
```bibtex
@article{zheng2018dags,
  title={DAGs with NO TEARS: Continuous optimization for structure learning},
  author={Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and others},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

**Graphical Lasso:**
```bibtex
@article{friedman2008sparse,
  title={Sparse inverse covariance estimation with the graphical lasso},
  author={Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert},
  journal={Biostatistics},
  volume={9},
  number={3},
  pages={432--441},
  year={2008},
  publisher={Oxford University Press}
}
```

### Additional References

For more information on causal discovery and DAG learning, see:

- **Causal Discovery Methods**: [Spirtes et al., 2000](https://doi.org/10.7551/mitpress/1754.001.0001)
- **Linear Structural Equation Models**: [Bollen, 1989](https://doi.org/10.1002/9781118619179)
- **Graphical Models**: [Koller & Friedman, 2009](https://mitpress.mit.edu/9780262013192/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For local development, you can work with the ignored files without affecting the repository.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{build2024,
  author = {Ajorlou, Hamed and Rey, Samuel and García Marques, Antonio and Mateos, Gonzalo},
  title = {BUILD: Bottom-Up Inference of Linear DAGs},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/hamedajorlou/BUILD}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
