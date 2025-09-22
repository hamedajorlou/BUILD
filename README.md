# TopoGreedy: Topological Greedy Algorithm for Causal Discovery

TopoGreedy is an algorithm for causal discovery that combines topological ordering with sparse precision matrix estimation. This implementation provides efficient and scalable solutions for learning causal structures from observational data. Here we test different implementations of Graphical Lasso and we provide results for different types of graphs with various number of nodes.

## Overview

The TopoGreedy algorithm works by:
1. Estimating the sparse precision matrix using different Graphical Lasso implementations
2. Identifying leaf nodes in the causal graph through iterative analysis
3. Reconstructing the adjacency matrix representing the causal structure
4. Applying edge thresholding to remove weak connections

## Features

### Multiple Graphical Lasso Implementations
- **Original Implementation**: Standard graphical lasso with coordinate descent
- **FISTA**: Fast Iterative Shrinkage-Thresholding Algorithm for improved convergence
- **Trench-Block**: Optimized implementation using block matrix operations
- **Sklearn Integration**: Support for scikit-learn's GraphicalLasso implementation

### Algorithm Variants
- `TopoGreedy_org`: Original implementation with standard graphical lasso
- `TopoGreedy_Fista`: Implementation using FISTA optimization
- `TopoGreedy_TB`: Implementation with Trench-Block optimization
- `TopoGreedy_sk`: Implementation using scikit-learn's GraphicalLasso

### Key Features
- Efficient adjacency matrix estimation
- Automatic hyperparameter selection
- Support for different optimization strategies
- Robust handling of numerical instabilities
- Comprehensive logging and visualization options

## Installation

```bash
# Clone the repository
git clone https://github.com/hamedajorlou/TopoGreedy.git
cd TopoGreedy

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from TopoGreedy import TopoGreedy

# Initialize the model
model = TopoGreedy(verbose=True)

# Fit the model
W_est, Theta_est = model.fit(X, max_iter=2000, tol=1e-4)
```

### Advanced Usage

```python
# Using FISTA implementation
from TopoGreedy import TopoGreedy_Fista

model = TopoGreedy_Fista(verbose=True)
W_est, Theta_est = model.fit(
    X,
    max_iter=2000,
    tol=1e-4
)

# Using Trench-Block implementation
from TopoGreedy import TopoGreedy_TB

model = TopoGreedy_TB(verbose=True)
W_est, Theta_est = model.fit(
    X,
    max_iter=2000,
    tol=1e-4
)
```

## API Reference

### Class: TopoGreedy

#### Parameters
- `verbose` (bool): Enable/disable verbose output
- `dtype` (numpy.dtype): Data type for computations
- `seed` (int): Random seed for reproducibility

#### Methods
- `fit(X, max_iter=2000, tol=1e-4)`: Fit the model to data
  - `X`: Input data matrix of shape (n_samples, n_features)
  - `max_iter`: Maximum number of iterations
  - `tol`: Convergence tolerance
  - Returns: `(W_est, Theta_est)` - Estimated adjacency matrix and precision matrix

### Utility Functions
- `estimate_adjacency_matrix(theta_matrix, p, edge_threshold=0.2)`: Estimate adjacency matrix from precision matrix
- `compute_lambda(n, d)`: Compute regularization parameter
- `soft_threshold(x, threshold)`: Apply soft thresholding operator

## Technical Details

### Graphical Lasso Implementation
The implementation includes several variants of the graphical lasso algorithm:
1. **Original Implementation**: Uses coordinate descent with careful handling of numerical stability
2. **FISTA**: Implements the accelerated proximal gradient method
3. **Trench-Block**: Optimizes computation using block matrix operations

### Adjacency Matrix Estimation
The algorithm uses an iterative process to:
1. Identify leaf nodes in the graph
2. Update the precision matrix
3. Reconstruct the adjacency matrix
4. Apply thresholding to remove weak edges

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- scikit-learn >= 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{topogreedy2024,
  author = {Ajorlou, Hamed and Rey, Samuel and García Marques, Antonio and Mateos, Gonzalo},
  title = {TopoGreedy: Topological Greedy Algorithm for Causal Discovery},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/hamedajorlou/TopoGreedy}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 