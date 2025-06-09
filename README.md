# TopoGreedy Algorithm

This repository contains an implementation of the TopoGreedy algorithm for causal discovery using graphical lasso optimization. The algorithm estimates the causal structure in data by combining topological ordering with sparse precision matrix estimation.

## Features

- Multiple implementations of Graphical Lasso:
  - Original implementation
  - FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
  - Trench-Block optimization
- Efficient adjacency matrix estimation
- Support for different optimization strategies

## Installation

```bash
git clone https://github.com/yourusername/TopoGreedy.git
cd TopoGreedy
pip install -r requirements.txt
```

## Usage

```python
from TopoGreedy import TopoGreedy

# Initialize the model
model = TopoGreedy(verbose=True)

# Fit the model
W_est, Theta_est = model.fit(X, max_iter=2000, tol=1e-4)
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
numpy
matplotlib
scikit-learn
```

## License

MIT License 