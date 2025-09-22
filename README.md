# BUILD: Bottom-Up Inference of Linear DAGs

BUILD is a novel algorithm for causal discovery that uses a bottom-up approach to efficiently discover Directed Acyclic Graphs (DAGs) from observational data. This implementation provides state-of-the-art performance in learning causal structures with improved scalability and accuracy.

![BUILD Algorithm Overview](DAGprec.jpeg)

## Overview

The BUILD algorithm works by:
1. Starting from leaf nodes and building up the DAG structure
2. Using bottom-up inference to determine causal relationships
3. Iteratively building the DAG from bottom to top
4. Leveraging linear DAG assumptions for efficient discovery

---

## Features

### BUILD Algorithm
- **Deterministic**: introduces a deterministic algorithm to recover the structure of the DAG.
- **Linear DAG Focus**: Specialized for linear structural equation models.
- **Iterative Construction**: Builds DAG structure iteratively from bottom to top.
- **Robust to estimation errors**: Employing optional refreshing of the weights to curb estimation error propagation.

### Comprehensive Baselines
- **CoLiDE**: Both equal and non-equal variance implementations  
- **DAGMA**: Linear and non-negative variants with multiple optimization strategies  
- **Gao et al.**: Optimal estimation of Gaussian DAG models
- **Daskalasis et al.**: Learning Gaussian DAG Models without Condition Number Bounds
  
### Key Features
- Multiple optimization strategies (FISTA, coordinate descent, block operations)  
- Automatic hyperparameter selection  
- Robust handling of numerical instabilities  
- Comprehensive evaluation metrics  
- Support for different graph types and noise models  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hamedajorlou/BUILD.git
cd BUILD

# Install dependencies
pip install -r requirements.txt

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0


## References

### Key Papers

**BUILD Algorithm:**
```bibtex
@article{ajorlou2024build,
  title={BUILD: Bottom-Up Inference of Linear DAGs},
  author={Ajorlou, Hamed and Rey, Samuel and Mateos, Gonzalo and leus, Geert and García Marques, Antonio and},
  year={2025}
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
