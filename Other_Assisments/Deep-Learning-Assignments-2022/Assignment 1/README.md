Assignment 1 — Deep Learning

Author: Mohammad taha majlesi - 810101504

Overview

This folder contains solutions and supporting materials for Assignment 1. The assignment is composed of three parts (P1–P3). Each part is implemented as a Jupyter notebook and focuses on a different topic:

- P1: Recommender System (matrix factorization)
- P2: Support Vector Machine (SVM) on a heart disease dataset
- P3: Regression (Linear Regression, Locally Weighted Linear Regression, KNN)

This README points you to part-specific READMEs that explain the goals, datasets, algorithms, and instructions for reproducing results.

Files

- `DL2022-HW1-P1.ipynb` — Recommender system notebook
- `DL2022-HW1-P2.ipynb` — SVM notebook
- `DL2022-HW1-P3.ipynb` — Regression notebook
- `data/` — folder with datasets used by the notebooks
- `README_P1.md`, `README_P2.md`, `README_P3.md` — detailed READMEs for each part

How to run

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required packages (example):

```bash
pip install -r requirements.txt
# If no requirements.txt is available, install common packages:
pip install numpy pandas matplotlib seaborn scikit-learn jupyter cvxopt
```

3. Start Jupyter and open the notebooks:

```bash
jupyter notebook
# or
jupyter lab
```

Reproducibility notes

- The notebooks are organized to run sequentially (top-to-bottom). If you modified cells, restart the kernel and run all cells from the top.
- Some parts use `cvxopt` (for quadratic programming) which can require a working BLAS/LAPACK installation on your system.

Contact

If you need clarifications, open an issue or contact the author (Mohammad taha majlesi).
