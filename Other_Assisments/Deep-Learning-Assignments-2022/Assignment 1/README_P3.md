HW1 — Problem 3: Regression (Linear, LWLR, KNN)

Author: Mohammad taha majlesi - 810101504

Problem statement

Compare and implement different regression approaches on a simple one-dimensional dataset: standard Linear Regression, Locally Weighted Linear Regression (LWLR), and k-Nearest Neighbors (KNN). Evaluate models using mean squared error and visualize predictions.

Dataset

- `data/data01.csv` — Two-column dataset (x, y). The notebook drops an unnecessary index column if present.

Notebook structure and approach

1. Data split

   - Shuffle and split the dataset into 80% training and 20% testing.

2. Linear Regression

   - Fit using closed-form least squares (pseudo-inverse) and report train/test MSE.
   - Visualize the fitted line against train and test points.

3. Locally Weighted Linear Regression (LWLR)

   - For each query point x, build a diagonal weight matrix W where W\_{ii} = exp(-||x_i - x||^2 / (2 \* tau^2)).
   - Compute theta = (X^T W X)^{-1} X^T W y and predict for the query.
   - Evaluate test MSE and visualize.

4. k-Nearest Neighbors (KNN) regression
   - Implement a simple KNN regressor using Euclidean distance and average of neighbor targets.
   - Use cross-validation (grid search over k) to find a good k and report test MSE.

How to run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

2. Start Jupyter and open `DL2022-HW1-P3.ipynb`.
3. Run cells from top to bottom. Adjust `hyperparameter` (tau) for LWLR and `k` for KNN to see their effects on bias/variance.

Notes & tips

- LWLR is non-parametric and computationally expensive for large datasets because it solves a small weighted least-squares problem per query point. It's excellent for visualizing local fits on small datasets.
- KNN regression is simple, robust, and also local; choose k carefully (small k → low bias, high variance; large k → high bias, low variance).
- Plot predictions vs ground truth to inspect where models succeed or fail.

Extensions

- Add numerical cross-validation to select tau for LWLR and k for KNN.
- Compare results with polynomial regression or kernel ridge regression for additional insight.

Reproducibility

- Set the random seed at the notebook start for repeatable train/test splits.
- Use vectorized numpy operations to speed up the KNN implementation or rely on scikit-learn's `KNeighborsRegressor` for production experiments.
