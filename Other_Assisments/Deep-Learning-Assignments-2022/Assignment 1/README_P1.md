HW1 — Problem 1: Recommender System (Matrix Factorization)

Author: Mohammad taha majlesi - 810101504

Problem statement

Design and implement a collaborative-filtering recommender system using matrix factorization. The task is to predict user preferences (implicit play counts) for artists and evaluate recommendation quality on held-out data.

Dataset

- `data/user_artists.dat`: user-artist interactions (implicit play counts)
- `data/artists.dat`: artist metadata (id, name)

The dataset is sparse: most user-item pairs are unobserved. The notebooks include preprocessing to create a user×item matrix where missing values are treated as zeros (implicit feedback).

Approach

1. Preprocessing

   - Merge artist metadata with play counts.
   - Create normalized play counts (scaled between 0 and 1) or other normalization strategies.
   - Build a user×artist matrix (`ratings`) where rows are users and columns are artists.

2. Train / Validation split

   - For users with enough observations, randomly hold out a small number of observed entries (DELETE_RATING_COUNT) as validation entries and set them to zero in the training matrix.

3. Matrix factorization model

   - Factorize the rating matrix R ≈ P^T Q, where P (latent factors × users) and Q (latent factors × items).
   - Optimize by stochastic gradient updates on observed entries (or closed-form SVD-like approximations), with L2 regularization.
   - Track RMSE on training and validation sets each epoch.

4. Evaluation
   - RMSE over observed validation entries.
   - Precision@K, Recall@K, NDCG@K for top-K recommendation quality (implementation can be added to the notebook).

Files & key functions

- `DL2022-HW1-P1.ipynb`
  - `train_test_split(ratings)`: splits train and validation by removing some observations for active users
  - `rmse(prediction, ground_truth)`: RMSE computed only on observed values of ground_truth
  - `Recommender` class: fits P and Q, provides predictions and per-user recommendation helpers

How to run

1. Ensure data files are present in the `data/` subfolder.
2. Launch Jupyter and open `DL2022-HW1-P1.ipynb`.
3. Run cells from the top. If you want deterministic results, set the random seed cell early in the notebook.

Tips and notes

- The simple SGD-style update in the notebook iterates over nonzero entries and updates the corresponding latent vectors. For large datasets, consider mini-batching or alternating least squares (ALS).
- Increasing the number of latent factors and epochs typically reduces training error but may require stronger regularization to avoid overfitting.
- Use sparse matrix libraries (scipy.sparse) for large-scale recommender systems.

Reproducibility & dependencies

- Python 3.8+ recommended
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn

Optional improvements

- Add Precision@K / Recall@K and NDCG metrics for top-K recommendation evaluation
- Add negative-sampling or Bayesian Personalized Ranking (BPR) loss for implicit feedback
- Add hyperparameter search for latent dimension, learning rate, and regularization
