```markdown
# Assignment 2 — Question 2

Author: Mohammad taha majlesi - 810101504

Problem
-------
Train a simple feed-forward neural network using PyTorch to predict home-team match results (win/draw/loss) from features such as team FIFA ranks and points.

Key files
---------
- `q2.ipynb` — notebook with data loading (pandas), preprocessing, model definition and training.
- `Figures/q2_*` — correlation matrices and train/validation loss & accuracy plots.

How it was implemented
----------------------
- Data: soccer match dataset (CSV loaded via pandas). Feature engineering includes encoding categorical teams and normalizing numeric features.
- Model: small MLP with three hidden layers (10, 20, 8) and softmax output for 3 classes.
- Training: cross-entropy loss, training for up to 100 epochs (notebook logs accuracy per epoch).
- Evaluation: accuracy on a held-out test set; plots for loss and accuracy are saved to `Figures/`.

How to run
----------
1. Create and activate a Python3 virtual environment and install dependencies.
2. Open `q2.ipynb` and run the cells. Ensure the dataset CSV (if external) is present or modify the path in the notebook.

Notes
-----
- Small datasets can benefit from cross-validation; consider stratified k-fold to get more reliable estimates.
- Class imbalance (if present) can be addressed with class weights or resampling.

```
