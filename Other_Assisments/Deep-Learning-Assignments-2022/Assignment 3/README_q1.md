# Assignment 3 — Question 1

Author: Mohammad taha majlesi - 810101504

Overview
--------
Compare the performance of a multilayer perceptron (MLP) and a convolutional neural network (CNN) with roughly the same number of parameters on an image classification task. The notebook explores regularization strategies (dropout vs BlockDropout) and architectural choices (factorized kernels).

Notebooks and files
-------------------
- `q1.ipynb` — experiment code, model definitions, training loops and evaluation.
- `Figures/` — TensorBoard snapshots and plots saved by the notebook.

How to run
----------
1. Create and activate a Python virtualenv and install dependencies.
2. Open `q1.ipynb` and run all cells. Use a GPU runtime if available for faster training.

Notes
-----
- Standard convolutional dropout is often less effective than structured variants (e.g., BlockDropout) for conv layers due to spatial correlation in activations. The notebook demonstrates this empirically.
