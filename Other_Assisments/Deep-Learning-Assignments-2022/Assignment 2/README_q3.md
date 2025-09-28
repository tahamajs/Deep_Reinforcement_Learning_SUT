```markdown
# Assignment 2 — Question 3

Author: Mohammad taha majlesi - 810101504

Problem
-------
Train an MLP to classify hand-sign alphabet images into 25 classes. Compare training with different optimizers and with/without dropout.

Key files
---------
- `q3.ipynb` — notebook implementing MLP variants, training loops and evaluation.
- `Figures/q3_*` — sample images, loss/accuracy plots saved by the notebook.

How it was implemented
----------------------
- Data: Hand sign image dataset; images are flattened for MLP input.
- Models compared: MLP + Adam, MLP + SGD, and MLP + Adam with Dropout.
- Training: 20 epochs per experiment; track loss and accuracy per epoch.

Results summary
---------------
- Adam converged faster than SGD and produced smoother loss curves.
- Dropout reduced overfitting and gave the best validation accuracy in this experiment.

How to run
----------
1. Ensure image dataset is available in the notebook's expected path.
2. Open `q3.ipynb` and run all cells. Use GPU-backed runtime for faster training if available.

```
