```markdown
# Assignment 3 — Question 2

Author: Mohammad taha majlesi - 810101504

Overview
--------
This question implements knowledge distillation on CIFAR-10. A ResNet50 is fine-tuned (or linear-tuned) and used as a teacher to train a smaller ResNet18 student model via the distillation loss.

Notebooks and files
-------------------
- `q2.ipynb` — training and evaluation code, plots and saved artifacts.
- `Figures/` — loss/accuracy plots, classification report and confusion matrix.

How to run
----------
1. Ensure pretrained models (if required) are available or the notebook downloads them.
2. Open `q2.ipynb` and run with GPU for reasonable runtime.

Notes
-----
- The notebook reports experiments with temperature T=10 and alpha=0.5 as the best found hyperparameters.

```
