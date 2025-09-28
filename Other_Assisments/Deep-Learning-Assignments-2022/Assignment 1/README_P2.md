HW1 — Problem 2: Support Vector Machine (SVM)

Author: Mohammad taha majlesi - 810101504

Problem statement

Use the SVM algorithm to classify patients based on clinical features into "at risk of heart disease" vs "not at risk." Explore preprocessing, train standard SVM models with different kernels (linear, polynomial, RBF) using scikit-learn, and implement an SVM solver based on the dual QP formulation using `cvxopt`.

Dataset

- `data/Heart_Disease_Dataset.csv` — Clinical features and a `target` column (0 = healthy, 1 = disease). See `Dataset_Description.pdf` (if present) for column definitions.

Notebook structure and approach

1. Exploratory Data Analysis (EDA)

   - Visualize distributions (age, sex) across target classes.
   - Check for missing values and class balance.

2. Outlier detection and removal

   - Use z-score (|z| > 3) on selected continuous columns to identify outliers and remove them.

3. Feature engineering and normalization

   - Normalize numerical features (e.g., age, resting bp, cholesterol, max heart rate, oldpeak) using standard scaling or min–max scaling depending on model needs.

4. Train/test split

   - Convert labels 0 → -1 and 1 → +1 where required (especially for custom QP SVM).
   - Use a standard 70/30 split (or scikit-learn's `train_test_split`).

5. SVM with scikit-learn

   - Train three SVMs with kernels: linear, polynomial, and RBF.
   - For RBF, tune `gamma` to reach high accuracy (the notebook suggests aiming for >= 90% on the validation/test set if possible).
   - Evaluate: Accuracy, Precision, Recall, F1-score (classification_report helper function in the notebook computes these).

6. Implement SVM dual (QP)
   - Implement kernel functions and build the Q matrix: Q\_{ij} = y_i y_j K(x_i, x_j).
   - Solve the QP using `cvxopt.solvers.qp` to obtain dual variables α.
   - Determine support vectors and compute bias/intercept and decision function.

Evaluation metrics

- Accuracy = (# correct) / N
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 _ (Precision _ Recall) / (Precision + Recall)

How to run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn cvxopt
```

2. Start Jupyter and open `DL2022-HW1-P2.ipynb`.
3. Execute cells from top to bottom. If using the QP implementation, ensure `cvxopt` is installed and functional (it may rely on BLAS/LAPACK on the system).

Tips & troubleshooting

- If `cvxopt` installation fails, consider installing with conda: `conda install -c conda-forge cvxopt` for easier linear algebra dependency resolution.
- Use feature scaling before training SVMs (especially for RBF/polynomial kernels) to improve numerical stability and performance.
- For imbalanced classes, consider reporting per-class metrics or use class weights in SVC (`class_weight='balanced'`).

Extensions

- Add cross-validation to tune C, gamma, and polynomial degree.
- Implement kernelized SVM for RBF/poly in the QP solver and compare support vectors with scikit-learn's solution.
- Try SMOTE or other resampling techniques if class imbalance hurts performance.
