## Visualization guide for CA12 (RA-U-OBAC)

This file describes how to generate publication-quality figures from the demo outputs and logs.

1. Run the demo to produce checkpoints and eval returns:

```bash
python -m paperAssignments.Assignments1-50.CA12.scripts.train_ra_u_obac --steps 2000
```

By default the demo writes checkpoints and an `eval_returns.csv` to:
`outputs/ca12_checkpoints/eval_returns.csv`

2. Generate plots (requires pandas, matplotlib, seaborn):

```bash
python paperAssignments/Assignments1-50/CA12/analysis/plot_results.py \
  --ckpt_dir outputs/ca12_checkpoints \
  --out_dir paperAssignments/Assignments1-50/CA12/pictures
```

3. Files produced:

- `paperAssignments/Assignments1-50/CA12/pictures/fig_eval_returns.png` — Eval return curve (raw + smoothed).
- `paperAssignments/Assignments1-50/CA12/pictures/fig_losses.png` — Training loss curves (if you provide a loss CSV).

4. Notebook (optional)

If you prefer a Jupyter workflow, create a notebook that:

- Loads `eval_returns.csv` with pandas.
- Calls the `plot_eval_returns` function or reproduces the plotting logic with seaborn.
- Saves figures via `plt.savefig('../pictures/fig_...png', dpi=300)`.

5. Including figures in the paper

Place the generated PNG files in `paperAssignments/Assignments1-50/CA12/pictures/` and reference them in `report.tex` (placeholders are already present).






