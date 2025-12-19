This folder holds example figures used for the example report.

Files:
- `loss.png`: Training and validation loss vs epoch. Caption: "Training and validation loss across epochs (mean ± std across seeds if available). Config: `configs/example.yaml`."
- `predictions.png`: True vs predicted scatter (regression). Caption: "True vs predicted on validation set; dashed line is y=x. Report RMSE in caption."

To regenerate these example figures locally, run:

```bash
python scripts/make_placeholder_figures.py --out outputs/example_run/pictures
```

Replace these placeholder figures with your actual run artifacts before submission.
