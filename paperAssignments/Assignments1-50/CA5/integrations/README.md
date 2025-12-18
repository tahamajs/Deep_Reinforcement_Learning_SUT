# LightZero / mu-zero Minimal Integration

This folder contains a minimal adapter and example showing how to map a policy+value model
to a MuZero/LightZero-style API and use PUCT MCTS for planning.

Files:

- `lightzero_adapter.py` — adapter exposing `initial_inference` and `recurrent_inference`.
- `../scripts/lightzero_run.py` — example planning loop that wraps a small model with the adapter
  and runs PUCT to choose actions in a toy 1D environment.

Usage:

1. Run the example:
   ```bash
   python paperAssignments/Assignments1-50/CA5/scripts/lightzero_run.py --sims 100 --horizon 8
   ```
2. Replace `SmallDiscreteModel` with your learned model, or wrap CrossHQ actor/critic with
   the `CrossHQMCTSAdapter` to produce priors and values from the CrossHQ modules.

Notes:

- This is a minimal scaffold for integration. For full LightZero / ma_muzero support,
  adapt the adapter to match the target project's exact API (initial_inference/recurrent_inference signatures,
  hidden state formats, embedding pipeline).
