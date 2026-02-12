# projects/ – Standalone DRL Projects

Two main codebases are provided here. Each has its own `requirements.txt` and README for specifics.

## amasa/
Applied RL project (AMASA). See `projects/amasa/README.md` for dataset/setup details. Install via:
```bash
pip install -r projects/amasa/requirements.txt
```
Then follow the run instructions in that README (typically `python main.py` or notebooks).

## grad_rl/
Graduate-level research playground for DRL experiments. See `projects/grad_rl/README.md` and install:
```bash
pip install -r projects/grad_rl/requirements.txt
```
Use this tree for custom experiments and ablations; configs and scripts live alongside the code.

Add new projects under this directory with their own README and requirements to keep isolation between environments.
