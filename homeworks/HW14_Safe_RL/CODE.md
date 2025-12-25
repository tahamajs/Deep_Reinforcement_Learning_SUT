HW14 Safe RL — Code & Notebook Usage

This short guide explains how to run the provided notebook implementations and where to find the key classes and helpers.

Key files

- Notebook: `homeworks/HW14_Safe_RL/code/HW14_Notebook.ipynb`
  - Contains classes: `PPOLagrangian`, `CPOAgent`, `SafetyLayer`
  - Training helpers: `train_ppo_lagrangian`, `train_cpo`
  - Evaluation helper: `evaluate_agent` (saves a short video and prints mean reward & cost)

Recommended quick run (interactive)

1. Activate your venv and install requirements:
   ```bash
   python -m pip install -r homeworks/HW14_Safe_RL/requirements.txt
   ```
2. Launch Jupyter Lab / Notebook:
   ```bash
   jupyter lab
   ```
3. Open `HW14_Notebook.ipynb` and run cells in order. Suggested first config:
   - Use `SafeCartPoleEnv` (or a light safe env), set `num_episodes=50`, `batch_size_steps=1024`.

Minimal example (pseudocode to run inside notebook):

```python
from HW14_Notebook import PPOLagrangian, SafeCartPoleEnv, train_ppo_lagrangian, evaluate_agent

env = SafeCartPoleEnv()
agent = PPOLagrangian(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], cost_limit=1.0)
train_ppo_lagrangian(env, agent, num_episodes=100)
evaluate_agent(agent.policy, env, num_episodes=3, save_path='./hw14_eval.mp4')
```

Notes

- The SafetyLayer provided is a simple projection-based shield designed for educational experiments. For production/robotics, replace with a QP-based projection or verified shield.
- The CPO implementation in the notebook is a simplified, practical variant suitable for experiments; it is not a drop-in replacement for the full ICML CPO codebase.
- If using MuJoCo environments, ensure proper MuJoCo setup (GL backend, licenses if required).

Contact

- For questions about running the code or desired integrations (QP shield, verified SMT check), reply in the issue or ask here and I can add example cells.






