import json
import sys

# Read the notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 7 - ReplayBuffer
replay_buffer_code = """class ReplayBuffer:
    \"\"\"A simple numpy replay buffer.\"\"\"

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        \"\"\"Initialize.\"\"\"
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0


    def store(self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        \"\"\"Store the transition in buffer.\"\"\"
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self) -> Dict[str, np.ndarray]:
        \"\"\"Randomly sample a batch of experiences from memory.\"\"\"
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )


    def __len__(self) -> int:
        return self.size
"""

nb['cells'][7]['source'] = [line + '\n' for line in replay_buffer_code.split('\n')[:-1]] + [replay_buffer_code.split('\n')[-1]]

# Add missing import for OUNoise
nb['cells'][9]['source'][13] = "        self.state = np.float64(0.0)\n"
nb['cells'][9]['source'].insert(0, "import copy\n\n")

# Save the updated notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully!")

