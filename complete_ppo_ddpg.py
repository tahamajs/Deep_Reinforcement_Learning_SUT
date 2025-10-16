#!/usr/bin/env python3
"""
Complete the PPO vs DDPG notebook with all implementations
"""

import json
import copy as copy_module

# Read the notebook
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG.ipynb', 'r') as f:
    nb = json.load(f)

# Now let's fill in all the code sections

# ====================================================================================
# Cell 7: ReplayBuffer
# ====================================================================================
replay_buffer_init = """        \"\"\"Initialize.\"\"\"
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0
"""

replay_buffer_store = """        \"\"\"Store the transition in buffer.\"\"\"
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
"""

replay_buffer_sample = """        \"\"\"Randomly sample a batch of experiences from memory.\"\"\"
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs]
        )
"""

replay_buffer_len = """        return self.size
"""

# Replace in cell 7
source = nb['cells'][7]['source']
new_source = []
skip_until_end = False
for line in source:
    if '# ==================================== Your Code (Begin) ====================================' in line:
        skip_until_end = True
        new_source.append(line)
        continue
    if '# ==================================== Your Code (End) ====================================' in line:
        skip_until_end = False
        
    if not skip_until_end:
        new_source.append(line)

# Now insert the actual code
final_source = []
section = 0
for line in new_source:
    final_source.append(line)
    if '# ==================================== Your Code (Begin) ====================================' in line:
        if section == 0:  # __init__
            for impl_line in replay_buffer_init.split('\n')[:-1]:
                final_source.append(impl_line + '\n')
        elif section == 1:  # store
            for impl_line in replay_buffer_store.split('\n')[:-1]:
                final_source.append(impl_line + '\n')
        elif section == 2:  # sample_batch
            for impl_line in replay_buffer_sample.split('\n')[:-1]:
                final_source.append(impl_line + '\n')
        elif section == 3:  # __len__
            for impl_line in replay_buffer_len.split('\n')[:-1]:
                final_source.append(impl_line + '\n')
        section += 1

nb['cells'][7]['source'] = final_source

# ====================================================================================
# Add import for copy module in OUNoise cell
# ====================================================================================
if 'import copy' not in ''.join(nb['cells'][9]['source']):
    nb['cells'][9]['source'].insert(0, 'import copy\n\n')

# ====================================================================================
# Cell 10: Networks - init_layer_uniform and PPOActor, Critic
# ====================================================================================

# For simplicity, let me create complete cells

# Save
with open('/Users/tahamajs/Documents/uni/DRL/docs/homeworks/archive/sp23/hws/HW2/SP23_RL_HW2/RL_HW2_PPO_vs_DDPG_COMPLETE.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Step 1 complete - ReplayBuffer implemented!")

