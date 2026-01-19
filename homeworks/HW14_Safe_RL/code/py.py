import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import math
from copy import deepcopy
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import io
from PIL import Image

# ======================
# CONFIGURATION SECTION
# ======================
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ======================
# CORE IMPLEMENTATIONS
# ======================

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.std = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.clamp(self.std(x), min=1e-4)
        return mean, std

class SafetyLayer:
    def __init__(self, cost_fn, cost_limit, fallback=None, max_iters=10):
        self.cost_fn = cost_fn
        self.cost_limit = cost_limit
        self.fallback = fallback
        self.max_iters = max_iters
    def is_safe(self, state, action):
        try:
            c = float(self.cost_fn(state, action))
        except Exception:
            return False
        return c <= self.cost_limit
    def safe_action(self, state, action):
        if self.is_safe(state, action):
            return action
        if self.fallback is not None and self.is_safe(state, self.fallback):
            return self.fallback
        a = np.array(action, dtype=float)
        lo, hi = 0.0, 1.0
        safe_a = None
        for _ in range(self.max_iters):
            mid = (lo + hi) / 2.0
            cand = (1 - mid) * a
            if self.is_safe(state, cand):
                safe_a = cand
                hi = mid
            else:
                lo = mid
        if safe_a is None:
            zero = np.zeros_like(a)
            if self.is_safe(state, zero):
                return zero
            return action
        return safe_a

class PPOLagrangian:
    def __init__(
        self,
        state_dim,
        action_dim,
        policy_net=None,
        value_net=None,
        cost_value_net=None,
        cost_limit=10.0,
        clip=0.2,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        lr_value=1e-3,
        lr_cost_value=1e-3,
        lr_lambda=1e-3,
        initial_lambda=0.0,
    ):
        self.policy = policy_net or Policy(state_dim, action_dim)
        self.value = value_net or nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.cost_value = cost_value_net or nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)
        self.optimizer_cost_value = optim.Adam(
            self.cost_value.parameters(), lr=lr_cost_value
        )
        self.cost_limit = cost_limit
        self.lr_lambda = lr_lambda
        self.lambda_coef = float(initial_lambda)
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        mean, std = self.policy(state_t)
        dist = Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action.squeeze(0).cpu().numpy(), logp.squeeze(0)
    def compute_gae(self, rewards, values, dones, gamma=None, lam=None):
        gamma = self.gamma if gamma is None else gamma
        lam = self.lam if lam is None else lam
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        returns = adv + values
        return adv, returns
    def update(self, batch, epochs=10, batch_size=64):
        states = torch.FloatTensor(batch["states"]).to(DEVICE)
        actions = torch.FloatTensor(batch["actions"]).to(DEVICE)
        old_logps = torch.FloatTensor(batch["logps"]).to(DEVICE)
        returns = torch.FloatTensor(batch["returns"]).to(DEVICE)
        advs = torch.FloatTensor(batch["advs"]).to(DEVICE)
        cost_returns = torch.FloatTensor(batch["cost_returns"]).to(DEVICE)
        cost_advs = torch.FloatTensor(batch["cost_advs"]).to(DEVICE)
        N = states.shape[0]
        inds = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, N, batch_size):
                mb_inds = inds[start : start + batch_size]
                s_mb = states[mb_inds]
                a_mb = actions[mb_inds]
                old_logp_mb = old_logps[mb_inds]
                ret_mb = returns[mb_inds]
                adv_mb = advs[mb_inds]
                mean, std = self.policy(s_mb)
                dist = Normal(mean, std)
                logp_mb = dist.log_prob(a_mb).sum(dim=-1)
                ratio = torch.exp(logp_mb - old_logp_mb)
                surrogate1 = ratio * adv_mb
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_mb
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                cost_adv_mb = cost_advs[mb_inds]
                cost_loss = (ratio * cost_adv_mb).mean()
                lagrangian_loss = policy_loss + self.lambda_coef * cost_loss
                self.optimizer_policy.zero_grad()
                lagrangian_loss.backward()
                self.optimizer_policy.step()
                value_preds = self.value(s_mb).squeeze()
                value_loss = F.mse_loss(value_preds, ret_mb)
                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()
                cost_value_preds = self.cost_value(s_mb).squeeze()
                cost_value_loss = F.mse_loss(cost_value_preds, cost_returns[mb_inds])
                self.optimizer_cost_value.zero_grad()
                cost_value_loss.backward()
                self.optimizer_cost_value.step()

        mean_cost = np.mean(batch["costs"])  # Use actual costs, not discounted returns
        self.lambda_coef = max(
            0.0, self.lambda_coef + self.lr_lambda * (mean_cost - self.cost_limit)
        )
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "cost_value_loss": cost_value_loss.item(),
            "lambda": self.lambda_coef,
            "mean_cost": mean_cost,
        }

class CPOAgent:
    def __init__(self, state_dim, action_dim, cost_limit=10.0, delta_kl=0.01):
        self.policy = Policy(state_dim, action_dim)
        self.value_r = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.value_c = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.cost_limit = cost_limit
        self.delta_kl = delta_kl
        self.optimizer_value_r = optim.Adam(self.value_r.parameters(), lr=1e-3)
        self.optimizer_value_c = optim.Adam(self.value_c.parameters(), lr=1e-3)
    def compute_advantages(
        self, rewards, values, costs, cost_values, dones, gamma=0.99, lam=0.97
    ):
        rewards = np.array(rewards)
        costs = np.array(costs)
        dones = np.array(dones, dtype=float)
        values = np.array(values)
        cost_values = np.array(cost_values)
        adv_r = np.zeros_like(rewards)
        adv_c = np.zeros_like(costs)
        last_r = 0
        last_c = 0
        for t in reversed(range(len(rewards))):
            next_val_r = values[t + 1] if t + 1 < len(values) else 0
            next_val_c = cost_values[t + 1] if t + 1 < len(cost_values) else 0
            delta_r = rewards[t] + gamma * next_val_r * (1 - dones[t]) - values[t]
            delta_c = costs[t] + gamma * next_val_c * (1 - dones[t]) - cost_values[t]
            last_r = delta_r + gamma * lam * (1 - dones[t]) * last_r
            last_c = delta_c + gamma * lam * (1 - dones[t]) * last_c
            adv_r[t] = last_r
            adv_c[t] = last_c
        returns_r = adv_r + values
        returns_c = adv_c + cost_values
        return adv_r, adv_c, returns_r, returns_c
    def update(self, states, actions, rewards, costs, dones):
        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.FloatTensor(actions).to(DEVICE)
        with torch.no_grad():
            values_r = self.value_r(states_t).squeeze().cpu().numpy()
            values_c = self.value_c(states_t).squeeze().cpu().numpy()
        adv_r, adv_c, returns_r, returns_c = self.compute_advantages(
            rewards, values_r, costs, values_c, dones
        )
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)
        for _ in range(10):
            vpred = self.value_r(states_t).squeeze()
            loss_v = F.mse_loss(vpred, torch.FloatTensor(returns_r).to(DEVICE))
            self.optimizer_value_r.zero_grad()
            loss_v.backward()
            self.optimizer_value_r.step()
            vpred_c = self.value_c(states_t).squeeze()
            loss_vc = F.mse_loss(vpred_c, torch.FloatTensor(returns_c).to(DEVICE))
            self.optimizer_value_c.zero_grad()
            loss_vc.backward()
            self.optimizer_value_c.step()
        mean, std = self.policy(states_t)
        dist = Normal(mean, std)
        logp = dist.log_prob(actions_t).sum(dim=-1)
        surrogate_r = (logp * torch.FloatTensor(adv_r).to(DEVICE)).mean()
        surrogate_c = (logp * torch.FloatTensor(adv_c).to(DEVICE)).mean()
        J_c = float(np.sum(costs))
        if J_c <= self.cost_limit:
            loss = -surrogate_r
        else:
            loss = surrogate_c
        self.policy.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.policy.parameters():
                if p.grad is not None:
                    p.data += 0.01 * p.grad
        return {"policy_loss": loss.item(), "J_c": J_c}

def train_ppo_lagrangian(
    env, agent: PPOLagrangian, num_episodes=500, batch_size_steps=2048
):
    ep_rewards = []
    ep_costs = []
    obs = env.reset()
    for ep in range(num_episodes):
        states, actions, logps, rewards, costs, dones, values = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        steps = 0
        while steps < batch_size_steps:
            state = obs
            action, logp = agent.select_action(state)
            safe_action = shield.safe_action(state, action)
            next_obs, reward, terminated, truncated, info = env.step(safe_action)
            cost = info.get("cost", 0.0)
            states.append(state)
            actions.append(safe_action)
            logps.append(logp.cpu().item() if hasattr(logp, "cpu") else float(logp))
            rewards.append(reward)
            costs.append(cost)
            dones.append(terminated or truncated)
            obs = next_obs
            steps += 1
            if terminated or truncated:
                obs = env.reset()
        
        # FIX: Convert lists to numpy arrays BEFORE tensor conversion
        with torch.no_grad():
            vals = agent.value(torch.FloatTensor(np.array(states)).to(DEVICE)).squeeze().cpu().numpy()
            cost_vals = agent.cost_value(torch.FloatTensor(np.array(states)).to(DEVICE)).squeeze().cpu().numpy()
        
        advs, returns = agent.compute_gae(rewards, vals, dones)
        cost_advs, cost_returns = agent.compute_gae(costs, cost_vals, dones)
        batch = {
            "states": np.array(states),
            "actions": np.array(actions),
            "logps": np.array(logps),
            "returns": np.array(returns),
            "advs": np.array(advs),
            "cost_returns": np.array(cost_returns),
            "cost_advs": np.array(cost_advs),
                "costs": np.array(costs),  # ADD THIS LINE

        }
        stats = agent.update(batch)
        ep_rewards.append(sum(rewards))
        ep_costs.append(sum(costs))
        if ep % 10 == 0:
            print(
                f"Ep {ep}: reward={np.mean(ep_rewards[-10:]) if ep_rewards else 0:.2f}, cost={np.mean(ep_costs[-10:]) if ep_costs else 0:.2f}, lambda={stats.get('lambda'):.2f}"
            )
    return ep_rewards, ep_costs

    
def train_cpo(env, agent: CPOAgent, num_episodes=500):
    rewards_hist = []
    costs_hist = []
    for ep in range(num_episodes):
        states, actions, rewards, costs, dones = [], [], [], [], []
        obs = env.reset()
        done = False
        while not done:
            mean, std = agent.policy(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
            action = Normal(mean, std).sample().squeeze(0).detach().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)
            cost = info.get("cost", 0.0)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            costs.append(cost)
            dones.append(terminated or truncated)
            obs = next_obs
            done = terminated or truncated
        stats = agent.update(np.array(states), np.array(actions), rewards, costs, dones)
        rewards_hist.append(sum(rewards))
        costs_hist.append(sum(costs))
        if ep % 10 == 0:
            print(
                f"CPO Ep {ep}: reward={rewards_hist[-1]:.2f}, cost={costs_hist[-1]:.2f}"
            )
    return rewards_hist, costs_hist

def evaluate_agent(
    agent_policy, env, num_episodes=5, max_steps=1000, save_path="./eval.mp4"
):
    frames = []
    rewards = []
    costs = []
    # Create safety layer for evaluation
    cost_fn = lambda s, a: np.sum(a**2) * 0.1
    shield = SafetyLayer(cost_fn=cost_fn, cost_limit=5.0, fallback=[0.0, 0.0])
    
    for ep in range(num_episodes):
        obs = env.reset()
        ep_r = 0.0
        ep_c = 0.0
        for t in range(max_steps):
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mean, std = agent_policy(state_t)
                action = mean.cpu().numpy().squeeze(0)
            safe_action = shield.safe_action(obs, action)  # Apply safety layer
            obs, r, terminated, truncated, info = env.step(safe_action)
            ep_r += r
            ep_c += info.get("cost", 0.0)
            if ep == 0:
                frames.append(env.render())
            if terminated or truncated:
                break
        rewards.append(ep_r)
        costs.append(ep_c)
    if frames:
        imageio.mimsave(save_path, frames, fps=20)
    print(f"Eval mean reward: {np.mean(rewards):.2f}, mean cost: {np.mean(costs):.2f}")
    return rewards, costs


# ======================
# EXAMPLE ENVIRONMENT
# ======================
class SimpleEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.max_steps = 100
        self.reset()
        
    def reset(self):
        self.state = np.random.rand(self.state_dim)
        self.goal = np.array([0.5, 0.5, 0.5, 0.5])
        self.step_count = 0
        return self.state
        
    def step(self, action):
        # FIX: Apply action ONLY to first 2 dimensions of state
        self.state[:2] = np.clip(self.state[:2] + action * 0.1, 0, 1)
        
        # Reward: Move toward goal
        reward = -np.linalg.norm(self.state - self.goal)
        
        # Cost: Energy consumption
        cost = np.sum(action**2) * 0.1
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.state, reward, done, done, {"cost": cost}
    
    def render(self):
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(self.state)), self.state, color='skyblue')
        plt.title('State Visualization')
        plt.xlabel('State Dimension')
        plt.ylabel('Value')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return np.array(Image.open(buf))

        
        
# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Initialize environment
    env = SimpleEnv()
    
    # Create agent
    agent = PPOLagrangian(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        clip=0.2,
            cost_limit=15.0,  

        gamma=0.99,
        lr=1e-4,  # Reduced from 3e-4
        lr_value=5e-4,  # Reduced from 1e-3
        lr_cost_value=5e-4,  # Reduced from 1e-3
        lr_lambda=1e-3,
        initial_lambda=0.0
    )
    cost_fn = lambda s, a: np.sum(a**2) * 0.1
    shield = SafetyLayer(
        cost_fn=cost_fn,
        cost_limit=5.0,
        fallback=[0.0, 0.0]
    )

    # Train agent
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    rewards, costs = train_ppo_lagrangian(
        env=env,
        agent=agent,
        num_episodes=200,
        batch_size_steps=2048
    )
    
    # Save training metrics as images
    print("\n" + "="*50)
    print("Saving Training Metrics...")
    print("="*50)
    
    # Plot and save rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, 'b-', alpha=0.7, label='Episode Reward')
    plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), 'r-', label='10-ep Avg')
    plt.title('Training Rewards', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_rewards.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot and save costs
    plt.figure(figsize=(12, 6))
    plt.plot(costs, 'g-', alpha=0.7, label='Episode Cost')
    plt.plot(np.convolve(costs, np.ones(10)/10, mode='valid'), 'm-', label='10-ep Avg')
    plt.title('Training Costs', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_costs.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final policy
    torch.save(agent.policy.state_dict(), os.path.join(SAVE_DIR, "final_policy.pth"))
    
    # Evaluate agent
    print("\n" + "="*50)
    print("Evaluating Agent...")
    print("="*50)
    eval_rewards, eval_costs = evaluate_agent(
        agent_policy=agent.policy,
        env=env,
        num_episodes=5,
        max_steps=100,
        save_path=os.path.join(SAVE_DIR, "evaluation_video.mp4")
    )
    
    # Save evaluation metrics
    with open(os.path.join(SAVE_DIR, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Average Reward: {np.mean(eval_rewards):.4f}\n")
        f.write(f"Average Cost: {np.mean(eval_costs):.4f}\n")
        f.write(f"Reward Std: {np.std(eval_rewards):.4f}\n")
        f.write(f"Cost Std: {np.std(eval_costs):.4f}\n")
    
    print("\n" + "="*50)
    print("COMPLETION REPORT")
    print("="*50)
    print(f"Training completed successfully!")
    print(f"Results saved to: {SAVE_DIR}")
    print(f"Training rewards plot: {os.path.join(SAVE_DIR, 'training_rewards.png')}")
    print(f"Training costs plot: {os.path.join(SAVE_DIR, 'training_costs.png')}")
    print(f"Evaluation video: {os.path.join(SAVE_DIR, 'evaluation_video.mp4')}")
    print(f"Evaluation metrics: {os.path.join(SAVE_DIR, 'evaluation_metrics.txt')}")