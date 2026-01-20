import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import copy
from torch.distributions import Normal
import gym
from collections import deque, namedtuple
import random
import argparse
import matplotlib.pyplot as plt

# Fix for numpy bool8 deprecation warning
np.bool8 = np.bool_

##############################################
# Helper Functions
##############################################

def save(args, save_name, model, wandb, ep=None):
    """Save model weights to disk and wandb."""
    save_dir = "./trained_models/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep is None:
        torch.save(
            model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth"
        )
        # wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        # wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    """Collect random samples from the environment and add to dataset."""
    state, _ = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        # Fixed: Unpack 5 values from env.step() and handle done properly
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

def evaluate(env, policy, eval_runs=5):
    """Evaluate the policy on the environment."""
    reward_batch = []
    for i in range(eval_runs):
        state, _ = env.reset()
        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)
            # Fixed: Unpack 5 values from env.step() and handle done properly
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

##############################################
# Replay Buffer
##############################################

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            device (str): device to store tensors on
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

##############################################
# Actor and Critic Networks
##############################################

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=32,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
        device="cpu"
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Number of nodes in hidden layers
            log_std_min (float): Minimum log standard deviation
            log_std_max (float): Maximum log standard deviation
            device (str): Device to run on (cpu/cuda)
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device  # Store the device
        
        # Define an MLP (2-layers) as a shared backbone
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Define mu and log_std heads
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        """Evaluate the policy and return action and log probability."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)  # Use self.device
        
        # Calculate action
        action = torch.tanh(e)
        
        # Log-probability calculation using change-of-variable formula
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(
            1, keepdim=True
        )
        return action, log_prob
    
    def get_action(self, state):
        """Get action for a single state (stochastic)."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(self.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        """Get deterministic action for a single state."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()

class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Number of nodes in hidden layers
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##############################################
# CQL-SAC Agent
##############################################

class CQLSAC(nn.Module):
    """Interacts with and learns from the environment using CQL-SAC algorithm."""
    def __init__(
        self,
        state_size,
        action_size,
        tau,
        hidden_size,
        learning_rate,
        temp,
        cql_weight,
        target_action_gap,
        device,
    ):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            tau (float): soft update parameter for target networks
            hidden_size (int): number of nodes in hidden layers
            learning_rate (float): learning rate for optimizers
            temp (float): temperature parameter for CQL
            cql_weight (float): weight for CQL regularization
            target_action_gap (float): target action gap for CQL
            device (str): device to run on (cpu/cuda)
        """
        super(CQLSAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        self.clip_grad_param = 1
        self.target_entropy = -action_size  # -dim(A)
        
        # Entropy coefficient
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        
        # CQL parameters
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(
            params=[self.cql_log_alpha], lr=learning_rate
        )
        
        # Actor Network
        self.actor_local = Actor(state_size, action_size, hidden_size, device=device).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=learning_rate
        )
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if eval:
            action = self.actor_local.get_det_action(state)
        else:
            action = self.actor_local.get_action(state)
        return action.numpy()
    
    def calc_policy_loss(self, states, alpha):
        """Calculates the policy (actor) loss."""
        actions_pred, log_pis = self.actor_local.evaluate(states)
        # FIX: Remove .squeeze(0) to keep batch dimension
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1, q2)
        actor_loss = (alpha * log_pis - min_Q).mean()
        return actor_loss, log_pis


    def _compute_policy_values(self, obs_pi, obs_q):
        """Computes the policy values adjusted by the policy's entropy term."""
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)
        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)
        return qs1 - log_pis.detach(), qs2 - log_pis.detach()
    
    def _compute_random_values(self, obs, actions, critic):
        """Computes Q-values for random actions with a fixed uniform log-probability adjustment."""
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5**self.action_size)
        return random_values - random_log_probs
    
    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = -(
            self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()
        ).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().detach()
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            
            # Compute Q_target_next
            Q_target_next = (
                torch.min(Q_target1_next, Q_target2_next) - self.alpha * new_log_pi
            )
            
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_target_next * (1 - dones))
        
        # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, Q_targets.detach())
        critic2_loss = F.mse_loss(q2, Q_targets.detach())
        
        # ------------------------ CQL Addon ------------------------ #
        # Sample random actions uniformly from the action space
        random_actions = (
            torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1])
            .uniform_(-1, 1)
            .to(self.device)
        )
        
        # Repeat states to match the number of random actions
        num_repeat = int(random_actions.shape[0] / states.shape[0])
        temp_states = (
            states.unsqueeze(1)
            .repeat(1, num_repeat, 1)
            .view(states.shape[0] * num_repeat, states.shape[1])
        )
        
        # Compute Q-values for actions sampled from the current policy
        current_pi_values1, current_pi_values2 = self._compute_policy_values(
            temp_states, temp_states
        )
        
        # Compute Q-values for random actions
        random_values1 = self._compute_random_values(
            temp_states, random_actions, self.critic1
        ).reshape(states.shape[0], num_repeat, 1)
        random_values2 = self._compute_random_values(
            temp_states, random_actions, self.critic2
        ).reshape(states.shape[0], num_repeat, 1)
        
        # Reshape the current policy values to group per state
        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)
        
        # Concatenate random, current policy Q-values for log-sum-exp computation
        cat_q1 = torch.cat([random_values1, current_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2], 1)
        
        # Compute the CQL regularization loss
        cql1_scaled_loss = (
            (
                torch.logsumexp(cat_q1 / self.temp, dim=1).mean()
                * self.cql_weight
                * self.temp
            )
            - q1.mean()
        ) * self.cql_weight
        cql2_scaled_loss = (
            (
                torch.logsumexp(cat_q2 / self.temp, dim=1).mean()
                * self.cql_weight
                * self.temp
            )
            - q2.mean()
        ) * self.cql_weight
        
        # Combine critic losses with CQL regularization
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return (
            actor_loss.item(),
            alpha_loss.item(),
            critic1_loss.item(),
            critic2_loss.item(),
            cql1_scaled_loss.item(),
            cql2_scaled_loss.item(),
            current_alpha,
            0.0,  # Placeholder for cql_alpha_loss
            0.0,  # Placeholder for cql_alpha
        )
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

##############################################
# Configuration and Training
##############################################

def get_config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CQL-SAC Implementation")
    parser.add_argument(
        "--run_name", type=str, default="CQL-SAC", help="Run name, default: CQL-SAC"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        help="Gym environment name, default: Pendulum-v1",
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of episodes, default: 200"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=100_000,
        help="Maximal training dataset size, default: 100_000",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument(
        "--log_video",
        type=int,
        default=0,
        help="Log agent behaviour to wandb when set to 1, default: 0",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Saves the network every x episodes, default: 100",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size, default: 256"
    )
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for CQL")
    parser.add_argument(
        "--cql_weight", type=float, default=0.1, help="CQL weight" # Changed from 1.0 to 0.1
    )
    parser.add_argument("--target_action_gap", type=float, default=10, help="Target action gap")
    parser.add_argument("--tau", type=float, default=5e-3, help="Soft update parameter")
    args = parser.parse_args()
    return args

def train(config):
    """Train the CQL-SAC agent on the specified environment."""
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    
    # --- IMPROVEMENT 1: Better Hyperparameters for Stability ---
    # If config doesn't have these, set them manually here for stability
    if not hasattr(config, 'cql_weight'): config.cql_weight = 0.01  # Much lower for online
    if not hasattr(config, 'learning_rate'): config.learning_rate = 1e-4 # Lower LR for stability
    
    # Initialize agent
    agent = CQLSAC(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        tau=config.tau,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        temp=config.temperature,
        cql_weight=config.cql_weight,
        target_action_gap=config.target_action_gap,
        device=device,
    )
    
    # --- IMPROVEMENT 2: Add Learning Rate Schedulers ---
    # These will decay the learning rate over time to help convergence
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.actor_optimizer, T_max=config.episodes)
    critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.critic1_optimizer, T_max=config.episodes)
    critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.critic2_optimizer, T_max=config.episodes)
    
    # Initialize replay buffer
    buffer = ReplayBuffer(
        buffer_size=config.buffer_size, batch_size=config.batch_size, device=device
    )
    
    # Collect random samples for initial buffer
    print("Collecting random samples for initial buffer...")
    collect_random(env=env, dataset=buffer, num_samples=10000)
    
    # Evaluate initial policy
    eval_reward = evaluate(eval_env, agent)
    print(f"Initial Test Reward: {eval_reward:.2f}")
    print("-" * 50)
    
    # Training loop
    rewards_history = []  # Store per episode reward
    average10_history = []  # Store moving average
    average10 = deque(maxlen=10)
    
    for i in range(1, config.episodes + 1):
        state, _ = env.reset()
        episode_steps = 0
        rewards = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            rewards += reward
            episode_steps += 1
            
            # Learn from experience
            if len(buffer) > config.batch_size:
                experiences = buffer.sample()
                (
                    policy_loss,
                    alpha_loss,
                    bellmann_error1,
                    bellmann_error2,
                    cql1_loss,
                    cql2_loss,
                    current_alpha,
                    lagrange_alpha_loss,
                    lagrange_alpha,
                ) = agent.learn(experiences)
        
        # --- IMPROVEMENT 3: Step the Schedulers ---
        actor_scheduler.step()
        critic1_scheduler.step()
        critic2_scheduler.step()
        
        # Record results
        average10.append(rewards)
        rewards_history.append(rewards)
        average10_history.append(np.mean(average10))
        
        # Print progress
        print(
            f"Episode: {i} | Reward: {rewards:.2f} | Average10: {np.mean(average10):.2f} | Steps: {episode_steps}"
        )
        
        # Save model periodically
        if i % config.save_every == 0:
            save_dir = "./trained_models/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(
                agent.actor_local.state_dict(),
                save_dir + config.run_name + "_CQL-SAC_" + str(i) + ".pth",
            )
            print(f"Model saved at episode {i}")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config.save_fig), exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(rewards_history, label="Reward per Episode")
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(average10_history, label="Average Reward (10 episodes)", color="orange")
    plt.title("Moving Average Reward (10 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(config.save_fig, dpi=200)
    print(f"Training plot saved to {config.save_fig}")
    
    # Final evaluation
    final_eval_reward = evaluate(eval_env, agent)
    print(f"\nFinal Evaluation Reward: {final_eval_reward:.2f}")

##############################################
# Main Execution
##############################################

if __name__ == "__main__":
    # Parse command line arguments
    config = get_config()
    
    # Set default save figure path
    config.save_fig = "results/cql_sac_training.png"
    
    # Create trained_models directory if it doesn't exist
    os.makedirs("./trained_models", exist_ok=True)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Start training
    print(f"Starting training for {config.run_name} on {config.env} environment...")
    print(f"Configuration: {config}")
    train(config)
    