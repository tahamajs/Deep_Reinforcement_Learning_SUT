import numpy as np
import torch
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime
# Import all the implementations
from bcq_implementation import BCQ, VAE, PerturbationNetwork
from cql_implementation import CQL, QNetwork, PolicyNetwork
from iql_implementation import IQL, VNetwork
from mopo_implementation import MOPO, DynamicsModel
# Create results directory
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_offline_dataset(env, num_episodes=100, max_steps=200):
    """Generate a simple offline dataset using a random policy"""
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    for _ in range(num_episodes):
        # FIX: Unpack the tuple returned by env.reset()
        state, _ = env.reset()  # Corrected line
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Convert state to a 1D array of length 4
            states.append(np.array(state, dtype=np.float32).flatten())
            actions.append(np.array(action, dtype=np.float32))
            rewards.append(reward)
            next_states.append(np.array(next_state, dtype=np.float32).flatten())
            dones.append(float(terminated or truncated))
            
            state = next_state
            if terminated or truncated:
                break
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    
    return states, actions, rewards, next_states, dones

def train_and_evaluate(algorithm_name, agent, dataset, env, num_episodes=100):
    """Train an agent and evaluate its performance"""
    print(f"\n{'='*50}")
    print(f"Training {algorithm_name} agent")
    print(f"{'='*50}")
    
    # Convert dataset to tensors (already tensors, so just use them directly)
    states = dataset["states"]
    actions = dataset["actions"].unsqueeze(1)  # [batch, 1]
    rewards = dataset["rewards"]
    next_states = dataset["next_states"]
    dones = dataset["dones"]

    # Training loop
    train_losses = {"q_loss": [], "policy_loss": []}
    for epoch in range(num_episodes):
        # Sample a batch
        batch = (states, actions, rewards, next_states, dones)
        
        # Initialize losses for BCQ
        losses = {}
        
        # Train step - handle IQL specially
        if algorithm_name == "BCQ":
            # BCQ doesn't have a standard train_step
            pass
        elif algorithm_name == "IQL":
            # IQL expects separate arguments
            losses = agent.train_step(states, actions, rewards, next_states, dones)
        else:
            # For CQL and others that expect a tuple
            losses = agent.train_step(batch)
        
        # Print progress
        if epoch % 10 == 0:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()]) if losses else "N/A"
            print(f"Epoch {epoch}: {loss_str}")
    
    # Define policy function based on algorithm type
    if algorithm_name == "BCQ":
        # For discrete action spaces (CartPole), use argmax
        def policy_fn(state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = agent.select_action(state_tensor)
            
            # Handle both tensor and numpy array outputs
            if isinstance(action_tensor, np.ndarray):
                action_tensor = torch.tensor(action_tensor)
            elif not isinstance(action_tensor, torch.Tensor):
                raise TypeError(f"Unexpected action type: {type(action_tensor)}")
                
            action = torch.argmax(action_tensor).item()
            return int(action)  # Ensure discrete action (0 or 1)
    else:
        # For continuous action spaces (other algorithms), convert to discrete for CartPole
        def policy_fn(state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            try:
                action_tensor = agent.select_action(state_tensor)
            except AttributeError:
                # For CQL, use the policy attribute
                actions, _ = agent.policy.sample(state_tensor)
                action_tensor = actions
            
            # Convert continuous action to discrete (0 or 1) for CartPole
            action_value = action_tensor.detach().numpy()[0, 0]  # Get scalar value
            discrete_action = 1 if action_value >= 0 else 0
            return discrete_action
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(env, policy_fn, episodes=10)
    
    # Save training curves
    plt.figure(figsize=(10, 5))
    if "q_loss" in train_losses:
        plt.plot(train_losses["q_loss"], label="Q Loss")
    if "policy_loss" in train_losses:
        plt.plot(train_losses["policy_loss"], label="Policy Loss")
    plt.title(f"{algorithm_name} Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"results/{algorithm_name}_training_{timestamp}.png")
    plt.close()
    
    print(f"\n{algorithm_name} Evaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

def evaluate_policy(env, policy_fn, episodes=10, max_steps=200):
    """Evaluate a policy in the environment"""
    returns = []
    for _ in range(episodes):
        # FIX: Unpack the tuple returned by env.reset()
        state, _ = env.reset()  # Corrected line
        total_reward = 0.0
        for _ in range(max_steps):
            action = policy_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    
    return np.mean(returns), np.std(returns)

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create environment
    env = gym.make("CartPole-v1")
    
    # FOR CARTPOLE: We're using discrete actions as scalars (0 or 1), so action_dim = 1
    action_dim = 1  # FIXED: Use 1 instead of env.action_space.n
    
    # Generate offline dataset
    print("Generating offline dataset...")
    states, actions, rewards, next_states, dones = generate_offline_dataset(env, num_episodes=100)
    
    # Create dataset dictionary
    dataset = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones
    }
    
    # Convert to tensors for training
    dataset = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in dataset.items()
    }
    
    # Initialize agents with action_dim=1
    bcq_agent = BCQ(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,  # FIXED: 1 instead of env.action_space.n
        latent_dim=16,
        lr=3e-4
    )
    
    cql_agent = CQL(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,  # FIXED
        alpha=1.0,
        lr=3e-4
    )
    
    iql_agent = IQL(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,  # FIXED
        expectile=0.7,
        temperature=0.05,
        lr=3e-4
    )
    
    mopo_agent = MOPO(
        state_dim=env.observation_space.shape[0],
        action_dim=action_dim,  # FIXED
        ensemble_size=5,
        lambda_u=1.0,
        lr=1e-3
    )
    
    # Train BCQ
    print("\nTraining BCQ agent...")
    bcq_mean_reward, bcq_std_reward = train_and_evaluate("BCQ", bcq_agent, dataset, env)
    
    # Train CQL
    print("\nTraining CQL agent...")
    cql_mean_reward, cql_std_reward = train_and_evaluate("CQL", cql_agent, dataset, env)
    
    # Train IQL
    print("\nTraining IQL agent...")
    iql_mean_reward, iql_std_reward = train_and_evaluate("IQL", iql_agent, dataset, env)
    
    # Train MOPO dynamics model
    print("\nTraining MOPO dynamics model...")
    dataset_list = [
        (dataset["states"][i].numpy(), dataset["actions"][i].numpy(), 
        dataset["rewards"][i].numpy(), dataset["next_states"][i].numpy(), 
        dataset["dones"][i].numpy())
        for i in range(len(dataset["states"]))
    ]
    mopo_agent.train_dynamics(dataset_list, epochs=10)
    
    # Evaluate MOPO
    def mopo_policy_fn(state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = mopo_agent.select_action(state_tensor)
        action = action_tensor.detach().numpy()[0]
        return int(action)  # Convert float to integer
    
    mopo_mean_reward, mopo_std_reward = evaluate_policy(env, mopo_policy_fn, episodes=10)
    
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    print(f"BCQ: {bcq_mean_reward:.2f} ± {bcq_std_reward:.2f}")
    print(f"CQL: {cql_mean_reward:.2f} ± {cql_std_reward:.2f}")
    print(f"IQL: {iql_mean_reward:.2f} ± {iql_std_reward:.2f}")
    print(f"MOPO: {mopo_mean_reward:.2f} ± {mopo_std_reward:.2f}")
    
    # Save results to file
    with open(f"results/results_{timestamp}.txt", "w") as f:
        f.write(f"Offline RL Algorithm Comparison\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*50}\n")
        f.write(f"BCQ: {bcq_mean_reward:.2f} ± {bcq_std_reward:.2f}\n")
        f.write(f"CQL: {cql_mean_reward:.2f} ± {cql_std_reward:.2f}\n")
        f.write(f"IQL: {iql_mean_reward:.2f} ± {iql_std_reward:.2f}\n")
        f.write(f"MOPO: {mopo_mean_reward:.2f} ± {mopo_std_reward:.2f}\n")

if __name__ == "__main__":
    main()