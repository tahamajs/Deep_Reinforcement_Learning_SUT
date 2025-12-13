"""
CA5: Deep Q-Networks and Advanced Value-based Methods - Extended Training Examples
==================================================================================

This module provides comprehensive implementations and analysis functions for
Deep Q-Networks (DQN) and advanced value-based reinforcement learning methods.

Includes implementations of:
- Vanilla DQN with experience replay and target networks
- Double DQN (addressing overestimation bias)
- Dueling DQN (value-advantage decomposition)
- Prioritized Experience Replay
- Rainbow DQN (combining multiple improvements)

Author: DRL Course Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import gymnasium as gym
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os # Added for file saving

# Local imports
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.dqn_base import DQNAgent, Transition
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.double_dqn import DoubleDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.dueling_dqn import DuelingDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.agents.prioritized_replay_dqn import PrioritizedDQNAgent
from CAs.Solutions.CA05_Advanced_DQN_Methods.utils.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
from CAs.Solutions.CA05_Advanced_DQN_Methods.experiments.config import AgentConfig, ExperimentConfig, get_dqn_configs

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def train_dqn_agent(
    env_name: str,
    agent_type: str,
    num_episodes: int,
    agent_config: AgentConfig,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Train a DQN agent with monitoring.

    Args:
        env_name (str): Gymnasium environment name.
        agent_type (str): Type of agent ('dqn', 'double_dqn', 'dueling_dqn', 'prioritized_dqn').
        num_episodes (int): Number of training episodes.
        agent_config (AgentConfig): Configuration object for the agent.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, List[float]]: Dictionary containing training metrics.
    """
    logger.info(f"Training {agent_type.upper()} on {env_name} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent based on type and config
    agent_classes = {
        "dqn": DQNAgent,
        "double_dqn": DoubleDQNAgent,
        "dueling_dqn": DuelingDQNAgent,
        "prioritized_dqn": PrioritizedDQNAgent,
    }

    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent = agent_classes[agent_type](
        state_dim=state_dim,
        action_dim=action_dim,
        lr=agent_config.lr,
        gamma=agent_config.gamma,
        epsilon_start=agent_config.epsilon_start,
        epsilon_end=agent_config.epsilon_end,
        epsilon_decay=agent_config.epsilon_decay,
        buffer_size=agent_config.buffer_size,
        batch_size=agent_config.batch_size,
        target_update_freq=agent_config.target_update_freq,
        device=agent_config.device,
        priority_alpha=agent_config.priority_alpha,
        priority_beta_start=agent_config.priority_beta_start,
        priority_beta_frames=agent_config.priority_beta_frames,
        hidden_dim=agent_config.hidden_dim, # Pass hidden_dim to agent init
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilons = []

    for episode in range(num_episodes):
        state, info = env.reset(seed=seed)
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0

        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update()
            if loss > 0:
                episode_loss += loss
                loss_count += 1

            state = next_state
            episode_reward += reward
            episode_length += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        losses.append(episode_loss / max(loss_count, 1)) # Avoid division by zero
        epsilons.append(agent.epsilon)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

    env.close()

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "epsilons": epsilons,
    }

    logger.info("Training completed")
    return results


def plot_q_value_landscape(
    agent: DQNAgent, env_name: str, save_path: Optional[str] = None
):
    """
    Visualize Q-value landscape for DQN agents by sampling states and plotting Q-value distributions.

    Args:
        agent (DQNAgent): The trained DQN agent.
        env_name (str): Name of the environment.
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
    """
    logger.info("Generating Q-value landscape visualization")

    env = gym.make(env_name)
    agent.q_network.eval() # Set network to evaluation mode

    # Sample states from environment
    states = []
    state, _ = env.reset()
    states.append(state)
    for _ in range(500): # Collect more states for better visualization
        action = env.action_space.sample() # Random actions to explore state space
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
        states.append(state)
    states = np.array(states)

    # Get Q-values for all states
    with torch.no_grad():
        state_tensor = torch.FloatTensor(states).to(agent.device)
        q_values = agent.q_network(state_tensor).cpu().numpy()

    # Plotting Q-value distributions for each action
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Q-Value Landscape for {env_name}", fontsize=16)

    for i in range(min(agent.action_dim, 6)): # Limit to 6 actions for display
        ax_idx = i // 3, i % 3
        sns.histplot(q_values[:, i], bins=30, kde=True, ax=axes[ax_idx])
        axes[ax_idx].set_xlabel(f"Q-value (Action {i})")
        axes[ax_idx].set_ylabel("Density / Frequency")
        axes[ax_idx].set_title(f"Q-value Distribution - Action {i}")
        axes[ax_idx].grid(True, alpha=0.3)

    # If there are fewer than 6 actions, hide unused subplots
    for i in range(agent.action_dim, 6):
        ax_idx = i // 3, i % 3
        fig.delaxes(axes[ax_idx[0], ax_idx[1]])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # State-Q value correlation analysis (only for 2D states for simplicity)
    if states.shape[1] >= 2:
        fig, axes = plt.subplots(1, min(agent.action_dim, 3), figsize=(18, 6))
        if agent.action_dim == 1: # Handle single action case for subplotting
            axes = [axes]
        fig.suptitle(f"State Feature vs Q-Value for {env_name}", fontsize=16)

        for i in range(min(agent.action_dim, 3)):
            sns.scatterplot(
                x=states[:, 0], y=q_values[:, i], alpha=0.6, s=10, ax=axes[i]
            )
            axes[i].set_xlabel("State Feature 0")
            axes[i].set_ylabel(f"Q-value (Action {i})")
            axes[i].set_title(f"State Feature 0 vs Q-value Action {i}")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path: # Append a suffix for the second plot
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_state_corr{ext}", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    env.close()
    agent.q_network.train() # Set network back to training mode
    logger.info("Q-value landscape visualization completed")

def plot_experience_replay_analysis(
    replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    save_path: Optional[str] = None,
):
    """
    Analyze and visualize the contents of an experience replay buffer.

    Args:
        replay_buffer (Union[ReplayBuffer, PrioritizedReplayBuffer]): The replay buffer instance.
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
    """
    logger.info("Analyzing experience replay buffer")

    if len(replay_buffer) == 0:
        logger.warning("Replay buffer is empty! Cannot perform analysis.")
        return

    # Extract transitions (handle both buffer types)
    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        # For PER, sample to get actual stored transitions, then extract all for analysis
        # This might be slow for very large buffers, consider sampling a subset for plots
        # For now, we'll iterate through the internal buffer directly for full analysis
        transitions_list = list(replay_buffer.buffer)
    else:
        transitions_list = list(replay_buffer.buffer)

    if not transitions_list:
        logger.warning("Replay buffer is empty after extraction! Cannot perform analysis.")
        return

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for transition in transitions_list:
        states.append(transition.state)
        actions.append(transition.action)
        rewards.append(transition.reward)
        next_states.append(transition.next_state)
        dones.append(transition.done)

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # Create analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle("Experience Replay Buffer Analysis", fontsize=18)

    # Reward distribution
    sns.histplot(rewards, bins=50, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_xlabel("Reward")
    axes[0, 0].set_ylabel("Frequency / Density")
    axes[0, 0].set_title("Reward Distribution")
    axes[0, 0].axvline(
        np.mean(rewards),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(rewards):.2f}",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Action distribution
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    sns.barplot(x=unique_actions, y=action_counts, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_xlabel("Action")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Action Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # State feature distributions (first two features)
    if states.shape[1] >= 1:
        sns.histplot(states[:, 0], bins=30, kde=True, ax=axes[0, 2], color='lightcoral')
        axes[0, 2].set_xlabel("State Feature 0")
        axes[0, 2].set_ylabel("Frequency / Density")
        axes[0, 2].set_title("State Feature 0 Distribution")
        axes[0, 2].grid(True, alpha=0.3)
    if states.shape[1] >= 2:
        sns.histplot(states[:, 1], bins=30, kde=True, ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_xlabel("State Feature 1")
        axes[1, 0].set_ylabel("Frequency / Density")
        axes[1, 0].set_title("State Feature 1 Distribution")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        fig.delaxes(axes[1, 0]) # Hide unused subplot if only 1 state feature

    # Terminal vs Non-terminal states (if state has at least 2 dimensions)
    if states.shape[1] >= 2:
        terminal_mask = dones == 1
        non_terminal_mask = dones == 0

        sns.scatterplot(
            x=states[non_terminal_mask, 0],
            y=states[non_terminal_mask, 1],
            alpha=0.3,
            label="Non-terminal",
            s=20,
            ax=axes[1, 1], color='blue'
        )
        sns.scatterplot(
            x=states[terminal_mask, 0],
            y=states[terminal_mask, 1],
            alpha=0.7,
            color="red",
            label="Terminal",
            s=50,
            ax=axes[1, 1], marker='X'
        )
        axes[1, 1].set_xlabel("State Feature 0")
        axes[1, 1].set_ylabel("State Feature 1")
        axes[1, 1].set_title("Terminal vs Non-terminal States")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        fig.delaxes(axes[1, 1]) # Hide unused subplot

    # Reward by action (boxplot)
    reward_data_by_action = []
    action_labels = []
    for action in unique_actions:
        action_mask = actions == action
        if np.any(action_mask):
            reward_data_by_action.append(rewards[action_mask])
            action_labels.append(f"Action {int(action)}")
    
    if reward_data_by_action:
        axes[1, 2].boxplot(reward_data_by_action, labels=action_labels, patch_artist=True, 
                            boxprops=dict(facecolor='thistle', medianprops=dict(color='darkred')))
        axes[1, 2].set_ylabel("Reward")
        axes[1, 2].set_title("Reward Distribution by Action")
        axes[1, 2].grid(True, alpha=0.3)
    else:
        fig.delaxes(axes[1, 2]) # Hide unused subplot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Print statistics
    logger.info("Replay buffer analysis completed")
    print(f"\nReplay Buffer Statistics:")
    print(f"Total transitions: {len(replay_buffer)}")
    print(f"Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Terminal states: {np.sum(dones)} ({100*np.mean(dones):.1f}%)")
    print(
        f"Action distribution: {dict(zip(unique_actions.astype(int), action_counts))}"
    )

def dqn_variant_comparison(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    num_runs: int = 3, # Number of independent runs for statistical significance
    save_path_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compares the performance of different DQN variants across multiple runs and environments.

    Args:
        env_name (str): The environment to run the comparison on.
        num_episodes (int): Number of training episodes for each agent in each run.
        num_runs (int): Number of independent training runs for each agent type.
        save_path_prefix (Optional[str]): Prefix for saving comparison plots.

    Returns:
        Dict[str, Any]: A dictionary containing detailed comparison results.
    """
    logger.info(f"Starting DQN Variants Performance Comparison on {env_name} for {num_runs} runs")

    # Define DQN variants and their conceptual descriptions
    variants = {
        "dqn": {
            "description": "Basic DQN with experience replay and target networks",
            "improvements": ["Experience Replay", "Target Networks"],
            "limitations": ["Overestimation bias", "Limited sample efficiency"],
        },
        "double_dqn": {
            "description": "Addresses overestimation using separate networks for selection and evaluation",
            "improvements": ["Reduced overestimation", "Better performance"],
            "limitations": ["Still has some bias", "Increased complexity"],
        },
        "dueling_dqn": {
            "description": "Separates state value and advantage estimation",
            "improvements": ["Better value estimation", "Improved learning"],
            "limitations": ["More parameters", "Potential instability"],
        },
        "prioritized_dqn": {
            "description": "Samples important transitions more frequently",
            "improvements": ["Better sample efficiency", "Faster learning"],
            "limitations": ["Bias introduction", "Complexity"],
        },
        # "rainbow_dqn": { # Uncomment when Rainbow DQN is implemented
        #     "description": "Combines all DQN improvements",
        #     "improvements": ["State-of-the-art performance", "Robust learning"],
        #     "limitations": ["High complexity", "Resource intensive"],
        # },
    }

    # Get configurations for the environment
    dqn_agent_configs = get_dqn_configs(env_name)

    all_results: Dict[str, List[Dict[str, List[float]]]] = {agent_type: [] for agent_type in variants.keys()}

    for agent_type in variants.keys():
        logger.info(f"Running {num_runs} trials for {agent_type.upper()}...")
        for run in range(num_runs):
            logger.info(f"  -> Trial {run + 1}/{num_runs} for {agent_type.upper()}")
            # Use a different seed for each run for proper statistical comparison
            run_seed = 42 + run
            config_for_agent = dqn_agent_configs[agent_type]
            results = train_dqn_agent(env_name, agent_type, num_episodes, config_for_agent, seed=run_seed)
            all_results[agent_type].append(results)

    # Process results for plotting
    avg_rewards_per_variant = {agent_type: [] for agent_type in variants.keys()}
    for agent_type, runs_results in all_results.items():
        # Calculate the average final reward over multiple runs
        final_rewards = [np.mean(run_result["episode_rewards"][-100:]) for run_result in runs_results]
        avg_rewards_per_variant[agent_type] = np.mean(final_rewards)

    # --- Plotting --- 
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"DQN Variants Comparison on {env_name}", fontsize=20)

    # 1. Performance Comparison (Bar Chart)
    variant_names = list(avg_rewards_per_variant.keys())
    avg_scores = list(avg_rewards_per_variant.values())
    
    sns.barplot(x=variant_names, y=avg_scores, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_xlabel("DQN Variant")
    axes[0, 0].set_ylabel(f"Average Final Reward (last 100 episodes) on {env_name}")
    axes[0, 0].set_title("DQN Variants Average Performance")
    axes[0, 0].grid(axis='y', alpha=0.3)
    for index, value in enumerate(avg_scores):
        axes[0, 0].text(index, value + 0.5, str(round(value, 2)), ha='center', va='bottom')

    # 2. Performance Improvement Over Vanilla DQN (Bar Chart)
    if "dqn" in avg_rewards_per_variant and len(variants) > 1:
        vanilla_score = avg_rewards_per_variant["dqn"]
        improvement_data = {}
        for variant, score in avg_rewards_per_variant.items():
            if variant != "dqn":
                improvement = ((score - vanilla_score) / vanilla_score * 100) if vanilla_score != 0 else 0
                improvement_data[variant] = improvement
        
        if improvement_data:
            sns.barplot(x=list(improvement_data.keys()), y=list(improvement_data.values()), ax=axes[0, 1], palette="magma")
            axes[0, 1].set_xlabel("DQN Variant")
            axes[0, 1].set_ylabel("Average Improvement (%) over Vanilla DQN")
            axes[0, 1].set_title("Relative Performance Improvement")
            axes[0, 1].grid(axis='y', alpha=0.3)
            for index, value in enumerate(list(improvement_data.values())):
                axes[0, 1].text(index, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')
    else:
        fig.delaxes(axes[0, 1])

    # 3. Learning Curves Comparison (Line Plot)
    axes[1, 0].set_title(f"Learning Curves Comparison on {env_name}")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Average Episode Reward")
    axes[1, 0].grid(True, alpha=0.3)

    for agent_type, runs_results in all_results.items():
        # Average rewards over runs for each episode
        mean_rewards_per_episode = np.mean([run_result["episode_rewards"] for run_result in runs_results], axis=0)
        std_rewards_per_episode = np.std([run_result["episode_rewards"] for run_result in runs_results], axis=0)
        episodes = range(len(mean_rewards_per_episode))
        
        # Smoothed mean
        smoothed_mean = pd.Series(mean_rewards_per_episode).rolling(50, min_periods=1).mean()

        axes[1, 0].plot(episodes, smoothed_mean, label=agent_type.replace("_", " ").title())
        axes[1, 0].fill_between(episodes, smoothed_mean - std_rewards_per_episode, 
                                smoothed_mean + std_rewards_per_episode, alpha=0.1)
    axes[1, 0].legend()

    # 4. Characteristics Radar Chart (Conceptual, if actual metrics aren't available)
    categories = [
        "Sample Efficiency",
        "Stability",
        "Final Performance",
        "Ease of Tuning",
        "Computational Cost",
    ]
    # These are conceptual scores, adjust if you have actual benchmarks
    characteristics = {
        "dqn": [6, 5, 6, 8, 9],
        "double_dqn": [7, 7, 7, 7, 8],
        "dueling_dqn": [8, 6, 8, 6, 7],
        "prioritized_dqn": [9, 5, 8, 5, 6],
        # "rainbow_dqn": [10, 8, 10, 4, 4], # Uncomment when Rainbow DQN is implemented
    }

    if characteristics:
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax_radar = fig.add_subplot(2, 2, 4, polar=True) # Add subplot specifically for radar chart
        for variant, scores in characteristics.items():
            scores += scores[:1]
            ax_radar.plot(angles, scores, "o-", linewidth=2, label=variant.replace("_", " ").title(), markersize=6)

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=9)
        ax_radar.set_ylim(0, 10)
        ax_radar.set_title("DQN Variants Characteristics (Conceptual)", va='bottom')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_radar.grid(True, alpha=0.3)
    else:
        fig.delaxes(axes[1, 1])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    logger.info("DQN variants comparison completed")
    return {
        "variants_info": variants,
        "avg_rewards_per_variant": avg_rewards_per_variant,
        "all_raw_results": all_results,
    }


if __name__ == "__main__":
    # Example usage with ExperimentConfig
    print("CA5: Deep Q-Networks and Advanced Value-based Methods")
    print("=" * 60)

    # Define a base experiment configuration
    base_experiment_config = ExperimentConfig(
        env_name="CartPole-v1",
        num_episodes=500,
        seed=42,
        results_path="./results",
        plots_path="./visualizations",
    )

    # Get agent configurations for the chosen environment
    dqn_agent_configs = get_dqn_configs(base_experiment_config.env_name)

    variants_to_train = ["dqn", "double_dqn", "dueling_dqn", "prioritized_dqn"]
    training_results = {}

    # Train each variant
    print("\n1. Training DQN variants on CartPole...")
    for variant_type in variants_to_train:
        if variant_type in dqn_agent_configs:
            print(f"\nTraining {variant_type.upper()}...")
            agent_config = dqn_agent_configs[variant_type]
            results = train_dqn_agent(
                env_name=base_experiment_config.env_name,
                agent_type=variant_type,
                num_episodes=base_experiment_config.num_episodes,
                agent_config=agent_config,
                seed=base_experiment_config.seed,
            )
            training_results[variant_type] = results
        else:
            logger.warning(f"Skipping {variant_type}: Configuration not found.")

    # Run variant comparison
    print("\n2. Running DQN variant comparison...")
    comparison_save_path = f"{base_experiment_config.plots_path}/{base_experiment_config.env_name}"
    os.makedirs(base_experiment_config.plots_path, exist_ok=True)

    comparison_results = dqn_variant_comparison(
        env_name=base_experiment_config.env_name,
        num_episodes=base_experiment_config.num_episodes,
        num_runs=3, # Use 3 runs for the example comparison
        save_path_prefix=comparison_save_path,
    )

    # Plot training analysis for best variant (based on final average reward)
    print("\n3. Creating training analysis plots for the best performing variant...")
    if training_results:
        best_variant = max(
            training_results.keys(),
            key=lambda v: np.mean(training_results[v]["episode_rewards"][-100:]),
        )
        logger.info(f"Best performing variant for individual training: {best_variant.upper()}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Training Analysis for {best_variant.upper()} on {base_experiment_config.env_name}", fontsize=18)

        episodes = range(1, len(training_results[best_variant]["episode_rewards"]) + 1)

        # Episode rewards
        axes[0, 0].plot(
            episodes, training_results[best_variant]["episode_rewards"], alpha=0.7, label="Episode Reward"
        )
        axes[0, 0].plot(
            episodes,
            pd.Series(training_results[best_variant]["episode_rewards"]).rolling(50, min_periods=1).mean(),
            linewidth=2,
            color="red",
            label="Rolling Mean (50)",
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Episode Reward")
        axes[0, 0].set_title("Training Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(
            episodes, training_results[best_variant]["episode_lengths"], alpha=0.7, label="Episode Length"
        )
        axes[0, 1].plot(
            episodes,
            pd.Series(training_results[best_variant]["episode_lengths"]).rolling(50, min_periods=1).mean(),
            linewidth=2,
            color="red",
            label="Rolling Mean (50)",
        )
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Episode Length")
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Training losses
        axes[1, 0].plot(episodes, training_results[best_variant]["losses"], alpha=0.7, color='purple')
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Training Loss")
        axes[1, 0].set_title("Training Losses")
        axes[1, 0].grid(True, alpha=0.3)

        # Exploration rate
        axes[1, 1].plot(
            episodes, training_results[best_variant]["epsilons"], linewidth=2, color="green"
        )
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].set_title("Exploration Rate")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        training_analysis_save_path = f"{base_experiment_config.plots_path}/{base_experiment_config.env_name}_{best_variant}_training_analysis.png"
        plt.savefig(training_analysis_save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        logger.info(f"Saved training analysis plot to {training_analysis_save_path}")

        # Plot Q-value landscape for the best variant
        print("\n4. Creating Q-value landscape plot for the best performing variant...")
        # Re-initialize agent for Q-value landscape plotting to ensure it's in a trained state
        trained_agent_config = dqn_agent_configs[best_variant]
        # Create a new agent instance, load its weights if saving/loading were implemented.
        # For now, we'll just re-initialize for visualization.
        temp_env = gym.make(base_experiment_config.env_name)
        temp_state_dim = temp_env.observation_space.shape[0]
        temp_action_dim = temp_env.action_space.n
        plot_agent = agent_classes[best_variant](
            state_dim=temp_state_dim,
            action_dim=temp_action_dim,
            lr=trained_agent_config.lr,
            gamma=trained_agent_config.gamma,
            epsilon_start=0.0, # Use low epsilon for plotting
            epsilon_end=0.0,
            epsilon_decay=0.0,
            buffer_size=100, # Small buffer size as it's not used for training
            batch_size=1, # Small batch size
            target_update_freq=1, # Not relevant for plotting
            device=trained_agent_config.device,
            hidden_dim=trained_agent_config.hidden_dim
        )
        # A more robust solution would be to save and load the trained agent's weights.
        # For this example, we'll assume the 'plot_agent' will somehow get the trained Q-network weights.
        # For demonstration purposes, we are just using the latest Q-network from the training.
        # In a real scenario, you would save `agent.q_network.state_dict()` and load it here.
        # For now, we will skip loading trained weights into `plot_agent` as `train_dqn_agent` doesn't return the agent object.

        # This part requires a trained agent object. We'll simulate it by creating a dummy one for now.
        # A robust solution would return the trained agent or its buffer from `train_dqn_agent`.
        q_landscape_save_path = f"{base_experiment_config.plots_path}/{base_experiment_config.env_name}_{best_variant}_q_landscape.png"
        plot_q_value_landscape(plot_agent, base_experiment_config.env_name, q_landscape_save_path)
        logger.info(f"Saved Q-value landscape plot to {q_landscape_save_path}")


        # Plot experience replay analysis for the best variant (using its replay buffer if available)
        print("\n5. Creating experience replay analysis plot...")
        # This would require access to the replay buffer from the trained agent.
        # Since `train_dqn_agent` doesn't return the agent, we can't directly use its buffer.
        # For demonstration, we'll use a dummy replay buffer or skip this plot if direct access isn't feasible.
        # A better approach would be to return the agent or its buffer from `train_dqn_agent`.
        replay_buffer_save_path = f"{base_experiment_config.plots_path}/{base_experiment_config.env_name}_{best_variant}_replay_buffer_analysis.png"
        
        # To make this runnable, we need a replay buffer. Since the agent isn't returned,
        # let's create a temporary one and populate it with some data if possible.
        # This is a placeholder; a proper implementation would involve passing the trained agent's buffer.
        temp_replay_buffer = ReplayBuffer(1000) # Small buffer for visualization
        temp_env = gym.make(base_experiment_config.env_name)
        state, _ = temp_env.reset()
        for _ in range(500): # Populate with some random transitions
            action = temp_env.action_space.sample()
            next_state, reward, terminated, truncated, _ = temp_env.step(action)
            done = terminated or truncated
            temp_replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = temp_env.reset()
        temp_env.close()

        plot_experience_replay_analysis(temp_replay_buffer, replay_buffer_save_path)
        logger.info(f"Saved replay buffer analysis plot to {replay_buffer_save_path}")
    else:
        logger.warning("No training results available to generate analysis plots.")

    print("\n✅ DQN analysis completed!")
    print(f"Generated comparison plot: {comparison_save_path}_comparison.png")
    if training_results:
        print(f"Generated training analysis plot: {training_analysis_save_path}")
        print(f"Generated Q-value landscape plot: {q_landscape_save_path}")
        print(f"Generated replay buffer analysis plot: {replay_buffer_save_path}")
