import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional
import torch
import matplotlib.pyplot as plt
import os

from src.agents import REINFORCEAgent, REINFORCEBaselineAgent, ActorCriticAgent, PPOAgent, ContinuousPPOAgent
from src.config import Config

def train_reinforce_agent(
    env_name: str = Config.ENV_NAME_DISCRETE,
    episodes: int = Config.REINFORCE_EPISODES,
    lr: float = Config.REINFORCE_LR,
    gamma: float = Config.REINFORCE_GAMMA,
) -> Dict[str, List[float]]:
    """Train REINFORCE agent"""

    print(f"Training REINFORCE on {env_name}")
    print("=" * 40)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim, lr, gamma)

    scores = []
    losses = []

    for episode in range(episodes):
        state, _ = env.reset(seed=Config.SEED)
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(log_prob, reward)
            episode_reward += reward
            state = next_state

        loss = agent.update_policy()
        losses.append(loss)
        scores.append(episode_reward)

        if (episode + 1) % Config.PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | Loss: {loss:.4f}"
            )

    env.close()
    return {"scores": scores, "losses": losses}


def train_reinforce_baseline_agent(
    env_name: str = Config.ENV_NAME_DISCRETE,
    episodes: int = Config.REINFORCE_BASELINE_EPISODES,
    lr_policy: float = Config.REINFORCE_BASELINE_LR_POLICY,
    lr_value: float = Config.REINFORCE_BASELINE_LR_VALUE,
    gamma: float = Config.REINFORCE_BASELINE_GAMMA,
) -> Dict[str, List[float]]:
    """Train REINFORCE with baseline agent"""

    print(f"Training REINFORCE+Baseline on {env_name}")
    print("=" * 45)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEBaselineAgent(state_dim, action_dim, lr_policy, lr_value, gamma)

    scores = []
    policy_losses = []
    value_losses = []

    for episode in range(episodes):
        state, _ = env.reset(seed=Config.SEED)
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, log_prob, reward)
            episode_reward += reward
            state = next_state

        policy_loss, value_loss = agent.update_policy()
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        scores.append(episode_reward)

        if (episode + 1) % Config.PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}"
            )

    env.close()
    return {"scores": scores, "policy_losses": policy_losses, "value_losses": value_losses}


def train_actor_critic_agent(
    env_name: str = Config.ENV_NAME_DISCRETE,
    episodes: int = Config.ACTOR_CRITIC_EPISODES,
    lr_actor: float = Config.ACTOR_CRITIC_LR_ACTOR,
    lr_critic: float = Config.ACTOR_CRITIC_LR_CRITIC,
    gamma: float = Config.ACTOR_CRITIC_GAMMA,
) -> Dict[str, List[float]]:
    """Train Actor-Critic agent"""

    print(f"Training Actor-Critic on {env_name}")
    print("=" * 35)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim, lr_actor, lr_critic, gamma)

    scores = []
    actor_losses = []
    critic_losses = []

    for episode in range(episodes):
        state, _ = env.reset(seed=Config.SEED)
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actor_loss, critic_loss = agent.update(
                state, action, reward, next_state, done, log_prob
            )

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            episode_reward += reward
            state = next_state

        scores.append(episode_reward)

        if (episode + 1) % Config.PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}"
            )

    env.close()
    return {
        "scores": scores,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }


def train_ppo_agent(
    env_name: str = Config.ENV_NAME_DISCRETE,
    episodes: int = Config.PPO_EPISODES,
    lr: float = Config.PPO_LR,
    gamma: float = Config.PPO_GAMMA,
    eps_clip: float = Config.PPO_EPS_CLIP,
    k_epochs: int = Config.PPO_K_EPOCHS,
) -> Dict[str, List[float]]:
    """Train PPO agent"""

    print(f"Training PPO on {env_name}")
    print("=" * 25)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)

    scores = []
    policy_losses = []
    value_losses = []

    episode_rewards = []
    episode_length = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=Config.SEED)
        episode_reward = 0
        done = False

        while not done:
            action, log_prob, state_value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition((state, action, log_prob, reward, done))

            episode_reward += reward
            state = next_state
            episode_length += 1

            # If memory is full or episode ends, update policy
            # This condition needs careful adjustment based on PPO batching strategy
            # For simplicity, we'll update at the end of each episode for now
            # A more robust implementation would involve a fixed batch size from memory
            if done:
                policy_loss, value_loss = agent.update()
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                episode_rewards.append(episode_reward)
                episode_length = 0

        scores.append(episode_reward)

        if (episode + 1) % Config.PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
            print(
                f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f}"
            )

    env.close()
    return {"scores": scores, "policy_losses": policy_losses, "value_losses": value_losses}


def train_continuous_ppo_agent(
    env_name: str = Config.ENV_NAME_CONTINUOUS,
    episodes: int = Config.CONTINUOUS_PPO_EPISODES,
    lr: float = Config.CONTINUOUS_PPO_LR,
    gamma: float = Config.CONTINUOUS_PPO_GAMMA,
    eps_clip: float = Config.CONTINUOUS_PPO_EPS_CLIP,
    k_epochs: int = Config.CONTINUOUS_PPO_K_EPOCHS,
) -> Dict[str, List[float]]:
    """Train PPO agent for continuous control"""

    print(f"Training PPO on {env_name} (Continuous)")
    print("=" * 40)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ContinuousPPOAgent(state_dim, action_dim, lr, gamma, eps_clip, k_epochs)

    scores = []
    losses = []

    episode_rewards = []
    episode_length = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=Config.SEED)
        episode_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition((state, action, log_prob.item(), reward, done))

            episode_reward += reward
            state = next_state
            episode_length += 1

            # Update every N steps or at episode end
            # For continuous PPO, typically updates are done on batches of experience
            # For simplicity here, we collect and update when memory has enough transitions
            # A more common approach is to collect a fixed number of steps per agent and then update
            if len(agent.memory) >= Config.MAX_TIMESTEPS: # Example: update when memory is 'full' enough
                loss = agent.update()
                losses.append(loss)
                agent.memory = [] # Clear memory after update

        # Ensure update is called at episode end if memory is not empty
        if agent.memory:
            loss = agent.update()
            losses.append(loss)

        episode_rewards.append(episode_reward)
        scores.append(episode_reward)

        if (episode + 1) % Config.PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
            print(f"Episode {episode+1:4d} | Average Score: {avg_score:6.1f}")

    env.close()
    return {"scores": scores, "losses": losses}


def compare_policy_gradient_variants(
    env_name: str = Config.ENV_NAME_DISCRETE, episodes: int = Config.REINFORCE_EPISODES
) -> Dict[str, Dict]:
    """Compare different policy gradient variants"""

    print(f"Comparing Policy Gradient Variants on {env_name}")
    print("=" * 55)

    results = {}

    # Train REINFORCE
    print("\n1. Training REINFORCE...")
    results["REINFORCE"] = train_reinforce_agent(
        env_name, episodes=episodes, lr=Config.REINFORCE_LR, gamma=Config.REINFORCE_GAMMA
    )

    # Train REINFORCE + Baseline
    print("\n2. Training REINFORCE + Baseline...")
    results["REINFORCE_Baseline"] = train_reinforce_baseline_agent(
        env_name, episodes=episodes, lr_policy=Config.REINFORCE_BASELINE_LR_POLICY, lr_value=Config.REINFORCE_BASELINE_LR_VALUE, gamma=Config.REINFORCE_BASELINE_GAMMA
    )

    # Train Actor-Critic
    print("\n3. Training Actor-Critic...")
    results["Actor_Critic"] = train_actor_critic_agent(
        env_name, episodes=episodes, lr_actor=Config.ACTOR_CRITIC_LR_ACTOR, lr_critic=Config.ACTOR_CRITIC_LR_CRITIC, gamma=Config.ACTOR_CRITIC_GAMMA
    )

    # Train PPO
    print("\n4. Training PPO...")
    results["PPO"] = train_ppo_agent(
        env_name, episodes=episodes, lr=Config.PPO_LR, gamma=Config.PPO_GAMMA, eps_clip=Config.PPO_EPS_CLIP, k_epochs=Config.PPO_K_EPOCHS
    )

    return results


def plot_policy_gradient_comparison(
    results: Dict[str, Dict], save_path: Optional[str] = None
):
    """Plot comparison of policy gradient variants"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    methods = list(results.keys())
    colors = ["blue", "green", "red", "purple"]

    # Learning curves
    for method, color in zip(methods, colors):
        scores = results[method]["scores"]
        smoothed_scores = np.convolve(scores, np.ones(50) / 50, mode="valid")
        axes[0, 0].plot(smoothed_scores, label=method, color=color, linewidth=2)

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Smoothed Score")
    axes[0, 0].set_title("Learning Curves Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Final performance comparison
    final_scores = [np.mean(results[method]["scores"][-Config.PRINT_INTERVAL:]) for method in methods]
    axes[0, 1].bar(methods, final_scores, alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Final Average Score")
    axes[0, 1].set_title("Final Performance Comparison")
    axes[0, 1].grid(True, alpha=0.3)

    # Training stability (variance of scores)
    score_variances = [np.var(results[method]["scores"][-200:]) for method in methods]
    axes[1, 0].bar(
        methods, score_variances, alpha=0.7, edgecolor="black", color="orange"
    )
    axes[1, 0].set_ylabel("Score Variance")
    axes[1, 0].set_title("Training Stability (Lower is Better)")
    axes[1, 0].grid(True, alpha=0.3)

    # Sample efficiency (episodes to reach threshold)
    threshold = 195  # For CartPole
    sample_efficiencies = []

    for method in methods:
        scores = results[method]["scores"]
        episodes_to_threshold = len(scores)
        for i, score in enumerate(scores):
            if score >= threshold:
                episodes_to_threshold = i + 1
                break
        sample_efficiencies.append(episodes_to_threshold)

    axes[1, 1].bar(
        methods, sample_efficiencies, alpha=0.7, edgecolor="black", color="green"
    )
    axes[1, 1].set_ylabel("Episodes to Threshold")
    axes[1, 1].set_title("Sample Efficiency (Lower is Better)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def hyperparameter_sensitivity_analysis(
    env_name: str = Config.ENV_NAME_DISCRETE, episodes: int = Config.REINFORCE_EPISODES // 3
):
    """Analyze sensitivity to hyperparameters"""

    print("Policy Gradient Hyperparameter Sensitivity Analysis")
    print("=" * 55)

    # Test different learning rates
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    lr_results = {}

    print("\nTesting different learning rates...")
    for lr in learning_rates:
        print(f"  Learning Rate: {lr}")
        result = train_reinforce_baseline_agent(
            env_name, episodes=episodes, lr_policy=lr
        )
        lr_results[lr] = np.mean(result["scores"][-Config.PRINT_INTERVAL:])

    # Test different gamma values
    gamma_values = [0.9, 0.95, 0.99, 0.995]
    gamma_results = {}

    print("\nTesting different gamma values...")
    for gamma in gamma_values:
        print(f"  Gamma: {gamma}")
        result = train_reinforce_baseline_agent(env_name, episodes=episodes, gamma=gamma)
        gamma_results[gamma] = np.mean(result["scores"][-Config.PRINT_INTERVAL:])

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Learning rate sensitivity
    lrs = list(lr_results.keys())
    scores = list(lr_results.values())
    axes[0].plot(lrs, scores, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Final Average Score")
    axes[0].set_title("Learning Rate Sensitivity")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)

    # Gamma sensitivity
    gammas = list(gamma_results.keys())
    scores = list(gamma_results.values())
    axes[1].plot(gammas, scores, "o-", linewidth=2, markersize=8, color="green")
    axes[1].set_xlabel("Gamma (Discount Factor)")
    axes[1].set_ylabel("Final Average Score")
    axes[1].set_title("Discount Factor Sensitivity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(
        os.path.join("visualizations", "policy_gradient_hyperparameter_sensitivity.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    return {"learning_rates": lr_results, "gamma_values": gamma_results}


def curriculum_learning_demo(env_name: str = Config.ENV_NAME_DISCRETE, episodes: int = Config.REINFORCE_EPISODES):
    """Demonstrate curriculum learning with policy gradients"""

    print("Curriculum Learning Demonstration")
    print("=" * 40)

    # Create modified environments with different difficulties
    # NOTE: For a real curriculum, you would typically use custom Gym environments
    # or wrappers that allow dynamic modification of environment parameters.
    # Here, we simulate difficulty stages by training sequentially.
    env_configs = [
        {"name": "Easy", "episodes_per_stage": episodes // 3},
        {"name": "Medium", "episodes_per_stage": episodes // 3},
        {"name": "Hard", "episodes_per_stage": episodes // 3},
    ]

    curriculum_results = {}
    agent_for_curriculum = None

    for i, config in enumerate(env_configs):
        print(f"\nTraining on {config['name']} environment stage...")

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        if agent_for_curriculum is None:
            # Initialize agent for the first stage
            agent_for_curriculum = REINFORCEBaselineAgent(state_dim, action_dim)
        else:
            # For subsequent stages, re-use the trained agent
            # This simulates transferring knowledge.
            pass # Agent instance is carried over

        scores = []

        for episode in range(config["episodes_per_stage"]):
            state, _ = env.reset(seed=Config.SEED + i)
            episode_reward = 0
            done = False

            while not done:
                action, log_prob = agent_for_curriculum.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent_for_curriculum.store_transition(state, log_prob, reward)
                episode_reward += reward
                state = next_state

            agent_for_curriculum.update_policy()
            scores.append(episode_reward)

            if (episode + 1) % Config.PRINT_INTERVAL == 0:
                avg_score = np.mean(scores[-Config.PRINT_INTERVAL:])
                print(
                    f"  Stage {config['name']} Episode {episode+1:4d} | Average Score: {avg_score:6.1f}"
                )

        curriculum_results[config["name"]] = scores
        env.close()

    # Plot curriculum learning results
    fig, ax = plt.subplots(figsize=(10, 6))

    for env_type, scores in curriculum_results.items():
        smoothed_scores = np.convolve(scores, np.ones(20) / 20, mode="valid")
        ax.plot(smoothed_scores, label=env_type, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Score")
    ax.set_title("Curriculum Learning Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(os.path.join("visualizations", "curriculum_learning_demo.png"), dpi=300, bbox_inches="tight")
    plt.show()

    return curriculum_results
