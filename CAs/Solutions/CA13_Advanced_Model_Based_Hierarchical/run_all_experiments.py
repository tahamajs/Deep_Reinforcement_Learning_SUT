#!/usr/bin/env python3
"""
CA13: Comprehensive Experiment Runner
اجرای کامل تمام آزمایشات و تولید نتایج
"""

import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
from agents.model_free import DQNAgent
from agents.model_based import ModelBasedAgent
from agents.sample_efficient import SampleEfficientAgent
from agents.hierarchical import OptionsCriticAgent, FeudalAgent
from environments.grid_world import (
    SimpleGridWorld,
    MultiAgentGridWorld,
    StochasticGridWorld,
)
from models.world_model import VariationalWorldModel
from training_examples import (
    train_dqn_agent,
    train_model_based_agent,
    evaluate_agent,
    compare_agents,
    plot_training_curves,
    hyperparameter_sweep,
    set_seed,
    get_device,
)
from utils.visualization import (
    plot_learning_curves_comparison,
    plot_world_model_analysis,
    plot_multi_agent_analysis,
    create_summary_table,
    save_results,
)
from evaluation.advanced_evaluator import AdvancedRLEvaluator, IntegratedAdvancedAgent


def create_visualizations_folder():
    """ایجاد پوشه visualizations اگر وجود ندارد"""
    viz_folder = os.path.join(os.path.dirname(__file__), "visualizations")
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
        print(f"✓ پوشه visualizations ایجاد شد: {viz_folder}")
    return viz_folder


def run_single_agent_experiments():
    """اجرای آزمایشات تک عامل"""
    print("\n" + "=" * 80)
    print("🔬 آزمایشات تک عامل (Single Agent Experiments)")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device()

    # Create environment
    env = SimpleGridWorld(size=5, goal_reward=10.0, step_penalty=-0.1, max_steps=100)
    state_dim = 2
    action_dim = 4

    print(f"✓ Environment: SimpleGridWorld 5x5")
    print(f"✓ Device: {device}")

    # Initialize agents
    agents = {
        "DQN": DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=200,
        ),
        "Model-Based": ModelBasedAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-3,
        ),
        "Sample-Efficient": SampleEfficientAgent(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=128, lr=1e-3
        ),
    }

    # Training parameters
    num_episodes = 150
    eval_episodes = 10

    print(f"\n📊 آموزش {num_episodes} قسمت برای هر عامل...")

    # Train and compare agents
    results = compare_agents(
        env=env, agents=agents, num_episodes=num_episodes, eval_episodes=eval_episodes
    )

    # Create visualizations
    viz_folder = create_visualizations_folder()

    # Plot training curves
    plt.figure(figsize=(15, 10))
    plot_training_curves(
        results, save_path=os.path.join(viz_folder, "single_agent_training_curves.png")
    )

    # Create summary table
    summary_df = create_summary_table(results)
    print("\n📋 خلاصه نتایج:")
    print(summary_df.to_string(index=False))

    # Save results
    save_results(results, os.path.join(viz_folder, "single_agent_results.json"))

    return results


def run_hierarchical_experiments():
    """اجرای آزمایشات سلسله مراتبی"""
    print("\n" + "=" * 80)
    print("🏗️ آزمایشات سلسله مراتبی (Hierarchical RL Experiments)")
    print("=" * 80)

    # Setup
    set_seed(42)
    env = SimpleGridWorld(size=7, goal_reward=15.0, step_penalty=-0.1, max_steps=150)
    state_dim = 2
    action_dim = 4

    print(f"✓ Environment: SimpleGridWorld 7x7")

    # Initialize hierarchical agents
    hierarchical_agents = {
        "Options-Critic": OptionsCriticAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_options=4,
            hidden_dim=128,
            lr=1e-3,
        ),
        "Feudal": FeudalAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=16,
            hidden_dim=128,
            lr=1e-3,
            temporal_horizon=10,
        ),
    }

    # Training
    num_episodes = 100
    results = {}

    for name, agent in hierarchical_agents.items():
        print(f"\n🎯 آموزش {name}...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(150):
                if name == "Options-Critic":
                    action, option = agent.act(
                        obs, epsilon=max(0.1, 1.0 - episode / 100)
                    )
                else:
                    action = agent.act(obs, epsilon=max(0.1, 1.0 - episode / 100))

                next_obs, reward, done, _ = env.step(action)

                # Store experience for hierarchical agents
                experience = {
                    "state": obs,
                    "action": action,
                    "reward": reward,
                    "next_state": next_obs,
                    "terminated": done,
                }

                if name == "Options-Critic":
                    experience["option"] = option

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Update agent
            if episode % 10 == 0 and episode > 0:
                # Collect recent experiences for update
                recent_experiences = [experience]  # Simplified
                agent.update(recent_experiences)

            if episode % 20 == 0:
                avg_reward = (
                    np.mean(episode_rewards[-20:])
                    if len(episode_rewards) >= 20
                    else np.mean(episode_rewards)
                )
                print(f"  Episode {episode:3d}: Avg Reward = {avg_reward:6.2f}")

        results[name] = {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "training_steps": len(episode_rewards),
        }

    # Visualization
    viz_folder = create_visualizations_folder()
    plot_learning_curves_comparison(
        results, save_path=os.path.join(viz_folder, "hierarchical_results.png")
    )

    return results


def run_multi_agent_experiments():
    """اجرای آزمایشات چندعاملی"""
    print("\n" + "=" * 80)
    print("👥 آزمایشات چندعاملی (Multi-Agent Experiments)")
    print("=" * 80)

    # Setup
    set_seed(42)
    n_agents = 3
    env = MultiAgentGridWorld(
        size=7,
        n_agents=n_agents,
        goal_reward=10.0,
        collision_penalty=-2.0,
        max_steps=100,
    )

    print(f"✓ Environment: MultiAgentGridWorld {n_agents} agents")

    # Initialize multi-agent system
    from agents.model_free import MultiAgentDQN

    ma_system = MultiAgentDQN(
        n_agents=n_agents,
        state_dim=2 + 2 + 2 * (n_agents - 1),  # own_pos + own_goal + other_positions
        action_dim=5,  # 4 directions + stay
        hidden_dim=64,
        lr=1e-3,
        enable_communication=True,
    )

    # Training
    num_episodes = 100
    team_rewards = []
    individual_rewards = [[] for _ in range(n_agents)]
    collision_counts = []

    for episode in range(num_episodes):
        observations = env.reset()
        episode_team_reward = 0
        episode_individual_rewards = [0] * n_agents
        collision_count = 0

        for step in range(100):
            # Get actions from all agents
            actions = ma_system.act(observations, epsilon=max(0.1, 1.0 - episode / 100))

            # Step environment
            next_observations, rewards, done, info = env.step(actions)

            # Store experiences
            experiences = []
            for i in range(n_agents):
                experience = {
                    "state": observations[i],
                    "action": actions[i],
                    "reward": rewards[i],
                    "next_state": next_observations[i],
                    "done": done,
                }
                experiences.append(experience)

                # Check for collisions
                if i > 0 and any(
                    observations[i][:2] == observations[j][:2] for j in range(i)
                ):
                    collision_count += 1

            # Update multi-agent system
            ma_system.update(experiences)

            episode_team_reward += sum(rewards)
            for i in range(n_agents):
                episode_individual_rewards[i] += rewards[i]

            observations = next_observations

            if done:
                break

        team_rewards.append(episode_team_reward)
        for i in range(n_agents):
            individual_rewards[i].append(episode_individual_rewards[i])
        collision_counts.append(collision_count)

        if episode % 20 == 0:
            avg_team_reward = (
                np.mean(team_rewards[-20:])
                if len(team_rewards) >= 20
                else np.mean(team_rewards)
            )
            avg_collisions = (
                np.mean(collision_counts[-20:])
                if len(collision_counts) >= 20
                else np.mean(collision_counts)
            )
            print(
                f"  Episode {episode:3d}: Team Reward = {avg_team_reward:6.2f}, Collisions = {avg_collisions:.1f}"
            )

    # Prepare results
    ma_results = {
        "n_agents": n_agents,
        "team_rewards": team_rewards,
        "collision_counts": collision_counts,
    }

    for i in range(n_agents):
        ma_results[f"agent_{i}_rewards"] = individual_rewards[i]

    # Visualization
    viz_folder = create_visualizations_folder()
    plot_multi_agent_analysis(
        ma_results, save_path=os.path.join(viz_folder, "multi_agent_results.png")
    )

    return ma_results


def run_world_model_experiments():
    """اجرای آزمایشات مدل جهانی"""
    print("\n" + "=" * 80)
    print("🌍 آزمایشات مدل جهانی (World Model Experiments)")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device()

    # Create environment
    env = StochasticGridWorld(size=5, wind_prob=0.2, wind_strength=1)
    state_dim = 2
    action_dim = 4

    print(f"✓ Environment: StochasticGridWorld with wind")

    # Initialize world model
    world_model = VariationalWorldModel(
        obs_dim=state_dim, action_dim=action_dim, latent_dim=16, hidden_dim=64
    ).to(device)

    # Collect data for world model training
    print("📊 جمع‌آوری داده برای آموزش مدل جهانی...")

    dataset_size = 1000
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for _ in range(dataset_size):
        obs = env.reset()
        action = np.random.randint(0, action_dim)
        next_obs, reward, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)

    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations).to(device)
    action_tensor = torch.LongTensor(actions).to(device)
    reward_tensor = torch.FloatTensor(rewards).to(device)
    next_obs_tensor = torch.FloatTensor(next_observations).to(device)
    done_tensor = torch.BoolTensor(dones).to(device)

    # Train world model
    print("🎯 آموزش مدل جهانی...")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    batch_size = 64
    num_epochs = 50

    losses_history = {"total": [], "vae": [], "dynamics": [], "reward": [], "done": []}

    for epoch in range(num_epochs):
        epoch_losses = {"total": 0, "vae": 0, "dynamics": 0, "reward": 0, "done": 0}
        num_batches = 0

        # Mini-batch training
        for i in range(0, dataset_size, batch_size):
            end_idx = min(i + batch_size, dataset_size)

            batch_obs = obs_tensor[i:end_idx]
            batch_actions = action_tensor[i:end_idx]
            batch_rewards = reward_tensor[i:end_idx]
            batch_next_obs = next_obs_tensor[i:end_idx]
            batch_dones = done_tensor[i:end_idx]

            # Forward pass
            losses = world_model.compute_loss(
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
            )

            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            # Accumulate losses
            for key, loss in losses.items():
                epoch_losses[key] += loss.item()
            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses_history[key].append(epoch_losses[key])

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: Total Loss = {epoch_losses['total']:.4f}")

    # Test world model
    print("🧪 تست مدل جهانی...")

    # Generate imagined trajectory
    test_obs = env.reset()
    test_obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0).to(device)

    # Encode initial observation
    with torch.no_grad():
        mu, logvar = world_model.encode(test_obs_tensor)
        z_start = world_model.reparameterize(mu, logvar)

    # Generate imagined trajectory
    imagined_actions = [
        torch.randint(0, action_dim, (1,)).to(device) for _ in range(10)
    ]
    z_trajectory, imagined_rewards, imagined_dones = world_model.imagine_trajectory(
        z_start, imagined_actions, horizon=10
    )

    # Decode trajectory
    imagined_observations = world_model.decode_trajectory(z_trajectory)

    # Prepare results
    world_model_results = {
        "losses_history": losses_history,
        "imagined_trajectory": torch.cat(imagined_observations, dim=0).cpu().numpy(),
        "imagined_rewards": torch.cat(imagined_rewards, dim=0).cpu().numpy(),
        "final_losses": {k: v[-1] for k, v in losses_history.items()},
    }

    # Visualization
    viz_folder = create_visualizations_folder()
    plot_world_model_analysis(
        world_model_results,
        save_path=os.path.join(viz_folder, "world_model_results.png"),
    )

    return world_model_results


def run_comprehensive_evaluation():
    """اجرای ارزیابی جامع"""
    print("\n" + "=" * 80)
    print("📊 ارزیابی جامع (Comprehensive Evaluation)")
    print("=" * 80)

    # Setup
    set_seed(42)

    # Create multiple environments
    environments = [
        SimpleGridWorld(size=5),
        SimpleGridWorld(size=7),
        StochasticGridWorld(size=5, wind_prob=0.1),
    ]

    # Create agents
    agents = {
        "DQN": DQNAgent(state_dim=2, action_dim=4, hidden_dim=64, learning_rate=1e-3),
        "Model-Based": ModelBasedAgent(
            state_dim=2, action_dim=4, hidden_dim=64, learning_rate=1e-3
        ),
        "Sample-Efficient": SampleEfficientAgent(
            state_dim=2, action_dim=4, hidden_dim=64, lr=1e-3
        ),
    }

    # Initialize evaluator
    evaluator = AdvancedRLEvaluator(
        environments=environments,
        agents=agents,
        metrics=["sample_efficiency", "reward", "transfer"],
    )

    # Run comprehensive evaluation
    print("🔍 اجرای ارزیابی جامع...")
    results = evaluator.comprehensive_evaluation()

    # Generate report
    evaluator.generate_report()

    # Create plots
    viz_folder = create_visualizations_folder()
    evaluator.plot_results(
        save_path=os.path.join(viz_folder, "comprehensive_evaluation.png")
    )

    return results


def main():
    """تابع اصلی اجرای تمام آزمایشات"""
    print("🚀 شروع اجرای جامع آزمایشات CA13")
    print("=" * 80)

    start_time = time.time()
    all_results = {}

    try:
        # 1. Single Agent Experiments
        all_results["single_agent"] = run_single_agent_experiments()

        # 2. Hierarchical Experiments
        all_results["hierarchical"] = run_hierarchical_experiments()

        # 3. Multi-Agent Experiments
        all_results["multi_agent"] = run_multi_agent_experiments()

        # 4. World Model Experiments
        all_results["world_model"] = run_world_model_experiments()

        # 5. Comprehensive Evaluation
        all_results["comprehensive"] = run_comprehensive_evaluation()

        # Save all results
        viz_folder = create_visualizations_folder()
        save_results(
            all_results, os.path.join(viz_folder, "all_experiments_results.json")
        )

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("✅ تمام آزمایشات با موفقیت تکمیل شد!")
        print(f"⏱️ زمان کل اجرا: {total_time:.2f} ثانیه")
        print(f"📁 نتایج در پوشه: {viz_folder}")
        print("=" * 80)

        # Create final summary plot
        plt.figure(figsize=(16, 12))

        # Performance comparison across all experiments
        plt.subplot(2, 2, 1)
        experiment_names = list(all_results.keys())
        if "single_agent" in all_results:
            single_agent_final_rewards = []
            for agent_name, result in all_results["single_agent"].items():
                if "training" in result:
                    rewards = result["training"].get("rewards", [])
                    if rewards:
                        final_reward = (
                            np.mean(rewards[-20:])
                            if len(rewards) >= 20
                            else np.mean(rewards)
                        )
                        single_agent_final_rewards.append(final_reward)
            if single_agent_final_rewards:
                plt.bar(
                    ["DQN", "Model-Based", "Sample-Efficient"],
                    single_agent_final_rewards,
                    alpha=0.7,
                )
                plt.title("Single Agent Performance", fontweight="bold")
                plt.ylabel("Final Reward")

        # Hierarchical performance
        plt.subplot(2, 2, 2)
        if "hierarchical" in all_results:
            hierarchical_names = list(all_results["hierarchical"].keys())
            hierarchical_rewards = []
            for name in hierarchical_names:
                rewards = all_results["hierarchical"][name].get("rewards", [])
                if rewards:
                    final_reward = (
                        np.mean(rewards[-20:])
                        if len(rewards) >= 20
                        else np.mean(rewards)
                    )
                    hierarchical_rewards.append(final_reward)
            if hierarchical_rewards:
                plt.bar(hierarchical_names, hierarchical_rewards, alpha=0.7)
                plt.title("Hierarchical RL Performance", fontweight="bold")
                plt.ylabel("Final Reward")

        # Multi-agent team performance
        plt.subplot(2, 2, 3)
        if "multi_agent" in all_results:
            team_rewards = all_results["multi_agent"].get("team_rewards", [])
            if team_rewards:
                window = max(1, len(team_rewards) // 50)
                smoothed = (
                    pd.Series(team_rewards).rolling(window=window, min_periods=1).mean()
                )
                plt.plot(team_rewards, alpha=0.3, color="gray", label="Raw")
                plt.plot(smoothed, color="darkgreen", linewidth=2, label="Smoothed")
                plt.title("Multi-Agent Team Performance", fontweight="bold")
                plt.xlabel("Episode")
                plt.ylabel("Team Reward")
                plt.legend()

        # World model training progress
        plt.subplot(2, 2, 4)
        if "world_model" in all_results:
            losses_history = all_results["world_model"].get("losses_history", {})
            if "total" in losses_history:
                plt.plot(losses_history["total"], linewidth=2, label="Total Loss")
                if "vae" in losses_history:
                    plt.plot(losses_history["vae"], alpha=0.7, label="VAE Loss")
                if "dynamics" in losses_history:
                    plt.plot(
                        losses_history["dynamics"], alpha=0.7, label="Dynamics Loss"
                    )
                plt.title("World Model Training", fontweight="bold")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.yscale("log")

        plt.tight_layout()
        plt.savefig(
            os.path.join(viz_folder, "final_summary.png"), dpi=300, bbox_inches="tight"
        )
        plt.show()

    except Exception as e:
        print(f"❌ خطا در اجرای آزمایشات: {str(e)}")
        import traceback

        traceback.print_exc()

    return all_results


if __name__ == "__main__":
    results = main()
