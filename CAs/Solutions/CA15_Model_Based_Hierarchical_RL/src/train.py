import torch
import numpy as np
import time
import os

from typing import Any, Dict, List, Tuple, Union
from src.config import Config
from src.utils import set_seed, to_tensor, EpisodeMetrics, env_reset, env_step
from src.agents import DQNAgent, DynaQAgent, HierarchicalActorCritic, GoalConditionedAgent, FeudalNetwork, ModelPredictiveController, MonteCarloTreeSearch, ModelBasedValueExpansion, LatentSpacePlanner, WorldModel
from src.data import SimpleGridWorld # Assuming custom environments are in src/data.py or similar


def train_agent(
    config: Config,
    agent: Any,
    env: Any,
    agent_type: str,
    experiment_name: str = "default_experiment",
    save_best_model: bool = True,
) -> Dict[str, List[float]]:
    """Generic training loop for various RL agents."""
    print(f"\n🚀 Starting training for {agent_type} - {experiment_name}")
    set_seed(config.general.seed)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    q_losses: List[float] = []
    model_losses: List[float] = []
    
    start_time = time.time()

    state = env_reset(env)

    for episode in range(config.general.total_env_steps // config.environment.max_episode_steps):
        current_episode_reward = 0.0
        current_episode_length = 0
        done = False
        episode_q_losses = []
        episode_model_losses = []

        while not done and current_episode_length < config.environment.max_episode_steps:
            # Agent selects action
            if agent_type == "DynaQAgent" or agent_type == "DQNAgent":
                action = agent.get_action(state, training=True)
            elif agent_type == "HierarchicalActorCritic":
                # Manager selects subgoal, worker acts
                subgoal = agent.select_action(state, level=0)
                action = agent.select_action(state, level=1, subgoal=subgoal)
            elif agent_type == "GoalConditionedAgent":
                goal = np.array(env.current_goal) if hasattr(env, 'current_goal') else np.zeros(config.manager.subgoal_dim)
                action = agent.action(state, goal, noise_scale=0.1)
            elif agent_type == "ModelPredictiveController":
                action = agent.plan(state)
            elif agent_type == "MonteCarloTreeSearch":
                action = agent.get_best_action(state)
            elif agent_type == "LatentSpacePlanner":
                action = agent.plan(state)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            next_state, reward, done, info = env_step(env, action)

            # Store experience
            if agent_type == "DynaQAgent" or agent_type == "DQNAgent":
                agent.store_experience(state, action, reward, next_state, done)
            elif agent_type == "HierarchicalActorCritic":
                agent.store_worker_experience(state, action, reward, next_state, done, subgoal)
                # Manager experience is stored at manager update frequency
            elif agent_type == "GoalConditionedAgent":
                goal = np.array(env.current_goal) if hasattr(env, 'current_goal') else np.zeros(config.manager.subgoal_dim)
                agent.store_experience(state, action, reward, next_state, done, goal)

            current_episode_reward += reward
            current_episode_length += 1
            state = next_state
            agent.training_steps += 1

            # Update agent
            if agent_type == "DynaQAgent":
                if len(agent.replay_buffer) > config.general.batch_size:
                    q_loss = agent.update_q_function()
                    model_loss = agent.update_model()
                    agent.planning_step() # Model-based planning
                    episode_q_losses.append(q_loss)
                    episode_model_losses.append(model_loss)
                if agent.training_steps % config.general.target_update_freq == 0: # Assuming a target update freq for DynaQ
                    agent.update_target_network()

            elif agent_type == "DQNAgent":
                if len(agent.replay_buffer) > config.general.batch_size:
                    q_loss = agent.update()
                    episode_q_losses.append(q_loss)
                if agent.training_steps % config.general.target_update_freq == 0: # Assuming a target update freq for DQN
                    agent.update_target_network()
            
            elif agent_type == "HierarchicalActorCritic":
                if len(agent.worker_buffer) > config.general.batch_size:
                    actor_loss, critic_loss = agent.update_worker()
                    episode_q_losses.append(critic_loss) # Using critic loss as Q-loss for plotting
                
                # Update manager periodically
                if agent.worker_steps_counter % agent.update_frequency_worker_steps == 0:
                    if len(agent.manager_buffer) > config.general.batch_size:
                        manager_actor_loss, manager_critic_loss = agent.update_manager()
                        # Store manager's reward for this subgoal and next state for manager update
                        # (This is more complex and depends on extrinsic reward collection over subgoal duration)

            elif agent_type == "GoalConditionedAgent":
                if len(agent.replay_buffer) > config.general.batch_size:
                    actor_loss, critic_loss = agent.update()
                    episode_q_losses.append(critic_loss)

            # World Model training (if applicable)
            if agent_type == "WorldModel": # WorldModel is a model, not an agent itself, will be trained separately.
                pass # Implement WorldModel training loop separately if needed.

        # End of episode
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)
        if episode_q_losses: q_losses.append(np.mean(episode_q_losses))
        if episode_model_losses: model_losses.append(np.mean(episode_model_losses))

        # Log progress
        if episode % config.general.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.general.log_interval:]) if len(episode_rewards) > config.general.log_interval else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-config.general.log_interval:]) if len(episode_lengths) > config.general.log_interval else np.mean(episode_lengths)
            print(f"Episode {episode}/{config.general.total_env_steps // config.environment.max_episode_steps} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Epsilon: {getattr(agent, 'epsilon', 'N/A'):.2f}")

        # Save best model (implement logic based on agent_type and performance metric)
        # if save_best_model and current_episode_reward > best_reward:
        #     best_reward = current_episode_reward
        #     torch.save(agent.state_dict(), os.path.join(config.general.results_dir, f"{experiment_name}_best_model.pth"))

    end_time = time.time()
    print(f"Training finished for {agent_type} - {experiment_name} in {end_time - start_time:.2f} seconds.")

    return {"episode_rewards": episode_rewards, "episode_lengths": episode_lengths, "q_losses": q_losses, "model_losses": model_losses}


def run_model_based_experiments(config: Config) -> Dict[str, Any]:
    """Run model-based RL experiments."""
    print("\n🔄 Running Model-Based RL Experiments")
    print("=" * 50)
    results = {}

    # Environment setup (using SimpleGridWorld for now, will generalize later)
    env = SimpleGridWorld(grid_size=config.environment.grid_size, goal_pos=(config.environment.grid_size - 1, config.environment.grid_size - 1))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    config.dynamics_model.state_dim = state_dim
    config.dynamics_model.action_dim = action_dim
    config.worker.state_dim = state_dim
    config.worker.action_dim = action_dim

    # 1. Dyna-Q Agent
    print("1. Training Dyna-Q Agent...")
    dyna_agent = DynaQAgent(config, state_dim, action_dim)
    dyna_q_results = train_agent(config, dyna_agent, env, "DynaQAgent", "dyna_q_experiment")
    results["dyna_q"] = dyna_q_results

    return results


def run_hierarchical_experiments(config: Config) -> Dict[str, Any]:
    """Run hierarchical RL experiments."""
    print("\n🔄 Running Hierarchical RL Experiments")
    print("=" * 50)
    results = {}

    # Environment setup
    env = SimpleGridWorld(grid_size=config.environment.grid_size, goal_pos=(config.environment.grid_size - 1, config.environment.grid_size - 1))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    config.manager.state_dim = state_dim
    config.manager.subgoal_dim = state_dim # Subgoals are states
    config.worker.state_dim = state_dim
    config.worker.action_dim = action_dim
    config.worker.goal_dim = state_dim

    # 1. Goal-Conditioned Agent (with HER concept)
    print("1. Training Goal-Conditioned Agent...")
    gc_agent = GoalConditionedAgent(config, state_dim, action_dim, state_dim) # goal_dim = state_dim
    gc_results = train_agent(config, gc_agent, env, "GoalConditionedAgent", "goal_conditioned_experiment")
    results["goal_conditioned"] = gc_results

    # 2. Hierarchical Actor-Critic (simplified Feudal-like for now)
    print("2. Training Hierarchical Actor-Critic (Feudal-like)...")
    hac_agent = HierarchicalActorCritic(config, state_dim, action_dim, state_dim) # subgoal_dim = state_dim
    hac_results = train_agent(config, hac_agent, env, "HierarchicalActorCritic", "hierarchical_ac_experiment")
    results["hierarchical_ac"] = hac_results

    return results


def run_planning_experiments(config: Config) -> Dict[str, Any]:
    """Run planning algorithm experiments."""
    print("\n🔄 Running Planning Algorithm Experiments")
    print("=" * 50)
    results = {}

    # Environment setup
    env = SimpleGridWorld(grid_size=config.environment.grid_size, goal_pos=(config.environment.grid_size - 1, config.environment.grid_size - 1))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    config.dynamics_model.state_dim = state_dim
    config.dynamics_model.action_dim = action_dim

    # Train a simple dynamics model for planning algorithms
    print("Training a Dynamics Model for Planning Algorithms...")
    dynamics_model = DynamicsModel(state_dim, action_dim, config.dynamics_model.hidden_dim)
    model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=config.dynamics_model.learning_rate)
    model_training_data = []
    for _ in range(100): # Collect some initial data
        state = env_reset(env)
        done = False
        while not done:
            action = random.randrange(action_dim)
            next_state, reward, done, _ = env_step(env, action)
            model_training_data.append((state, action, reward, next_state))
            state = next_state
    
    for epoch in range(50): # Train model for a few epochs
        batch = random.sample(model_training_data, min(len(model_training_data), config.general.batch_size))
        states, actions, rewards, next_states = zip(*batch)
        states = to_tensor(np.array(states))
        actions = to_tensor(np.array(actions), dtype=torch.long)
        rewards = to_tensor(np.array(rewards).reshape(-1, 1))
        next_states = to_tensor(np.array(next_states))

        actions_one_hot = F.one_hot(actions, num_classes=action_dim).float() if action_dim > 1 else actions.float().unsqueeze(-1)
        predicted_next_states, predicted_rewards = dynamics_model(states, actions_one_hot)
        loss = dynamics_model_loss(predicted_next_states, next_states, predicted_rewards, rewards)

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
    print("Dynamics Model Trained.")

    # 1. Model Predictive Controller (using the trained dynamics model)
    print("1. Training Model Predictive Controller...")
    mpc_agent = ModelPredictiveController(config, dynamics_model, action_dim)
    mpc_results = train_agent(config, mpc_agent, env, "ModelPredictiveController", "mpc_experiment")
    results["mpc"] = mpc_results

    # 2. Monte Carlo Tree Search (using the trained dynamics model)
    print("2. Training Monte Carlo Tree Search...")
    mcts_agent = MonteCarloTreeSearch(config, dynamics_model, SimpleGridWorld, action_dim)
    mcts_results = train_agent(config, mcts_agent, env, "MonteCarloTreeSearch", "mcts_experiment")
    results["mcts"] = mcts_results

    return results

