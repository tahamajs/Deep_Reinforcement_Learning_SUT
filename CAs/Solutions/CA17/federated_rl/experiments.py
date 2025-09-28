"""
Experiments Module

This module contains comprehensive evaluation suites and experiments
for next-generation deep reinforcement learning paradigms.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import json
import os
from datetime import datetime
import pandas as pd
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from ..world_models import WorldModel, ImaginationAugmentedAgent
from ..multi_agent_rl import MADDPGAgent, PredatorPreyEnvironment
from ..causal_rl import CausalRLAgent, CausalBanditEnvironment
from ..quantum_rl import QuantumRLAgent, QuantumControlEnvironment
from ..federated_rl import FederatedRLServer
from ..advanced_safety import SafetyMonitor, ConstrainedPolicyOptimization
from ..utils import Config, Timer, compute_metrics, plot_learning_curve, plot_multiple_curves
from ..environments import ContinuousMountainCar, FederatedLearningEnvironment

class ExperimentRunner:
    """Base class for running RL experiments"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        self.config = config
        self.save_dir = save_dir
        self.results = defaultdict(list)
        self.timer = Timer()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def run_experiment(self) -> Dict[str, Any]:
        """Run the experiment"""
        raise NotImplementedError

    def save_results(self, filename: Optional[str] = None):
        """Save experiment results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__}_{timestamp}.json"

        filepath = os.path.join(self.save_dir, filename)

        # Convert results to serializable format
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, (list, tuple)):
                serializable_results[key] = [self._make_serializable(v) for v in value]
            else:
                serializable_results[key] = self._make_serializable(value)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filepath}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable"""
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def plot_results(self):
        """Plot experiment results"""
        pass

class WorldModelExperiment(ExperimentRunner):
    """Experiment for world models and imagination-augmented agents"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup
        self.env = ContinuousMountainCar(goal_velocity=config.goal_velocity)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Agent setup
        self.agent = ImaginationAugmentedAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config.hidden_dim,
            imagination_horizon=config.imagination_horizon,
            learning_rate=config.learning_rate
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run world model experiment"""
        print("Running World Model Experiment...")

        self.timer.start()
        episode_rewards = []
        imagination_errors = []
        prediction_errors = []

        for episode in range(self.config.n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # Store experience
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train agent
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    metrics = self.agent.train_step()
                    if metrics:
                        imagination_errors.append(metrics.get('imagination_error', 0))
                        prediction_errors.append(metrics.get('prediction_error', 0))

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.n_episodes}, "
                      f"Reward: {episode_reward:.2f}")

        self.timer.stop()

        # Store results
        self.results['episode_rewards'] = episode_rewards
        self.results['imagination_errors'] = imagination_errors
        self.results['prediction_errors'] = prediction_errors
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot world model experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Learning curve
        axes[0, 0].plot(self.results['episode_rewards'])
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Imagination errors
        if self.results['imagination_errors']:
            axes[0, 1].plot(self.results['imagination_errors'])
            axes[0, 1].set_title('Imagination Error')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Error')
            axes[0, 1].grid(True, alpha=0.3)

        # Prediction errors
        if self.results['prediction_errors']:
            axes[1, 0].plot(self.results['prediction_errors'])
            axes[1, 0].set_title('Prediction Error')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True, alpha=0.3)

        # Rolling average reward
        if len(self.results['episode_rewards']) > 10:
            rolling_avg = pd.Series(self.results['episode_rewards']).rolling(10).mean()
            axes[1, 1].plot(rolling_avg)
            axes[1, 1].set_title('Rolling Average Reward (window=10)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'world_model_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class MultiAgentExperiment(ExperimentRunner):
    """Experiment for multi-agent reinforcement learning"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup
        self.env = PredatorPreyEnvironment(
            n_predators=config.n_predators,
            n_prey=config.n_prey,
            grid_size=config.grid_size,
            max_steps=config.max_steps
        )

        # Agent setup
        obs_dim = self.env.observation_space.shape[0]
        self.agent = MADDPGAgent(
            n_predators=config.n_predators,
            n_prey=config.n_prey,
            obs_dim=obs_dim,
            action_dim=5,  # 5 actions per agent
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run multi-agent experiment"""
        print("Running Multi-Agent Experiment...")

        self.timer.start()
        predator_rewards_history = []
        prey_rewards_history = []
        capture_rates = []

        for episode in range(self.config.n_episodes):
            obs, _ = self.env.reset()
            episode_predator_rewards = []
            episode_prey_rewards = []
            captures = 0

            for step in range(self.config.max_steps):
                # Get actions for all agents
                actions = self.agent.select_actions(obs)

                # Execute actions
                next_obs, rewards, done, _, _ = self.env.step(actions)

                # Store transitions
                self.agent.store_transition(obs, actions, rewards, next_obs, done)

                # Train agents
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    self.agent.train_step()

                obs = next_obs
                episode_predator_rewards.append(np.mean(rewards['predators']))
                episode_prey_rewards.append(np.mean(rewards['prey']))

                if done:
                    break

            # Calculate capture rate
            initial_prey = self.config.n_prey
            final_prey = len(self.env.prey_positions)
            capture_rate = (initial_prey - final_prey) / initial_prey
            capture_rates.append(capture_rate)

            predator_rewards_history.append(np.mean(episode_predator_rewards))
            prey_rewards_history.append(np.mean(episode_prey_rewards))

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.n_episodes}, "
                      f"Predator Reward: {predator_rewards_history[-1]:.2f}, "
                      f"Prey Reward: {prey_rewards_history[-1]:.2f}, "
                      f"Capture Rate: {capture_rate:.2f}")

        self.timer.stop()

        # Store results
        self.results['predator_rewards'] = predator_rewards_history
        self.results['prey_rewards'] = prey_rewards_history
        self.results['capture_rates'] = capture_rates
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot multi-agent experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Predator rewards
        axes[0, 0].plot(self.results['predator_rewards'], label='Predators', color='red')
        axes[0, 0].set_title('Predator Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Prey rewards
        axes[0, 1].plot(self.results['prey_rewards'], label='Prey', color='blue')
        axes[0, 1].set_title('Prey Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Capture rates
        axes[1, 0].plot(self.results['capture_rates'], color='green')
        axes[1, 0].set_title('Capture Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Capture Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # Combined rewards
        axes[1, 1].plot(self.results['predator_rewards'], label='Predators', color='red', alpha=0.7)
        axes[1, 1].plot(self.results['prey_rewards'], label='Prey', color='blue', alpha=0.7)
        axes[1, 1].set_title('Combined Rewards')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'multi_agent_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class CausalRLExperiment(ExperimentRunner):
    """Experiment for causal reinforcement learning"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup
        self.env = CausalBanditEnvironment(
            n_arms=config.n_arms,
            n_contexts=config.n_contexts
        )

        # Agent setup
        self.agent = CausalRLAgent(
            n_arms=config.n_arms,
            n_contexts=config.n_contexts,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run causal RL experiment"""
        print("Running Causal RL Experiment...")

        self.timer.start()
        episode_rewards = []
        causal_discoveries = []
        counterfactual_regrets = []

        for episode in range(self.config.n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < self.config.max_steps:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _, info = self.env.step(action)

                # Store experience
                self.agent.store_transition(obs, action, reward, next_obs, done)

                # Train agent
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    metrics = self.agent.train_step()
                    if metrics:
                        causal_discoveries.append(metrics.get('causal_strength', 0))
                        counterfactual_regrets.append(metrics.get('counterfactual_regret', 0))

                obs = next_obs
                episode_reward += reward
                step += 1

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.n_episodes}, "
                      f"Reward: {episode_reward:.2f}")

        self.timer.stop()

        # Store results
        self.results['episode_rewards'] = episode_rewards
        self.results['causal_discoveries'] = causal_discoveries
        self.results['counterfactual_regrets'] = counterfactual_regrets
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot causal RL experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Learning curve
        axes[0, 0].plot(self.results['episode_rewards'])
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Causal discoveries
        if self.results['causal_discoveries']:
            axes[0, 1].plot(self.results['causal_discoveries'])
            axes[0, 1].set_title('Causal Discovery Strength')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Causal Strength')
            axes[0, 1].grid(True, alpha=0.3)

        # Counterfactual regrets
        if self.results['counterfactual_regrets']:
            axes[1, 0].plot(self.results['counterfactual_regrets'])
            axes[1, 0].set_title('Counterfactual Regret')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Regret')
            axes[1, 0].grid(True, alpha=0.3)

        # Rolling average reward
        if len(self.results['episode_rewards']) > 10:
            rolling_avg = pd.Series(self.results['episode_rewards']).rolling(10).mean()
            axes[1, 1].plot(rolling_avg)
            axes[1, 1].set_title('Rolling Average Reward (window=10)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'causal_rl_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class QuantumRLExperiment(ExperimentRunner):
    """Experiment for quantum-enhanced reinforcement learning"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup
        self.env = QuantumControlEnvironment(
            n_qubits=config.n_qubits,
            max_steps=config.max_steps
        )

        # Agent setup
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.agent = QuantumRLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run quantum RL experiment"""
        print("Running Quantum RL Experiment...")

        self.timer.start()
        episode_rewards = []
        fidelities = []
        quantum_entropies = []

        for episode in range(self.config.n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < self.config.max_steps:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _, info = self.env.step(action)

                # Store experience
                self.agent.store_transition(obs, action, reward, next_obs, done)

                # Train agent
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    metrics = self.agent.train_step()
                    if metrics:
                        quantum_entropies.append(metrics.get('quantum_entropy', 0))

                obs = next_obs
                episode_reward += reward
                fidelities.append(info['fidelity'])
                step += 1

            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.n_episodes}, "
                      f"Reward: {episode_reward:.4f}, "
                      f"Final Fidelity: {fidelities[-1]:.4f}")

        self.timer.stop()

        # Store results
        self.results['episode_rewards'] = episode_rewards
        self.results['fidelities'] = fidelities
        self.results['quantum_entropies'] = quantum_entropies
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot quantum RL experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Learning curve
        axes[0, 0].plot(self.results['episode_rewards'])
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Final fidelities per episode
        final_fidelities = [self.results['fidelities'][i] for i in range(len(self.results['episode_rewards']))]
        axes[0, 1].plot(final_fidelities)
        axes[0, 1].set_title('Final Fidelity per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Fidelity')
        axes[0, 1].grid(True, alpha=0.3)

        # Quantum entropy
        if self.results['quantum_entropies']:
            axes[1, 0].plot(self.results['quantum_entropies'])
            axes[1, 0].set_title('Quantum Entropy')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)

        # Reward vs Fidelity correlation
        axes[1, 1].scatter(final_fidelities, self.results['episode_rewards'], alpha=0.6)
        axes[1, 1].set_xlabel('Final Fidelity')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].set_title('Reward vs Fidelity Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'quantum_rl_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class FederatedRLExperiment(ExperimentRunner):
    """Experiment for federated reinforcement learning"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup
        self.env = FederatedLearningEnvironment(
            n_clients=config.n_clients,
            data_size=config.data_size,
            heterogeneity=config.heterogeneity
        )

        # Server setup
        self.server = FederatedRLServer(
            n_clients=config.n_clients,
            model_dim=1,  # Simple linear model
            learning_rate=config.learning_rate
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run federated RL experiment"""
        print("Running Federated RL Experiment...")

        self.timer.start()
        global_losses = []
        participation_rates = []
        communication_costs = []

        for round_num in range(self.config.n_rounds):
            obs, _ = self.env.reset()

            # Select clients to participate (simple strategy: random selection)
            n_selected = max(1, int(self.config.participation_rate * self.config.n_clients))
            selected_clients = np.random.choice(self.config.n_clients, n_selected, replace=False)
            action = np.zeros(self.config.n_clients)
            action[selected_clients] = 1

            # Execute federated round
            next_obs, reward, done, _, info = self.env.step(action)

            # Store and aggregate updates
            self.server.aggregate_updates(selected_clients, reward)

            global_losses.append(info['global_loss'])
            participation_rates.append(info['participation_rate'])
            communication_costs.append(len(selected_clients))  # Simple cost metric

            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{self.config.n_rounds}, "
                      f"Global Loss: {global_losses[-1]:.4f}, "
                      f"Participation Rate: {participation_rates[-1]:.2f}")

        self.timer.stop()

        # Store results
        self.results['global_losses'] = global_losses
        self.results['participation_rates'] = participation_rates
        self.results['communication_costs'] = communication_costs
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot federated RL experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Global loss
        axes[0, 0].plot(self.results['global_losses'])
        axes[0, 0].set_title('Global Loss')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Participation rates
        axes[0, 1].plot(self.results['participation_rates'])
        axes[0, 1].set_title('Participation Rate')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Participation Rate')
        axes[0, 1].grid(True, alpha=0.3)

        # Communication costs
        axes[1, 0].plot(self.results['communication_costs'])
        axes[1, 0].set_title('Communication Cost')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].grid(True, alpha=0.3)

        # Loss vs Participation correlation
        axes[1, 1].scatter(self.results['participation_rates'], self.results['global_losses'], alpha=0.6)
        axes[1, 1].set_xlabel('Participation Rate')
        axes[1, 1].set_ylabel('Global Loss')
        axes[1, 1].set_title('Loss vs Participation Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'federated_rl_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class SafetyExperiment(ExperimentRunner):
    """Experiment for advanced safety and robustness techniques"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)

        # Environment setup (using a simple continuous control task)
        self.env = ContinuousMountainCar()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Agent setup with safety constraints
        self.agent = ConstrainedPolicyOptimization(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            cost_limit=config.cost_limit
        )

        # Safety monitor
        self.safety_monitor = SafetyMonitor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            safety_threshold=config.safety_threshold
        )

    def run_experiment(self) -> Dict[str, Any]:
        """Run safety experiment"""
        print("Running Safety Experiment...")

        self.timer.start()
        episode_rewards = []
        episode_costs = []
        safety_violations = []
        constraint_values = []

        for episode in range(self.config.n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_cost = 0
            done = False
            violations = 0

            while not done:
                action = self.agent.select_action(state)

                # Check safety constraints
                is_safe, cost = self.safety_monitor.check_safety(state, action)
                if not is_safe:
                    violations += 1
                    # Apply safety intervention
                    action = self.safety_monitor.intervene(state)

                next_state, reward, done, _, _ = self.env.step(action)

                # Store experience
                self.agent.store_transition(state, action, reward, next_state, done, cost)

                # Train agent
                if len(self.agent.replay_buffer) > self.config.batch_size:
                    metrics = self.agent.train_step()
                    if metrics:
                        constraint_values.append(metrics.get('constraint_value', 0))

                state = next_state
                episode_reward += reward
                episode_cost += cost

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            safety_violations.append(violations)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.n_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Cost: {episode_cost:.2f}, "
                      f"Violations: {violations}")

        self.timer.stop()

        # Store results
        self.results['episode_rewards'] = episode_rewards
        self.results['episode_costs'] = episode_costs
        self.results['safety_violations'] = safety_violations
        self.results['constraint_values'] = constraint_values
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot safety experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Learning curve
        axes[0, 0].plot(self.results['episode_rewards'])
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Safety costs
        axes[0, 1].plot(self.results['episode_costs'])
        axes[0, 1].set_title('Safety Costs')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].grid(True, alpha=0.3)

        # Safety violations
        axes[1, 0].plot(self.results['safety_violations'])
        axes[1, 0].set_title('Safety Violations')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Violations')
        axes[1, 0].grid(True, alpha=0.3)

        # Constraint values
        if self.results['constraint_values']:
            axes[1, 1].plot(self.results['constraint_values'])
            axes[1, 1].set_title('Constraint Values')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Constraint Value')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'safety_experiment.png'), dpi=300, bbox_inches='tight')
        plt.show()

class ComparativeExperiment(ExperimentRunner):
    """Comparative experiment across multiple RL paradigms"""

    def __init__(self, config: Config, save_dir: str = "experiments"):
        super().__init__(config, save_dir)
        self.experiments = {}

    def add_experiment(self, name: str, experiment: ExperimentRunner):
        """Add an experiment to the comparison"""
        self.experiments[name] = experiment

    def run_experiment(self) -> Dict[str, Any]:
        """Run comparative experiment"""
        print("Running Comparative Experiment...")

        self.timer.start()
        all_results = {}

        for name, experiment in self.experiments.items():
            print(f"\n--- Running {name} ---")
            results = experiment.run_experiment()
            all_results[name] = results

        self.timer.stop()

        # Store comparative results
        self.results['comparative_results'] = all_results
        self.results['total_time'] = self.timer.get_elapsed()
        self.results['config'] = self.config.to_dict()

        return dict(self.results)

    def plot_results(self):
        """Plot comparative results"""
        if 'comparative_results' not in self.results:
            return

        results = self.results['comparative_results']

        # Extract rewards for comparison
        reward_curves = {}
        for name, exp_results in results.items():
            if 'episode_rewards' in exp_results:
                reward_curves[name] = exp_results['episode_rewards']

        if reward_curves:
            plot_multiple_curves(reward_curves, title="Comparative Performance")
            plt.savefig(os.path.join(self.save_dir, 'comparative_experiment.png'), dpi=300, bbox_inches='tight')
            plt.show()

def create_default_configs() -> Dict[str, Config]:
    """Create default configurations for all experiments"""

    configs = {}

    # World Model Experiment
    configs['world_model'] = Config(
        n_episodes=100,
        batch_size=64,
        hidden_dim=128,
        imagination_horizon=5,
        learning_rate=1e-3,
        goal_velocity=0.0
    )

    # Multi-Agent Experiment
    configs['multi_agent'] = Config(
        n_episodes=100,
        batch_size=64,
        hidden_dim=128,
        learning_rate=1e-3,
        n_predators=2,
        n_prey=1,
        grid_size=10,
        max_steps=100
    )

    # Causal RL Experiment
    configs['causal_rl'] = Config(
        n_episodes=100,
        batch_size=64,
        hidden_dim=128,
        learning_rate=1e-3,
        n_arms=3,
        n_contexts=2,
        max_steps=50
    )

    # Quantum RL Experiment
    configs['quantum_rl'] = Config(
        n_episodes=50,
        batch_size=32,
        hidden_dim=64,
        learning_rate=1e-3,
        n_qubits=2,
        max_steps=20
    )

    # Federated RL Experiment
    configs['federated_rl'] = Config(
        n_rounds=100,
        n_clients=10,
        data_size=100,
        heterogeneity=0.5,
        learning_rate=1e-3,
        participation_rate=0.5
    )

    # Safety Experiment
    configs['safety'] = Config(
        n_episodes=100,
        batch_size=64,
        hidden_dim=128,
        learning_rate=1e-3,
        cost_limit=1.0,
        safety_threshold=0.8
    )

    return configs

print("✅ Experiments module complete!")
print("Components implemented:")
print("- ExperimentRunner: Base class for experiments")
print("- WorldModelExperiment: World model evaluation")
print("- MultiAgentExperiment: Multi-agent RL evaluation")
print("- CausalRLExperiment: Causal RL evaluation")
print("- QuantumRLExperiment: Quantum RL evaluation")
print("- FederatedRLExperiment: Federated RL evaluation")
print("- SafetyExperiment: Safety and robustness evaluation")
print("- ComparativeExperiment: Cross-paradigm comparison")
print("- create_default_configs: Default experiment configurations")