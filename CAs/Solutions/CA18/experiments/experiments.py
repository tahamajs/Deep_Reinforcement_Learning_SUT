import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
import copy
import random
import time
import json
import os
from pathlib import Path
import pandas as pd
import seaborn as sns

# Base Experiment Class
class BaseExperiment:
    """Base class for reinforcement learning experiments"""

    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "experiments",
        random_seed: Optional[int] = None,
    ):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if random_seed is not None:
            self.set_random_seed(random_seed)

        self.results = defaultdict(list)
        self.start_time = time.time()

    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if step is None:
            step = len(self.results[metric_name])

        self.results[metric_name].append({
            'step': step,
            'value': value,
            'timestamp': time.time()
        })

    def save_results(self):
        """Save experiment results"""
        results_file = self.save_dir / "results.json"

        # Convert to serializable format
        serializable_results = {}
        for metric, values in self.results.items():
            serializable_results[metric] = values

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def load_results(self) -> Dict:
        """Load experiment results"""
        results_file = self.save_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}

    def plot_results(self, metrics: Optional[List[str]] = None, save_plot: bool = True):
        """Plot experiment results"""

        if metrics is None:
            metrics = list(self.results.keys())

        n_metrics = len(metrics)
        if n_metrics == 0:
            return

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric not in self.results:
                continue

            ax = axes[i]
            values = self.results[metric]

            if values:
                steps = [v['step'] for v in values]
                vals = [v['value'] for v in values]

                ax.plot(steps, vals, label=metric, alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()

        if save_plot:
            plt.savefig(self.save_dir / "results_plot.png", dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

# Quantum RL Experiment
class QuantumRLExperiment(BaseExperiment):
    """Experiment framework for quantum reinforcement learning"""

    def __init__(
        self,
        agent_class,
        environment_class,
        experiment_name: str = "quantum_rl_experiment",
        **experiment_kwargs
    ):
        super().__init__(experiment_name, **experiment_kwargs)

        self.agent_class = agent_class
        self.environment_class = environment_class

        # Experiment parameters
        self.n_episodes = 100
        self.max_steps_per_episode = 100
        self.eval_frequency = 10

        # Results tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.fidelities = []
        self.purities = []
        self.entanglements = []

    def run_experiment(self, agent_kwargs: Dict = {}, env_kwargs: Dict = {}):
        """Run the quantum RL experiment"""

        print(f"🚀 Starting Quantum RL Experiment: {self.experiment_name}")

        # Initialize agent and environment
        env = self.environment_class(**env_kwargs)
        agent = self.agent_class(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            **agent_kwargs
        )

        best_reward = -float('inf')

        for episode in range(self.n_episodes):
            # Training episode
            episode_reward, episode_length, episode_info = self._run_episode(
                agent, env, train=True
            )

            # Log metrics
            self.log_metric('episode_reward', episode_reward, episode)
            self.log_metric('episode_length', episode_length, episode)
            self.log_metric('fidelity', episode_info.get('fidelity', 0), episode)
            self.log_metric('purity', episode_info.get('purity', 0), episode)
            self.log_metric('entanglement', episode_info.get('entanglement', 0), episode)

            # Store for analysis
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.fidelities.append(episode_info.get('fidelity', 0))
            self.purities.append(episode_info.get('purity', 0))
            self.entanglements.append(episode_info.get('entanglement', 0))

            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_reward, eval_info = self._evaluate_agent(agent, env)
                self.log_metric('eval_reward', eval_reward, episode)
                print(f"Episode {episode}: Train Reward = {episode_reward:.2f}, "
                      f"Eval Reward = {eval_reward:.2f}, Fidelity = {episode_info.get('fidelity', 0):.4f}")

                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_best_model(agent, episode)

        # Save final results
        self.save_results()
        self._generate_experiment_report()

        print(f"✅ Experiment completed. Best evaluation reward: {best_reward:.2f}")

        return self.results

    def _run_episode(self, agent, env, train: bool = True) -> Tuple[float, int, Dict]:
        """Run a single episode"""

        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {}

        for step in range(self.max_steps_per_episode):
            # Get action
            action = agent.get_action(torch.FloatTensor(state).unsqueeze(0))
            if isinstance(action, torch.Tensor):
                action = action.squeeze(0).detach().numpy()

            # Take step
            next_state, reward, done, info = env.step(action)

            if train:
                # Store transition for training
                agent.store_transition(state, action, reward, next_state, done)

                # Update agent
                if hasattr(agent, 'update'):
                    agent.update()

            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_info = info

            if done:
                break

        return episode_reward, episode_length, episode_info

    def _evaluate_agent(self, agent, env, n_eval_episodes: int = 5) -> Tuple[float, Dict]:
        """Evaluate agent performance"""

        eval_rewards = []
        eval_infos = []

        for _ in range(n_eval_episodes):
            reward, _, info = self._run_episode(agent, env, train=False)
            eval_rewards.append(reward)
            eval_infos.append(info)

        avg_reward = np.mean(eval_rewards)
        avg_info = {
            key: np.mean([info.get(key, 0) for info in eval_infos])
            for key in eval_infos[0].keys()
        }

        return avg_reward, avg_info

    def _save_best_model(self, agent, episode: int):
        """Save best performing model"""
        model_path = self.save_dir / f"best_model_ep{episode}.pt"
        if hasattr(agent, 'save'):
            agent.save(model_path)
        elif hasattr(agent, 'state_dict'):
            torch.save(agent.state_dict(), model_path)

    def _generate_experiment_report(self):
        """Generate comprehensive experiment report"""

        report = {
            'experiment_name': self.experiment_name,
            'total_episodes': self.n_episodes,
            'duration': time.time() - self.start_time,
            'final_metrics': {
                'mean_reward': np.mean(self.episode_rewards[-10:]),
                'std_reward': np.std(self.episode_rewards[-10:]),
                'mean_fidelity': np.mean(self.fidelities[-10:]),
                'mean_purity': np.mean(self.purities[-10:]),
                'mean_entanglement': np.mean(self.entanglements[-10:]),
            },
            'best_performance': {
                'max_reward': np.max(self.episode_rewards),
                'max_fidelity': np.max(self.fidelities),
                'max_purity': np.max(self.purities),
            }
        }

        report_path = self.save_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

# Causal RL Experiment
class CausalRLExperiment(BaseExperiment):
    """Experiment framework for causal reinforcement learning"""

    def __init__(
        self,
        agent_class,
        environment_class,
        experiment_name: str = "causal_rl_experiment",
        **experiment_kwargs
    ):
        super().__init__(experiment_name, **experiment_kwargs)

        self.agent_class = agent_class
        self.environment_class = environment_class

        # Experiment parameters
        self.n_episodes = 200
        self.max_steps_per_episode = 50
        self.causal_discovery_frequency = 20

    def run_experiment(self, agent_kwargs: Dict = {}, env_kwargs: Dict = {}):
        """Run the causal RL experiment"""

        print(f"🔍 Starting Causal RL Experiment: {self.experiment_name}")

        env = self.environment_class(**env_kwargs)
        agent = self.agent_class(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **agent_kwargs
        )

        causal_graph_discovered = False

        for episode in range(self.n_episodes):
            # Run episode
            episode_reward, episode_info = self._run_causal_episode(agent, env)

            # Log metrics
            self.log_metric('episode_reward', episode_reward, episode)
            self.log_metric('causal_regret', episode_info.get('causal_regret', 0), episode)
            self.log_metric('intervention_accuracy', episode_info.get('intervention_accuracy', 0), episode)

            # Periodic causal discovery
            if episode % self.causal_discovery_frequency == 0:
                discovered_graph = agent.discover_causal_structure(env)
                causal_accuracy = self._evaluate_causal_discovery(discovered_graph, env)
                self.log_metric('causal_discovery_accuracy', causal_accuracy, episode)

                if not causal_graph_discovered and causal_accuracy > 0.8:
                    causal_graph_discovered = True
                    print(f"🎯 Causal structure discovered at episode {episode}!")

            if episode % 50 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                      f"Causal Regret = {episode_info.get('causal_regret', 0):.4f}")

        self.save_results()
        self._generate_causal_report()

        print("✅ Causal RL experiment completed.")

        return self.results

    def _run_causal_episode(self, agent, env) -> Tuple[float, Dict]:
        """Run episode with causal reasoning"""

        state = env.reset()
        episode_reward = 0
        causal_regret = 0
        intervention_accuracy = 0

        for step in range(self.max_steps_per_episode):
            # Causal intervention decision
            intervention = agent.decide_intervention(state)

            # Get action with causal awareness
            action = agent.get_action(torch.FloatTensor(state).unsqueeze(0), intervention)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Update causal model
            agent.update_causal_model(state, action, reward, next_state, intervention)

            # Compute causal regret
            optimal_action = agent.get_optimal_causal_action(state)
            causal_regret += abs(action - optimal_action)

            state = next_state
            episode_reward += reward

            if done:
                break

        intervention_accuracy = 1.0 - (causal_regret / (step + 1))

        return episode_reward, {
            'causal_regret': causal_regret,
            'intervention_accuracy': intervention_accuracy,
        }

    def _evaluate_causal_discovery(self, discovered_graph, env) -> float:
        """Evaluate accuracy of causal discovery"""
        # Simplified evaluation - compare discovered edges with true causal graph
        if hasattr(env, 'causal_graph'):
            true_edges = set(env.causal_graph.edges())
            discovered_edges = set(discovered_graph.edges())

            precision = len(true_edges & discovered_edges) / len(discovered_edges) if discovered_edges else 0
            recall = len(true_edges & discovered_edges) / len(true_edges) if true_edges else 0

            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return 0.0

    def _generate_causal_report(self):
        """Generate causal experiment report"""
        # Implementation similar to quantum report but focused on causal metrics
        pass

# Multi-Agent RL Experiment
class MultiAgentRLExperiment(BaseExperiment):
    """Experiment framework for multi-agent reinforcement learning"""

    def __init__(
        self,
        agent_class,
        environment_class,
        n_agents: int = 2,
        experiment_name: str = "multi_agent_rl_experiment",
        **experiment_kwargs
    ):
        super().__init__(experiment_name, **experiment_kwargs)

        self.agent_class = agent_class
        self.environment_class = environment_class
        self.n_agents = n_agents

    def run_experiment(self, agent_kwargs: Dict = {}, env_kwargs: Dict = {}):
        """Run multi-agent RL experiment"""

        print(f"👥 Starting Multi-Agent RL Experiment: {self.experiment_name}")

        env = self.environment_class(n_agents=self.n_agents, **env_kwargs)

        # Initialize agents
        agents = []
        for i in range(self.n_agents):
            agent = self.agent_class(
                state_dim=env.observation_space[i].shape[0],
                action_dim=env.action_space[i].shape[0],
                agent_id=i,
                **agent_kwargs
            )
            agents.append(agent)

        for episode in range(100):
            # Run multi-agent episode
            episode_rewards, episode_info = self._run_multi_agent_episode(agents, env)

            # Log metrics
            for i, reward in enumerate(episode_rewards):
                self.log_metric(f'agent_{i}_reward', reward, episode)

            self.log_metric('team_reward', np.mean(episode_rewards), episode)
            self.log_metric('communication_overhead', episode_info.get('communication_overhead', 0), episode)

            if episode % 20 == 0:
                print(f"Episode {episode}: Team Reward = {np.mean(episode_rewards):.2f}")

        self.save_results()
        print("✅ Multi-agent RL experiment completed.")

        return self.results

    def _run_multi_agent_episode(self, agents, env) -> Tuple[List[float], Dict]:
        """Run episode with multiple agents"""
        observations = env.reset()
        episode_rewards = [0] * self.n_agents
        communication_overhead = 0

        for step in range(50):
            actions = []

            # Get actions from all agents
            for i, agent in enumerate(agents):
                action = agent.get_action(torch.FloatTensor(observations[i]).unsqueeze(0))
                if isinstance(action, torch.Tensor):
                    action = action.squeeze(0).detach().numpy()
                actions.append(action)

                # Simulate communication
                if hasattr(agent, 'communicate'):
                    comm_data = agent.communicate(observations[i])
                    communication_overhead += len(comm_data) if comm_data else 0

            # Environment step
            next_observations, rewards, done, info = env.step(actions)

            # Update agents
            for i, agent in enumerate(agents):
                agent.store_transition(
                    observations[i], actions[i], rewards[i],
                    next_observations[i], done
                )
                if hasattr(agent, 'update'):
                    agent.update()

            # Accumulate rewards
            for i in range(self.n_agents):
                episode_rewards[i] += rewards[i]

            observations = next_observations

            if done:
                break

        return episode_rewards, {'communication_overhead': communication_overhead}

# Federated RL Experiment
class FederatedRLExperiment(BaseExperiment):
    """Experiment framework for federated reinforcement learning"""

    def __init__(
        self,
        client_class,
        server_class,
        environment_class,
        n_clients: int = 5,
        experiment_name: str = "federated_rl_experiment",
        **experiment_kwargs
    ):
        super().__init__(experiment_name, **experiment_kwargs)

        self.client_class = client_class
        self.server_class = server_class
        self.environment_class = environment_class
        self.n_clients = n_clients

        # Experiment parameters
        self.n_rounds = 50
        self.local_steps_per_round = 10

    def run_experiment(self, client_kwargs: Dict = {}, server_kwargs: Dict = {},
                      env_kwargs: Dict = {}):
        """Run federated RL experiment"""

        print(f"🔗 Starting Federated RL Experiment: {self.experiment_name}")

        # Initialize clients and server
        clients = []
        for i in range(self.n_clients):
            client_env = self.environment_class(**env_kwargs)
            client = self.client_class(
                client_id=i,
                environment=client_env,
                **client_kwargs
            )
            clients.append(client)

        server = self.server_class(n_clients=self.n_clients, **server_kwargs)

        for round_num in range(self.n_rounds):
            # Server sends global model to clients
            global_model = server.get_global_model()

            # Clients perform local training
            local_updates = []
            client_metrics = []

            for client in clients:
                client.set_global_model(global_model)
                update, metrics = client.local_train(self.local_steps_per_round)
                local_updates.append(update)
                client_metrics.append(metrics)

            # Server aggregates updates
            server.aggregate_updates(local_updates)

            # Log metrics
            avg_client_reward = np.mean([m.get('avg_reward', 0) for m in client_metrics])
            communication_cost = server.get_communication_cost()

            self.log_metric('round_avg_reward', avg_client_reward, round_num)
            self.log_metric('communication_cost', communication_cost, round_num)
            self.log_metric('global_model_accuracy', server.get_global_accuracy(), round_num)

            if round_num % 10 == 0:
                print(f"Round {round_num}: Avg Client Reward = {avg_client_reward:.2f}, "
                      f"Comm Cost = {communication_cost:.2f}")

        self.save_results()
        print("✅ Federated RL experiment completed.")

        return self.results

# Comparative Experiment Runner
class ComparativeExperimentRunner:
    """Run comparative experiments across different algorithms"""

    def __init__(self, save_dir: str = "comparative_experiments"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.experiment_results = {}

    def run_comparison(
        self,
        algorithms: Dict[str, Dict],
        environment_class,
        n_runs: int = 3,
        n_episodes: int = 100,
    ):
        """Run comparative experiments"""

        print(f"📊 Starting Comparative Experiment with {len(algorithms)} algorithms")

        for alg_name, alg_config in algorithms.items():
            print(f"\n🔬 Running {alg_name}...")

            alg_results = []

            for run in range(n_runs):
                print(f"  Run {run + 1}/{n_runs}")

                # Create experiment
                experiment_class = alg_config['experiment_class']
                experiment = experiment_class(
                    alg_config['agent_class'],
                    environment_class,
                    experiment_name=f"{alg_name}_run_{run}",
                    save_dir=str(self.save_dir / alg_name)
                )

                # Run experiment
                results = experiment.run_experiment(
                    agent_kwargs=alg_config.get('agent_kwargs', {}),
                    env_kwargs=alg_config.get('env_kwargs', {})
                )

                alg_results.append(results)

            self.experiment_results[alg_name] = alg_results

        # Generate comparison report
        self._generate_comparison_report()
        self._plot_comparison()

        print("✅ Comparative experiment completed.")

    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""

        report = {
            'timestamp': time.time(),
            'algorithms': {},
            'comparison_metrics': {}
        }

        for alg_name, runs in self.experiment_results.items():
            alg_metrics = []

            for run in runs:
                if 'episode_reward' in run:
                    rewards = [entry['value'] for entry in run['episode_reward']]
                    alg_metrics.append({
                        'final_reward': np.mean(rewards[-10:]),
                        'max_reward': np.max(rewards),
                        'convergence_speed': self._compute_convergence_speed(rewards),
                    })

            if alg_metrics:
                report['algorithms'][alg_name] = {
                    'mean_final_reward': np.mean([m['final_reward'] for m in alg_metrics]),
                    'std_final_reward': np.std([m['final_reward'] for m in alg_metrics]),
                    'mean_max_reward': np.mean([m['max_reward'] for m in alg_metrics]),
                    'mean_convergence_speed': np.mean([m['convergence_speed'] for m in alg_metrics]),
                }

        # Save report
        report_path = self.save_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def _compute_convergence_speed(self, rewards: List[float], threshold: float = 0.9) -> int:
        """Compute convergence speed (episodes to reach threshold of max reward)"""
        if not rewards:
            return len(rewards)

        max_reward = np.max(rewards)
        threshold_value = threshold * max_reward

        for i, reward in enumerate(rewards):
            if reward >= threshold_value:
                return i

        return len(rewards)

    def _plot_comparison(self):
        """Plot comparison results"""

        if not self.experiment_results:
            return

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        algorithms = list(self.experiment_results.keys())

        # Final reward comparison
        final_rewards = []
        final_reward_stds = []

        for alg in algorithms:
            runs = self.experiment_results[alg]
            finals = []

            for run in runs:
                if 'episode_reward' in run:
                    rewards = [entry['value'] for entry in run['episode_reward']]
                    finals.append(np.mean(rewards[-10:]))

            final_rewards.append(np.mean(finals))
            final_reward_stds.append(np.std(finals))

        axes[0, 0].bar(algorithms, final_rewards, yerr=final_reward_stds)
        axes[0, 0].set_title('Final Average Reward (±std)')
        axes[0, 0].set_ylabel('Reward')
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

        # Learning curves
        for alg in algorithms:
            runs = self.experiment_results[alg]
            all_rewards = []

            for run in runs:
                if 'episode_reward' in run:
                    rewards = [entry['value'] for entry in run['episode_reward']]
                    all_rewards.append(rewards)

            if all_rewards:
                mean_rewards = np.mean(all_rewards, axis=0)
                std_rewards = np.std(all_rewards, axis=0)

                episodes = range(len(mean_rewards))
                axes[0, 1].plot(episodes, mean_rewards, label=alg)
                axes[0, 1].fill_between(episodes,
                                       mean_rewards - std_rewards,
                                       mean_rewards + std_rewards,
                                       alpha=0.2)

        axes[0, 1].set_title('Learning Curves')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()

        # Convergence speed
        conv_speeds = []
        for alg in algorithms:
            runs = self.experiment_results[alg]
            speeds = []

            for run in runs:
                if 'episode_reward' in run:
                    rewards = [entry['value'] for entry in run['episode_reward']]
                    speed = self._compute_convergence_speed(rewards)
                    speeds.append(speed)

            conv_speeds.append(np.mean(speeds))

        axes[1, 0].bar(algorithms, conv_speeds)
        axes[1, 0].set_title('Convergence Speed (episodes)')
        axes[1, 0].set_ylabel('Episodes to 90% max reward')
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45)

        # Max reward comparison
        max_rewards = []
        for alg in algorithms:
            runs = self.experiment_results[alg]
            maxs = []

            for run in runs:
                if 'episode_reward' in run:
                    rewards = [entry['value'] for entry in run['episode_reward']]
                    maxs.append(np.max(rewards))

            max_rewards.append(np.mean(maxs))

        axes[1, 1].bar(algorithms, max_rewards)
        axes[1, 1].set_title('Maximum Reward Achieved')
        axes[1, 1].set_ylabel('Max Reward')
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / "comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

print("✅ Advanced Experiments implementations complete!")
print("Components implemented:")
print("- BaseExperiment: Base class for RL experiments with logging and plotting")
print("- QuantumRLExperiment: Experiment framework for quantum RL algorithms")
print("- CausalRLExperiment: Experiment framework for causal RL algorithms")
print("- MultiAgentRLExperiment: Experiment framework for multi-agent RL")
print("- FederatedRLExperiment: Experiment framework for federated RL")
print("- ComparativeExperimentRunner: Framework for comparing multiple algorithms")