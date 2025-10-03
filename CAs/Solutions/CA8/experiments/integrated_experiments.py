"""
Integrated Causal Multi-Modal RL Experiments
Experimental framework for evaluating integrated systems
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Handle both relative and absolute imports
try:
    from ..agents.causal_discovery import CausalGraph, CausalDiscovery
    from ..agents.causal_rl_agent import CausalRLAgent, CounterfactualRLAgent
    from ..environments.multi_modal_env import MultiModalGridWorld, MultiModalWrapper
    from ..evaluation.metrics import IntegratedMetrics
except ImportError:
    from agents.causal_discovery import CausalGraph, CausalDiscovery
    from agents.causal_rl_agent import CausalRLAgent, CounterfactualRLAgent
    from environments.multi_modal_env import MultiModalGridWorld, MultiModalWrapper
    from evaluation.metrics import IntegratedMetrics


class IntegratedExperiments:
    """Experimental framework for integrated causal multi-modal RL"""
    
    def __init__(self):
        self.metrics = IntegratedMetrics()
        self.results = {}
    
    def test_integrated_system(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 100,
        causal_algorithms: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test integrated causal multi-modal RL system"""
        if causal_algorithms is None:
            causal_algorithms = ["PC", "GES", "LiNGAM"]
        
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        results = {}
        
        for algorithm in causal_algorithms:
            print(f"Testing integrated system with {algorithm} causal discovery...")
            
            start_time = time.time()
            
            # Create causal graph for multi-modal RL
            variables = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'visual_features', 'text_features', 'reward']
            causal_graph = CausalGraph(variables)
            
            # Add causal relationships
            causal_graph.add_edge('agent_x', 'visual_features')
            causal_graph.add_edge('agent_y', 'visual_features')
            causal_graph.add_edge('goal_x', 'visual_features')
            causal_graph.add_edge('goal_y', 'visual_features')
            causal_graph.add_edge('agent_x', 'text_features')
            causal_graph.add_edge('agent_y', 'text_features')
            causal_graph.add_edge('goal_x', 'text_features')
            causal_graph.add_edge('goal_y', 'text_features')
            causal_graph.add_edge('visual_features', 'reward')
            causal_graph.add_edge('text_features', 'reward')
            
            # Create integrated agent
            class IntegratedCausalMultiModalAgent(CausalRLAgent):
                """Integrated causal multi-modal RL agent"""
                
                def __init__(self, wrapper, causal_graph, lr=1e-3):
                    self.wrapper = wrapper
                    state_dim = wrapper.total_dim
                    action_dim = 4
                    super().__init__(state_dim, action_dim, causal_graph, lr)
                
                def select_action(self, obs, deterministic=False):
                    """Select action from multi-modal observation"""
                    state = self.wrapper.process_observation(obs)
                    return super().select_action(state, deterministic)
                
                def train_episode(self, env, max_steps=1000):
                    """Train for one episode with multi-modal observations"""
                    obs, _ = env.reset()
                    episode_reward = 0
                    steps = 0
                    
                    states, actions, rewards, next_obss, dones = [], [], [], [], []
                    
                    while steps < max_steps:
                        action, _ = self.select_action(obs)
                        next_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        states.append(self.wrapper.process_observation(obs))
                        actions.append(action)
                        rewards.append(reward)
                        next_obss.append(self.wrapper.process_observation(next_obs))
                        dones.append(done)
                        
                        episode_reward += reward
                        steps += 1
                        obs = next_obs
                        
                        if done:
                            break
                    
                    if len(states) > 0:
                        self.update(states, actions, rewards, next_obss, dones)
                    
                    self.episode_rewards.append(episode_reward)
                    return episode_reward, steps
            
            agent = IntegratedCausalMultiModalAgent(wrapper, causal_graph, lr=1e-3)
            
            # Training
            episode_rewards = []
            for episode in range(n_episodes):
                reward, steps = agent.train_episode(env)
                episode_rewards.append(reward)
                
                # Update metrics
                self.metrics.update_rl(reward, steps)
            
            computation_time = time.time() - start_time
            
            # Calculate performance metrics
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            final_performance = np.mean(episode_rewards[-10:])  # Last 10 episodes
            
            results[algorithm] = {
                "episode_rewards": episode_rewards,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "final_performance": final_performance,
                "computation_time": computation_time,
                "causal_graph": causal_graph,
                "agent": agent
            }
            
            print(f"{algorithm}: Avg reward = {avg_reward:.3f} ± {std_reward:.3f}, "
                  f"Final performance = {final_performance:.3f}")
        
        return results
    
    def test_causal_intervention_effects(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 50,
        interventions: List[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test effects of causal interventions"""
        if interventions is None:
            interventions = [
                {},  # No intervention
                {"agent_x": 0.0},  # Fix agent x position
                {"agent_y": 0.0},  # Fix agent y position
                {"agent_x": 0.0, "agent_y": 0.0},  # Fix both positions
            ]
        
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        # Create causal graph
        variables = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'visual_features', 'text_features', 'reward']
        causal_graph = CausalGraph(variables)
        causal_graph.add_edge('agent_x', 'visual_features')
        causal_graph.add_edge('agent_y', 'visual_features')
        causal_graph.add_edge('goal_x', 'visual_features')
        causal_graph.add_edge('goal_y', 'visual_features')
        causal_graph.add_edge('agent_x', 'text_features')
        causal_graph.add_edge('agent_y', 'text_features')
        causal_graph.add_edge('goal_x', 'text_features')
        causal_graph.add_edge('goal_y', 'text_features')
        causal_graph.add_edge('visual_features', 'reward')
        causal_graph.add_edge('text_features', 'reward')
        
        # Create agent
        class InterventionTestAgent(CausalRLAgent):
            def __init__(self, wrapper, causal_graph, lr=1e-3):
                self.wrapper = wrapper
                state_dim = wrapper.total_dim
                action_dim = 4
                super().__init__(state_dim, action_dim, causal_graph, lr)
            
            def select_action(self, obs, deterministic=False):
                state = self.wrapper.process_observation(obs)
                return super().select_action(state, deterministic)
        
        agent = InterventionTestAgent(wrapper, causal_graph, lr=1e-3)
        
        results = {}
        
        for i, intervention in enumerate(interventions):
            intervention_name = f"intervention_{i}" if intervention else "no_intervention"
            print(f"Testing intervention: {intervention_name}")
            
            episode_rewards = []
            intervention_effects = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for step in range(20):
                    # Apply intervention if specified
                    if intervention:
                        # Simulate intervention effect
                        processed_obs = wrapper.process_observation(obs)
                        # Modify state based on intervention
                        if 'agent_x' in intervention:
                            processed_obs[-2] = intervention['agent_x']  # agent_x position
                        if 'agent_y' in intervention:
                            processed_obs[-1] = intervention['agent_y']  # agent_y position
                        
                        # Calculate intervention effect
                        effect = np.linalg.norm(processed_obs)
                        intervention_effects.append(effect)
                    
                    action, _ = agent.select_action(obs)
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            avg_effect = np.mean(intervention_effects) if intervention_effects else 0.0
            
            results[intervention_name] = {
                "episode_rewards": episode_rewards,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "avg_intervention_effect": avg_effect,
                "intervention": intervention
            }
            
            print(f"{intervention_name}: Avg reward = {avg_reward:.3f} ± {std_reward:.3f}, "
                  f"Avg effect = {avg_effect:.3f}")
        
        return results
    
    def test_counterfactual_reasoning(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """Test counterfactual reasoning capabilities"""
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        # Create causal graph
        variables = ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'visual_features', 'text_features', 'reward']
        causal_graph = CausalGraph(variables)
        causal_graph.add_edge('agent_x', 'visual_features')
        causal_graph.add_edge('agent_y', 'visual_features')
        causal_graph.add_edge('goal_x', 'visual_features')
        causal_graph.add_edge('goal_y', 'visual_features')
        causal_graph.add_edge('agent_x', 'text_features')
        causal_graph.add_edge('agent_y', 'text_features')
        causal_graph.add_edge('goal_x', 'text_features')
        causal_graph.add_edge('goal_y', 'text_features')
        causal_graph.add_edge('visual_features', 'reward')
        causal_graph.add_edge('text_features', 'reward')
        
        # Create counterfactual agent
        class CounterfactualTestAgent(CounterfactualRLAgent):
            def __init__(self, wrapper, causal_graph, lr=1e-3):
                self.wrapper = wrapper
                state_dim = wrapper.total_dim
                action_dim = 4
                super().__init__(state_dim, action_dim, causal_graph, lr)
            
            def select_action(self, obs, deterministic=False):
                state = self.wrapper.process_observation(obs)
                return super().select_action(state, deterministic)
        
        agent = CounterfactualTestAgent(wrapper, causal_graph, lr=1e-3)
        
        results = {
            "actual_rewards": [],
            "counterfactual_rewards": [],
            "counterfactual_quality": []
        }
        
        print("Testing counterfactual reasoning...")
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            trajectory = []
            actual_reward = 0
            
            # Collect actual trajectory
            for step in range(20):
                action, _ = agent.select_action(obs)
                next_obs, reward, done, _, _ = env.step(action)
                
                trajectory.append((obs, action, reward, next_obs, done))
                actual_reward += reward
                obs = next_obs
                
                if done:
                    break
            
            results["actual_rewards"].append(actual_reward)
            
            # Generate counterfactual scenarios
            counterfactual_scenarios = [
                {"agent_x": 0.0},  # What if agent started at x=0?
                {"agent_y": 0.0},  # What if agent started at y=0?
                {"agent_x": 0.0, "agent_y": 0.0},  # What if agent started at origin?
            ]
            
            episode_counterfactual_rewards = []
            
            for scenario in counterfactual_scenarios:
                cf_rewards = agent.counterfactual_reasoning(trajectory, scenario)
                cf_reward = np.mean(cf_rewards)
                episode_counterfactual_rewards.append(cf_reward)
            
            results["counterfactual_rewards"].append(episode_counterfactual_rewards)
            
            # Calculate counterfactual quality (how different from actual)
            cf_quality = np.std(episode_counterfactual_rewards)
            results["counterfactual_quality"].append(cf_quality)
        
        # Calculate summary statistics
        results["avg_actual_reward"] = np.mean(results["actual_rewards"])
        results["avg_counterfactual_reward"] = np.mean([np.mean(cf) for cf in results["counterfactual_rewards"]])
        results["avg_counterfactual_quality"] = np.mean(results["counterfactual_quality"])
        
        print(f"Counterfactual reasoning: Avg actual reward = {results['avg_actual_reward']:.3f}, "
              f"Avg counterfactual reward = {results['avg_counterfactual_reward']:.3f}, "
              f"Avg quality = {results['avg_counterfactual_quality']:.3f}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot integrated experimental results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Integrated system comparison
        if "integrated_system" in results:
            system_results = results["integrated_system"]
            algorithms = list(system_results.keys())
            avg_rewards = [system_results[alg]["avg_reward"] for alg in algorithms]
            final_performances = [system_results[alg]["final_performance"] for alg in algorithms]
            
            x = np.arange(len(algorithms))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, avg_rewards, width, label='Avg Reward', alpha=0.8)
            axes[0, 0].bar(x + width/2, final_performances, width, label='Final Performance', alpha=0.8)
            axes[0, 0].set_xlabel('Causal Algorithm')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Integrated System Performance')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(algorithms)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Causal intervention effects
        if "causal_interventions" in results:
            intervention_results = results["causal_interventions"]
            interventions = list(intervention_results.keys())
            avg_rewards = [intervention_results[intv]["avg_reward"] for intv in interventions]
            avg_effects = [intervention_results[intv]["avg_intervention_effect"] for intv in interventions]
            
            x = np.arange(len(interventions))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, avg_rewards, width, label='Avg Reward', alpha=0.8)
            axes[0, 1].bar(x + width/2, avg_effects, width, label='Avg Effect', alpha=0.8)
            axes[0, 1].set_xlabel('Intervention Type')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_title('Causal Intervention Effects')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(interventions, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Counterfactual reasoning
        if "counterfactual_reasoning" in results:
            cf_results = results["counterfactual_reasoning"]
            actual_rewards = cf_results["actual_rewards"]
            counterfactual_rewards = cf_results["counterfactual_rewards"]
            
            # Plot actual vs counterfactual rewards
            episodes = range(len(actual_rewards))
            axes[1, 0].plot(episodes, actual_rewards, 'o-', label='Actual', linewidth=2)
            
            # Plot average counterfactual rewards
            avg_cf_rewards = [np.mean(cf) for cf in counterfactual_rewards]
            axes[1, 0].plot(episodes, avg_cf_rewards, 's-', label='Counterfactual', linewidth=2)
            
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].set_title('Actual vs Counterfactual Rewards')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Counterfactual quality
        if "counterfactual_reasoning" in results:
            cf_results = results["counterfactual_reasoning"]
            cf_quality = cf_results["counterfactual_quality"]
            
            axes[1, 1].plot(range(len(cf_quality)), cf_quality, 'o-', linewidth=2, color='red')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Counterfactual Quality')
            axes[1, 1].set_title('Counterfactual Reasoning Quality')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_experiment(
        self, 
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive integrated experiments"""
        print("Running comprehensive integrated experiments...")
        
        # Integrated system experiment
        print("\n1. Integrated System Experiment")
        system_results = self.test_integrated_system()
        
        # Causal intervention effects experiment
        print("\n2. Causal Intervention Effects Experiment")
        intervention_results = self.test_causal_intervention_effects()
        
        # Counterfactual reasoning experiment
        print("\n3. Counterfactual Reasoning Experiment")
        counterfactual_results = self.test_counterfactual_reasoning()
        
        # Compile results
        results = {
            "integrated_system": system_results,
            "causal_interventions": intervention_results,
            "counterfactual_reasoning": counterfactual_results
        }
        
        # Plot results
        self.plot_results(results, save_path)
        
        return results
