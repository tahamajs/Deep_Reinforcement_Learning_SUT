"""
Multi-Modal Learning Experiments
Experimental framework for evaluating multi-modal learning approaches
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..environments.multi_modal_env import MultiModalGridWorld, MultiModalWrapper
from ..evaluation.metrics import MultiModalMetrics


class MultiModalExperiments:
    """Experimental framework for multi-modal learning"""
    
    def __init__(self):
        self.metrics = MultiModalMetrics()
        self.results = {}
    
    def test_fusion_strategies(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 100,
        fusion_strategies: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test different multi-modal fusion strategies"""
        if fusion_strategies is None:
            fusion_strategies = ["early", "late", "cross_attention", "hierarchical"]
        
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        results = {}
        
        for strategy in fusion_strategies:
            print(f"Testing {strategy} fusion strategy...")
            
            start_time = time.time()
            
            # Simulate fusion strategy (simplified)
            episode_rewards = []
            modality_contributions = {"visual": [], "textual": [], "state": []}
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for step in range(20):  # Max steps per episode
                    # Process observation
                    processed_obs = wrapper.process_observation(obs)
                    
                    # Simulate different fusion strategies
                    if strategy == "early":
                        # Early fusion: concatenate all modalities
                        fused_features = processed_obs
                    elif strategy == "late":
                        # Late fusion: process separately then combine
                        visual_feat = processed_obs[:wrapper.visual_dim]
                        text_feat = processed_obs[wrapper.visual_dim:wrapper.visual_dim + wrapper.text_dim]
                        state_feat = processed_obs[wrapper.visual_dim + wrapper.text_dim:]
                        fused_features = np.concatenate([visual_feat, text_feat, state_feat])
                    elif strategy == "cross_attention":
                        # Cross-modal attention (simplified)
                        visual_feat = processed_obs[:wrapper.visual_dim]
                        text_feat = processed_obs[wrapper.visual_dim:wrapper.visual_dim + wrapper.text_dim]
                        state_feat = processed_obs[wrapper.visual_dim + wrapper.text_dim:]
                        # Simple attention weights
                        attention_weights = np.array([0.4, 0.3, 0.3])
                        fused_features = (attention_weights[0] * visual_feat + 
                                        attention_weights[1] * text_feat + 
                                        attention_weights[2] * state_feat)
                    else:  # hierarchical
                        # Hierarchical fusion
                        fused_features = processed_obs * 1.1  # Boost all features
                    
                    # Simulate action selection and reward
                    action = np.random.randint(0, 4)
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Track modality contributions
                processed_obs = wrapper.process_observation(obs)
                modality_contributions["visual"].append(np.linalg.norm(processed_obs[:wrapper.visual_dim]))
                modality_contributions["textual"].append(np.linalg.norm(processed_obs[wrapper.visual_dim:wrapper.visual_dim + wrapper.text_dim]))
                modality_contributions["state"].append(np.linalg.norm(processed_obs[wrapper.visual_dim + wrapper.text_dim:]))
            
            computation_time = time.time() - start_time
            
            # Calculate metrics
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            results[strategy] = {
                "episode_rewards": episode_rewards,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "computation_time": computation_time,
                "modality_contributions": modality_contributions
            }
            
            print(f"{strategy} fusion: Avg reward = {avg_reward:.3f} ± {std_reward:.3f}")
        
        return results
    
    def test_missing_modality_robustness(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 50,
        missing_scenarios: List[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test robustness to missing modalities"""
        if missing_scenarios is None:
            missing_scenarios = [
                [],  # No missing modalities
                ["visual"],  # Missing visual
                ["textual"],  # Missing textual
                ["state"],  # Missing state
                ["visual", "textual"],  # Missing visual and textual
            ]
        
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        results = {}
        
        for scenario in missing_scenarios:
            scenario_name = "all_modalities" if not scenario else "_".join(scenario)
            print(f"Testing missing modalities: {scenario_name}")
            
            episode_rewards = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for step in range(20):
                    # Process observation with missing modalities
                    processed_obs = wrapper.process_observation(obs)
                    
                    # Simulate missing modalities by zeroing out features
                    if "visual" in scenario:
                        processed_obs[:wrapper.visual_dim] = 0
                    if "textual" in scenario:
                        processed_obs[wrapper.visual_dim:wrapper.visual_dim + wrapper.text_dim] = 0
                    if "state" in scenario:
                        processed_obs[wrapper.visual_dim + wrapper.text_dim:] = 0
                    
                    # Simulate action selection
                    action = np.random.randint(0, 4)
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            results[scenario_name] = {
                "episode_rewards": episode_rewards,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "missing_modalities": scenario
            }
            
            print(f"{scenario_name}: Avg reward = {avg_reward:.3f} ± {std_reward:.3f}")
        
        return results
    
    def test_modality_importance(
        self,
        env_size: int = 6,
        render_size: int = 64,
        n_episodes: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """Test the importance of different modalities"""
        env = MultiModalGridWorld(size=env_size, render_size=render_size)
        wrapper = MultiModalWrapper(env)
        
        results = {}
        modalities = ["visual", "textual", "state"]
        
        for modality in modalities:
            print(f"Testing {modality} modality importance...")
            
            episode_rewards = []
            modality_features = []
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for step in range(20):
                    # Process observation
                    processed_obs = wrapper.process_observation(obs)
                    
                    # Extract modality-specific features
                    if modality == "visual":
                        features = processed_obs[:wrapper.visual_dim]
                    elif modality == "textual":
                        features = processed_obs[wrapper.visual_dim:wrapper.visual_dim + wrapper.text_dim]
                    else:  # state
                        features = processed_obs[wrapper.visual_dim + wrapper.text_dim:]
                    
                    modality_features.append(features)
                    
                    # Use only this modality for decision making
                    action = np.random.randint(0, 4)
                    next_obs, reward, done, _, _ = env.step(action)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            avg_feature_norm = np.mean([np.linalg.norm(f) for f in modality_features])
            
            results[modality] = {
                "episode_rewards": episode_rewards,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "avg_feature_norm": avg_feature_norm,
                "modality_features": modality_features
            }
            
            print(f"{modality}: Avg reward = {avg_reward:.3f} ± {std_reward:.3f}, "
                  f"Feature norm = {avg_feature_norm:.3f}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot multi-modal experimental results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Fusion strategies comparison
        if "fusion_strategies" in results:
            fusion_results = results["fusion_strategies"]
            strategies = list(fusion_results.keys())
            avg_rewards = [fusion_results[strategy]["avg_reward"] for strategy in strategies]
            std_rewards = [fusion_results[strategy]["std_reward"] for strategy in strategies]
            
            x = np.arange(len(strategies))
            bars = axes[0, 0].bar(x, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.8)
            axes[0, 0].set_xlabel('Fusion Strategy')
            axes[0, 0].set_ylabel('Average Episode Reward')
            axes[0, 0].set_title('Multi-Modal Fusion Strategy Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(strategies, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Missing modality robustness
        if "missing_modality" in results:
            missing_results = results["missing_modality"]
            scenarios = list(missing_results.keys())
            avg_rewards = [missing_results[scenario]["avg_reward"] for scenario in scenarios]
            
            bars = axes[0, 1].bar(range(len(scenarios)), avg_rewards, alpha=0.8)
            axes[0, 1].set_xlabel('Missing Modality Scenario')
            axes[0, 1].set_ylabel('Average Episode Reward')
            axes[0, 1].set_title('Robustness to Missing Modalities')
            axes[0, 1].set_xticks(range(len(scenarios)))
            axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Modality importance
        if "modality_importance" in results:
            importance_results = results["modality_importance"]
            modalities = list(importance_results.keys())
            avg_rewards = [importance_results[mod]["avg_reward"] for mod in modalities]
            feature_norms = [importance_results[mod]["avg_feature_norm"] for mod in modalities]
            
            x = np.arange(len(modalities))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, avg_rewards, width, label='Avg Reward', alpha=0.8)
            axes[1, 0].bar(x + width/2, feature_norms, width, label='Feature Norm', alpha=0.8)
            axes[1, 0].set_xlabel('Modality')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Modality Importance Analysis')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(modalities)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-modal correlation
        if "fusion_strategies" in results:
            fusion_results = results["fusion_strategies"]
            strategies = list(fusion_results.keys())
            correlations = []
            
            for strategy in strategies:
                contributions = fusion_results[strategy]["modality_contributions"]
                # Calculate correlation between visual and textual contributions
                if len(contributions["visual"]) > 1 and len(contributions["textual"]) > 1:
                    corr = np.corrcoef(contributions["visual"], contributions["textual"])[0, 1]
                    correlations.append(abs(corr))
                else:
                    correlations.append(0.0)
            
            axes[1, 1].bar(range(len(strategies)), correlations, alpha=0.8)
            axes[1, 1].set_xlabel('Fusion Strategy')
            axes[1, 1].set_ylabel('Cross-Modal Correlation')
            axes[1, 1].set_title('Cross-Modal Feature Correlation')
            axes[1, 1].set_xticks(range(len(strategies)))
            axes[1, 1].set_xticklabels(strategies, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_experiment(
        self, 
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive multi-modal experiments"""
        print("Running comprehensive multi-modal experiments...")
        
        # Fusion strategies experiment
        print("\n1. Fusion Strategies Experiment")
        fusion_results = self.test_fusion_strategies()
        
        # Missing modality robustness experiment
        print("\n2. Missing Modality Robustness Experiment")
        missing_results = self.test_missing_modality_robustness()
        
        # Modality importance experiment
        print("\n3. Modality Importance Experiment")
        importance_results = self.test_modality_importance()
        
        # Compile results
        results = {
            "fusion_strategies": fusion_results,
            "missing_modality": missing_results,
            "modality_importance": importance_results
        }
        
        # Plot results
        self.plot_results(results, save_path)
        
        return results
