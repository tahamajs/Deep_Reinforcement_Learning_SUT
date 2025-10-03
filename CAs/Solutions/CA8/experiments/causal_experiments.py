"""
Causal Discovery Experiments
Experimental framework for evaluating causal discovery algorithms
"""

import numpy as np
import torch
import networkx as nx
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Handle both relative and absolute imports
try:
    from ..agents.causal_discovery import CausalGraph, CausalDiscovery
    from ..evaluation.metrics import CausalDiscoveryMetrics
except ImportError:
    from agents.causal_discovery import CausalGraph, CausalDiscovery
    from evaluation.metrics import CausalDiscoveryMetrics


class CausalDiscoveryExperiments:
    """Experimental framework for causal discovery algorithms"""
    
    def __init__(self):
        self.metrics = CausalDiscoveryMetrics()
        self.results = {}
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000, 
        n_vars: int = 4,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> Tuple[np.ndarray, List[str], nx.DiGraph]:
        """Generate synthetic data with known causal structure"""
        np.random.seed(seed)
        
        # Create true causal graph
        variables = [f"X{i}" for i in range(n_vars)]
        true_graph = nx.DiGraph()
        true_graph.add_nodes_from(variables)
        
        # Add some causal edges
        if n_vars >= 2:
            true_graph.add_edge("X0", "X1")
        if n_vars >= 3:
            true_graph.add_edge("X0", "X2")
        if n_vars >= 4:
            true_graph.add_edge("X1", "X3")
            true_graph.add_edge("X2", "X3")
        
        # Generate data according to causal structure
        data = np.zeros((n_samples, n_vars))
        
        # Generate X0 (exogenous)
        data[:, 0] = np.random.normal(0, 1, n_samples)
        
        # Generate other variables based on causal structure
        if n_vars >= 2:
            data[:, 1] = 0.5 * data[:, 0] + np.random.normal(0, noise_level, n_samples)
        
        if n_vars >= 3:
            data[:, 2] = 0.3 * data[:, 0] + np.random.normal(0, noise_level, n_samples)
        
        if n_vars >= 4:
            data[:, 3] = 0.4 * data[:, 1] + 0.3 * data[:, 2] + np.random.normal(0, noise_level, n_samples)
        
        return data, variables, true_graph
    
    def run_algorithm_comparison(
        self, 
        data: np.ndarray, 
        variables: List[str],
        true_graph: nx.DiGraph,
        algorithms: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different causal discovery algorithms"""
        if algorithms is None:
            algorithms = ["PC", "GES", "LiNGAM"]
        
        results = {}
        
        for algorithm in algorithms:
            print(f"Running {algorithm} algorithm...")
            
            start_time = time.time()
            
            try:
                if algorithm == "PC":
                    discovered_graph = CausalDiscovery.pc_algorithm(data, variables)
                elif algorithm == "GES":
                    discovered_graph = CausalDiscovery.ges_algorithm(data, variables)
                elif algorithm == "LiNGAM":
                    discovered_graph = CausalDiscovery.lingam_algorithm(data, variables)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                computation_time = time.time() - start_time
                
                # Convert to NetworkX for comparison
                discovered_nx = discovered_graph.to_networkx()
                
                # Update metrics
                self.metrics.update(true_graph, discovered_nx, computation_time)
                metrics = self.metrics.get_metrics()
                
                results[algorithm] = {
                    "graph": discovered_graph,
                    "nx_graph": discovered_nx,
                    "computation_time": computation_time,
                    "metrics": metrics
                }
                
                print(f"{algorithm} completed in {computation_time:.3f}s")
                print(f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
                
            except Exception as e:
                print(f"{algorithm} failed: {e}")
                results[algorithm] = {
                    "error": str(e),
                    "computation_time": time.time() - start_time
                }
        
        return results
    
    def run_scalability_experiment(
        self, 
        n_vars_range: List[int] = [3, 4, 5, 6, 7, 8],
        n_samples: int = 1000,
        algorithms: List[str] = None
    ) -> Dict[str, Dict[int, float]]:
        """Test algorithm scalability with different numbers of variables"""
        if algorithms is None:
            algorithms = ["PC", "GES", "LiNGAM"]
        
        scalability_results = {alg: {} for alg in algorithms}
        
        for n_vars in n_vars_range:
            print(f"Testing with {n_vars} variables...")
            
            # Generate data
            data, variables, true_graph = self.generate_synthetic_data(
                n_samples=n_samples, n_vars=n_vars
            )
            
            for algorithm in algorithms:
                try:
                    start_time = time.time()
                    
                    if algorithm == "PC":
                        discovered_graph = CausalDiscovery.pc_algorithm(data, variables)
                    elif algorithm == "GES":
                        discovered_graph = CausalDiscovery.ges_algorithm(data, variables)
                    elif algorithm == "LiNGAM":
                        discovered_graph = CausalDiscovery.lingam_algorithm(data, variables)
                    
                    computation_time = time.time() - start_time
                    scalability_results[algorithm][n_vars] = computation_time
                    
                except Exception as e:
                    print(f"{algorithm} failed with {n_vars} variables: {e}")
                    scalability_results[algorithm][n_vars] = float('inf')
        
        return scalability_results
    
    def run_noise_robustness_experiment(
        self,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
        n_samples: int = 1000,
        n_vars: int = 4,
        algorithms: List[str] = None
    ) -> Dict[str, Dict[float, Dict[str, float]]]:
        """Test algorithm robustness to noise"""
        if algorithms is None:
            algorithms = ["PC", "GES", "LiNGAM"]
        
        robustness_results = {alg: {} for alg in algorithms}
        
        for noise_level in noise_levels:
            print(f"Testing with noise level {noise_level}...")
            
            # Generate data
            data, variables, true_graph = self.generate_synthetic_data(
                n_samples=n_samples, n_vars=n_vars, noise_level=noise_level
            )
            
            for algorithm in algorithms:
                try:
                    if algorithm == "PC":
                        discovered_graph = CausalDiscovery.pc_algorithm(data, variables)
                    elif algorithm == "GES":
                        discovered_graph = CausalDiscovery.ges_algorithm(data, variables)
                    elif algorithm == "LiNGAM":
                        discovered_graph = CausalDiscovery.lingam_algorithm(data, variables)
                    
                    # Convert to NetworkX for comparison
                    discovered_nx = discovered_graph.to_networkx()
                    
                    # Calculate metrics
                    self.metrics.reset()
                    self.metrics.update(true_graph, discovered_nx)
                    metrics = self.metrics.get_metrics()
                    
                    robustness_results[algorithm][noise_level] = metrics
                    
                except Exception as e:
                    print(f"{algorithm} failed with noise {noise_level}: {e}")
                    robustness_results[algorithm][noise_level] = {
                        "accuracy": 0.0, "f1_score": 0.0, "error": str(e)
                    }
        
        return robustness_results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot experimental results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Algorithm comparison
        if "algorithm_comparison" in results:
            comp_results = results["algorithm_comparison"]
            algorithms = list(comp_results.keys())
            accuracies = [comp_results[alg].get("metrics", {}).get("accuracy", 0.0) 
                         for alg in algorithms]
            f1_scores = [comp_results[alg].get("metrics", {}).get("f1_score", 0.0) 
                        for alg in algorithms]
            
            x = np.arange(len(algorithms))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            axes[0, 0].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
            axes[0, 0].set_xlabel('Algorithm')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Algorithm Performance Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(algorithms)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Scalability results
        if "scalability" in results:
            scal_results = results["scalability"]
            for algorithm, times in scal_results.items():
                n_vars = list(times.keys())
                comp_times = list(times.values())
                axes[0, 1].plot(n_vars, comp_times, 'o-', label=algorithm, linewidth=2)
            
            axes[0, 1].set_xlabel('Number of Variables')
            axes[0, 1].set_ylabel('Computation Time (s)')
            axes[0, 1].set_title('Algorithm Scalability')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Noise robustness
        if "noise_robustness" in results:
            noise_results = results["noise_robustness"]
            for algorithm, noise_data in noise_results.items():
                noise_levels = list(noise_data.keys())
                accuracies = [noise_data[level].get("accuracy", 0.0) 
                             for level in noise_levels]
                axes[1, 0].plot(noise_levels, accuracies, 'o-', label=algorithm, linewidth=2)
            
            axes[1, 0].set_xlabel('Noise Level')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Noise Robustness')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        if "algorithm_comparison" in results:
            comp_results = results["algorithm_comparison"]
            algorithms = list(comp_results.keys())
            times = [comp_results[alg].get("computation_time", 0.0) 
                    for alg in algorithms]
            shd = [comp_results[alg].get("metrics", {}).get("structural_hamming_distance", 0.0) 
                  for alg in algorithms]
            
            axes[1, 1].scatter(times, shd, s=100, alpha=0.7)
            for i, alg in enumerate(algorithms):
                axes[1, 1].annotate(alg, (times[i], shd[i]), 
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('Computation Time (s)')
            axes[1, 1].set_ylabel('Structural Hamming Distance')
            axes[1, 1].set_title('Time vs Accuracy Trade-off')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comprehensive_experiment(
        self, 
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive causal discovery experiments"""
        print("Running comprehensive causal discovery experiments...")
        
        # Generate test data
        data, variables, true_graph = self.generate_synthetic_data()
        
        # Algorithm comparison
        print("\n1. Algorithm Comparison")
        algorithm_results = self.run_algorithm_comparison(data, variables, true_graph)
        
        # Scalability experiment
        print("\n2. Scalability Experiment")
        scalability_results = self.run_scalability_experiment()
        
        # Noise robustness experiment
        print("\n3. Noise Robustness Experiment")
        noise_results = self.run_noise_robustness_experiment()
        
        # Compile results
        results = {
            "algorithm_comparison": algorithm_results,
            "scalability": scalability_results,
            "noise_robustness": noise_results
        }
        
        # Plot results
        self.plot_results(results, save_path)
        
        return results
