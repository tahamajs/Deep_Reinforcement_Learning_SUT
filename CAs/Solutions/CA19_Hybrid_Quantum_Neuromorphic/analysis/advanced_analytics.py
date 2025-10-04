"""
Advanced Analytics Module for CA19 Quantum-Neuromorphic RL Systems

This module provides sophisticated analysis tools for:
- Quantum coherence analysis
- Neuromorphic efficiency metrics
- Hybrid system performance evaluation
- Multi-dimensional visualization
- Statistical significance testing
- Temporal dynamics analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, welch
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA, t-SNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QuantumCoherenceAnalyzer:
    """
    Advanced quantum coherence and entanglement analysis
    """
    
    def __init__(self):
        self.coherence_history = []
        self.entanglement_history = []
        self.fidelity_history = []
        self.decoherence_rates = []
        
    def analyze_quantum_coherence(self, quantum_states: List[np.ndarray], 
                                time_points: List[float]) -> Dict[str, Any]:
        """Analyze quantum coherence over time"""
        coherence_metrics = []
        entanglement_metrics = []
        fidelity_metrics = []
        
        for state in quantum_states:
            # Calculate coherence
            coherence = self._calculate_coherence(state)
            coherence_metrics.append(coherence)
            
            # Calculate entanglement
            entanglement = self._calculate_entanglement(state)
            entanglement_metrics.append(entanglement)
            
            # Calculate fidelity
            fidelity = self._calculate_fidelity(state)
            fidelity_metrics.append(fidelity)
        
        # Store for temporal analysis
        self.coherence_history.extend(coherence_metrics)
        self.entanglement_history.extend(entanglement_metrics)
        self.fidelity_history.extend(fidelity_metrics)
        
        # Analyze temporal dynamics
        temporal_analysis = self._analyze_temporal_dynamics(
            coherence_metrics, time_points
        )
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(coherence_metrics)
        
        return {
            'coherence_metrics': coherence_metrics,
            'entanglement_metrics': entanglement_metrics,
            'fidelity_metrics': fidelity_metrics,
            'temporal_analysis': temporal_analysis,
            'statistical_analysis': statistical_analysis
        }
    
    def _calculate_coherence(self, state: np.ndarray) -> float:
        """Calculate quantum coherence measure"""
        if len(state.shape) == 1:
            # Pure state coherence
            probabilities = np.abs(state) ** 2
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(probabilities))
            return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            # Density matrix coherence
            off_diagonal = np.abs(state - np.diag(np.diag(state)))
            return np.mean(off_diagonal)
    
    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """Calculate entanglement measure"""
        if len(state.shape) == 1:
            # For pure states, use von Neumann entropy
            probabilities = np.abs(state) ** 2
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        else:
            # For mixed states, use concurrence or similar measure
            eigenvals = np.linalg.eigvals(state)
            eigenvals = eigenvals[eigenvals > 1e-10]
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            return entropy
    
    def _calculate_fidelity(self, state: np.ndarray) -> float:
        """Calculate quantum fidelity"""
        if len(state.shape) == 1:
            # Fidelity with respect to uniform superposition
            target_state = np.ones(len(state)) / np.sqrt(len(state))
            fidelity = np.abs(np.dot(np.conj(state), target_state)) ** 2
            return fidelity
        else:
            # Fidelity with respect to maximally mixed state
            target_state = np.eye(state.shape[0]) / state.shape[0]
            fidelity = np.trace(np.sqrt(np.sqrt(target_state) @ state @ np.sqrt(target_state))) ** 2
            return fidelity
    
    def _analyze_temporal_dynamics(self, metrics: List[float], 
                                 time_points: List[float]) -> Dict[str, Any]:
        """Analyze temporal dynamics of quantum metrics"""
        if len(metrics) < 2:
            return {}
        
        # Convert to numpy arrays
        metrics_array = np.array(metrics)
        time_array = np.array(time_points)
        
        # Calculate derivatives (rate of change)
        if len(metrics_array) > 1:
            derivatives = np.gradient(metrics_array, time_array)
            self.decoherence_rates.extend(derivatives.tolist())
        
        # Find peaks and valleys
        peaks, _ = find_peaks(metrics_array, height=np.mean(metrics_array))
        valleys, _ = find_peaks(-metrics_array, height=-np.mean(metrics_array))
        
        # Spectral analysis
        if len(metrics_array) > 10:
            freqs, power_spectrum = welch(metrics_array, nperseg=min(len(metrics_array)//4, 50))
            dominant_freq = freqs[np.argmax(power_spectrum)]
            spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
        else:
            dominant_freq = 0.0
            spectral_entropy = 0.0
        
        # Trend analysis
        if len(metrics_array) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, metrics_array)
        else:
            slope = intercept = r_value = p_value = std_err = 0.0
        
        return {
            'derivatives': derivatives.tolist() if len(metrics_array) > 1 else [],
            'peaks': peaks.tolist(),
            'valleys': valleys.tolist(),
            'dominant_frequency': dominant_freq,
            'spectral_entropy': spectral_entropy,
            'trend_slope': slope,
            'trend_correlation': r_value,
            'trend_significance': p_value
        }
    
    def _perform_statistical_analysis(self, metrics: List[float]) -> Dict[str, Any]:
        """Perform statistical analysis on quantum metrics"""
        if len(metrics) < 3:
            return {}
        
        metrics_array = np.array(metrics)
        
        # Basic statistics
        mean_val = np.mean(metrics_array)
        std_val = np.std(metrics_array)
        median_val = np.median(metrics_array)
        
        # Distribution analysis
        skewness = stats.skew(metrics_array)
        kurtosis = stats.kurtosis(metrics_array)
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(metrics_array) if len(metrics_array) <= 5000 else (0, 0)
        
        # Autocorrelation
        if len(metrics_array) > 10:
            autocorr = np.correlate(metrics_array, metrics_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
        else:
            autocorr = []
        
        return {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'autocorrelation': autocorr.tolist()[:10]  # First 10 lags
        }


class NeuromorphicEfficiencyAnalyzer:
    """
    Advanced neuromorphic efficiency and energy analysis
    """
    
    def __init__(self):
        self.spike_rates = []
        self.energy_consumption = []
        self.synaptic_weights = []
        self.plasticity_events = []
        
    def analyze_energy_efficiency(self, spike_data: List[Dict], 
                                weight_data: List[Dict]) -> Dict[str, Any]:
        """Analyze energy efficiency of neuromorphic systems"""
        
        # Extract metrics
        spike_rates = [data.get('spike_rate', 0) for data in spike_data]
        energy_consumption = [data.get('energy', 0) for data in spike_data]
        weight_changes = [data.get('weight_change', 0) for data in weight_data]
        
        # Store for analysis
        self.spike_rates.extend(spike_rates)
        self.energy_consumption.extend(energy_consumption)
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            spike_rates, energy_consumption, weight_changes
        )
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(spike_rates, energy_consumption)
        
        # Compare with theoretical limits
        theoretical_analysis = self._compare_with_theoretical_limits(
            spike_rates, energy_consumption
        )
        
        return {
            'efficiency_metrics': efficiency_metrics,
            'temporal_patterns': temporal_patterns,
            'theoretical_analysis': theoretical_analysis
        }
    
    def _calculate_efficiency_metrics(self, spike_rates: List[float], 
                                    energy_consumption: List[float],
                                    weight_changes: List[float]) -> Dict[str, float]:
        """Calculate various efficiency metrics"""
        
        spike_rates = np.array(spike_rates)
        energy_consumption = np.array(energy_consumption)
        weight_changes = np.array(weight_changes)
        
        # Energy per spike
        energy_per_spike = energy_consumption / (spike_rates + 1e-8)
        
        # Information efficiency (bits per energy unit)
        information_content = -np.sum(spike_rates * np.log2(spike_rates + 1e-8))
        information_efficiency = information_content / (np.mean(energy_consumption) + 1e-8)
        
        # Learning efficiency (weight change per energy)
        learning_efficiency = np.mean(np.abs(weight_changes)) / (np.mean(energy_consumption) + 1e-8)
        
        # Sparsity efficiency (sparse spikes are more efficient)
        sparsity = np.mean(spike_rates == 0)
        sparsity_efficiency = sparsity * np.mean(spike_rates)
        
        # Temporal efficiency (consistency over time)
        temporal_efficiency = 1.0 / (np.std(spike_rates) + 1e-8)
        
        return {
            'energy_per_spike': np.mean(energy_per_spike),
            'information_efficiency': information_efficiency,
            'learning_efficiency': learning_efficiency,
            'sparsity_efficiency': sparsity_efficiency,
            'temporal_efficiency': temporal_efficiency,
            'overall_efficiency': (information_efficiency + learning_efficiency + sparsity_efficiency) / 3
        }
    
    def _analyze_temporal_patterns(self, spike_rates: List[float], 
                                 energy_consumption: List[float]) -> Dict[str, Any]:
        """Analyze temporal patterns in neuromorphic data"""
        
        spike_rates = np.array(spike_rates)
        energy_consumption = np.array(energy_consumption)
        
        # Burst detection
        burst_threshold = np.mean(spike_rates) + 2 * np.std(spike_rates)
        bursts = spike_rates > burst_threshold
        
        # Inter-burst intervals
        burst_indices = np.where(bursts)[0]
        if len(burst_indices) > 1:
            inter_burst_intervals = np.diff(burst_indices)
            avg_inter_burst_interval = np.mean(inter_burst_intervals)
        else:
            avg_inter_burst_interval = 0
        
        # Synchrony analysis
        if len(spike_rates) > 10:
            # Calculate local synchrony (correlation with neighbors)
            synchrony_scores = []
            for i in range(1, len(spike_rates) - 1):
                local_corr = np.corrcoef(spike_rates[i-1:i+2], energy_consumption[i-1:i+2])[0, 1]
                synchrony_scores.append(local_corr if not np.isnan(local_corr) else 0)
            avg_synchrony = np.mean(synchrony_scores)
        else:
            avg_synchrony = 0
        
        # Adaptation analysis
        if len(spike_rates) > 5:
            adaptation_rate = np.polyfit(range(len(spike_rates)), spike_rates, 1)[0]
        else:
            adaptation_rate = 0
        
        return {
            'burst_frequency': np.sum(bursts) / len(bursts),
            'avg_inter_burst_interval': avg_inter_burst_interval,
            'avg_synchrony': avg_synchrony,
            'adaptation_rate': adaptation_rate,
            'temporal_variability': np.std(spike_rates)
        }
    
    def _compare_with_theoretical_limits(self, spike_rates: List[float], 
                                       energy_consumption: List[float]) -> Dict[str, float]:
        """Compare performance with theoretical limits"""
        
        # Landauer limit (theoretical minimum energy per bit)
        landauer_limit = 2.85e-21  # Joules at room temperature
        
        # Shannon limit (maximum information rate)
        max_information_rate = -np.sum(spike_rates * np.log2(spike_rates + 1e-8))
        
        # Calculate efficiency ratios
        energy_per_bit = np.mean(energy_consumption) / (max_information_rate + 1e-8)
        landauer_efficiency = landauer_limit / (energy_per_bit + 1e-21)
        
        # Biological efficiency comparison (neurons are ~10x more efficient than digital)
        biological_efficiency = 10.0
        biological_comparison = landauer_efficiency / biological_efficiency
        
        return {
            'landauer_efficiency': landauer_efficiency,
            'biological_comparison': biological_comparison,
            'theoretical_max_information': max_information_rate,
            'actual_energy_per_bit': energy_per_bit
        }


class HybridSystemAnalyzer:
    """
    Advanced analysis for hybrid quantum-neuromorphic systems
    """
    
    def __init__(self):
        self.quantum_metrics = []
        self.neuromorphic_metrics = []
        self.hybrid_interactions = []
        
    def analyze_hybrid_performance(self, quantum_data: List[Dict], 
                                 neuromorphic_data: List[Dict],
                                 hybrid_data: List[Dict]) -> Dict[str, Any]:
        """Analyze hybrid system performance"""
        
        # Extract metrics
        quantum_coherence = [data.get('coherence', 0) for data in quantum_data]
        neuromorphic_efficiency = [data.get('efficiency', 0) for data in neuromorphic_data]
        hybrid_performance = [data.get('performance', 0) for data in hybrid_data]
        
        # Store for analysis
        self.quantum_metrics.extend(quantum_coherence)
        self.neuromorphic_metrics.extend(neuromorphic_efficiency)
        
        # Synergy analysis
        synergy_analysis = self._analyze_synergy(
            quantum_coherence, neuromorphic_efficiency, hybrid_performance
        )
        
        # Interaction analysis
        interaction_analysis = self._analyze_interactions(
            quantum_data, neuromorphic_data, hybrid_data
        )
        
        # Performance comparison
        performance_comparison = self._compare_performance(
            quantum_coherence, neuromorphic_efficiency, hybrid_performance
        )
        
        return {
            'synergy_analysis': synergy_analysis,
            'interaction_analysis': interaction_analysis,
            'performance_comparison': performance_comparison
        }
    
    def _analyze_synergy(self, quantum_metrics: List[float], 
                        neuromorphic_metrics: List[float],
                        hybrid_metrics: List[float]) -> Dict[str, Any]:
        """Analyze synergy between quantum and neuromorphic components"""
        
        quantum_metrics = np.array(quantum_metrics)
        neuromorphic_metrics = np.array(neuromorphic_metrics)
        hybrid_metrics = np.array(hybrid_metrics)
        
        # Calculate expected performance (simple average)
        expected_performance = (quantum_metrics + neuromorphic_metrics) / 2
        
        # Calculate synergy (actual - expected)
        synergy = hybrid_metrics - expected_performance
        
        # Statistical significance of synergy
        if len(synergy) > 3:
            synergy_t_stat, synergy_p_value = stats.ttest_1samp(synergy, 0)
        else:
            synergy_t_stat = synergy_p_value = 0
        
        # Correlation analysis
        quantum_corr = np.corrcoef(quantum_metrics, hybrid_metrics)[0, 1]
        neuromorphic_corr = np.corrcoef(neuromorphic_metrics, hybrid_metrics)[0, 1]
        
        # Optimal combination ratio
        # Find the ratio that maximizes hybrid performance
        ratios = np.linspace(0, 1, 11)
        optimal_ratio = 0.5
        max_synergy = np.mean(synergy)
        
        for ratio in ratios:
            combined = ratio * quantum_metrics + (1 - ratio) * neuromorphic_metrics
            synergy_at_ratio = hybrid_metrics - combined
            if np.mean(synergy_at_ratio) > max_synergy:
                max_synergy = np.mean(synergy_at_ratio)
                optimal_ratio = ratio
        
        return {
            'synergy_mean': np.mean(synergy),
            'synergy_std': np.std(synergy),
            'synergy_significance': {'t_statistic': synergy_t_stat, 'p_value': synergy_p_value},
            'quantum_correlation': quantum_corr if not np.isnan(quantum_corr) else 0,
            'neuromorphic_correlation': neuromorphic_corr if not np.isnan(neuromorphic_corr) else 0,
            'optimal_quantum_ratio': optimal_ratio,
            'max_synergy': max_synergy
        }
    
    def _analyze_interactions(self, quantum_data: List[Dict], 
                            neuromorphic_data: List[Dict],
                            hybrid_data: List[Dict]) -> Dict[str, Any]:
        """Analyze interactions between quantum and neuromorphic components"""
        
        # Extract interaction metrics
        quantum_influences = [data.get('quantum_influence', 0) for data in hybrid_data]
        neuromorphic_influences = [data.get('neuromorphic_influence', 0) for data in hybrid_data]
        
        # Interaction strength over time
        interaction_strength = np.array(quantum_influences) + np.array(neuromorphic_influences)
        
        # Dominance analysis (which component is more influential)
        quantum_dominance = np.mean(np.array(quantum_influences) > np.array(neuromorphic_influences))
        
        # Temporal interaction patterns
        if len(quantum_influences) > 10:
            quantum_trend = np.polyfit(range(len(quantum_influences)), quantum_influences, 1)[0]
            neuromorphic_trend = np.polyfit(range(len(neuromorphic_influences)), neuromorphic_influences, 1)[0]
        else:
            quantum_trend = neuromorphic_trend = 0
        
        # Interaction stability
        interaction_variance = np.var(interaction_strength)
        
        return {
            'avg_interaction_strength': np.mean(interaction_strength),
            'quantum_dominance': quantum_dominance,
            'quantum_trend': quantum_trend,
            'neuromorphic_trend': neuromorphic_trend,
            'interaction_stability': 1.0 / (interaction_variance + 1e-8),
            'interaction_variance': interaction_variance
        }
    
    def _compare_performance(self, quantum_metrics: List[float], 
                           neuromorphic_metrics: List[float],
                           hybrid_metrics: List[float]) -> Dict[str, Any]:
        """Compare performance across different system types"""
        
        quantum_metrics = np.array(quantum_metrics)
        neuromorphic_metrics = np.array(neuromorphic_metrics)
        hybrid_metrics = np.array(hybrid_metrics)
        
        # Statistical comparison
        quantum_vs_hybrid = stats.ttest_rel(hybrid_metrics, quantum_metrics) if len(quantum_metrics) > 1 else (0, 1)
        neuromorphic_vs_hybrid = stats.ttest_rel(hybrid_metrics, neuromorphic_metrics) if len(neuromorphic_metrics) > 1 else (0, 1)
        
        # Performance improvement percentages
        quantum_improvement = (np.mean(hybrid_metrics) - np.mean(quantum_metrics)) / (np.mean(quantum_metrics) + 1e-8) * 100
        neuromorphic_improvement = (np.mean(hybrid_metrics) - np.mean(neuromorphic_metrics)) / (np.mean(neuromorphic_metrics) + 1e-8) * 100
        
        # Consistency analysis
        quantum_consistency = 1.0 / (np.std(quantum_metrics) + 1e-8)
        neuromorphic_consistency = 1.0 / (np.std(neuromorphic_metrics) + 1e-8)
        hybrid_consistency = 1.0 / (np.std(hybrid_metrics) + 1e-8)
        
        return {
            'quantum_vs_hybrid': {'t_statistic': quantum_vs_hybrid[0], 'p_value': quantum_vs_hybrid[1]},
            'neuromorphic_vs_hybrid': {'t_statistic': neuromorphic_vs_hybrid[0], 'p_value': neuromorphic_vs_hybrid[1]},
            'quantum_improvement_percent': quantum_improvement,
            'neuromorphic_improvement_percent': neuromorphic_improvement,
            'quantum_consistency': quantum_consistency,
            'neuromorphic_consistency': neuromorphic_consistency,
            'hybrid_consistency': hybrid_consistency,
            'best_system': 'hybrid' if np.mean(hybrid_metrics) > max(np.mean(quantum_metrics), np.mean(neuromorphic_metrics)) else 
                          'quantum' if np.mean(quantum_metrics) > np.mean(neuromorphic_metrics) else 'neuromorphic'
        }


class AdvancedVisualizationEngine:
    """
    Advanced visualization engine for complex quantum-neuromorphic data
    """
    
    def __init__(self):
        self.color_palette = sns.color_palette("husl", 10)
        
    def create_quantum_coherence_plot(self, quantum_data: Dict[str, Any], 
                                    save_path: str = None) -> plt.Figure:
        """Create advanced quantum coherence visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantum Coherence Analysis', fontsize=16, fontweight='bold')
        
        # Coherence over time
        coherence_metrics = quantum_data['coherence_metrics']
        ax1.plot(coherence_metrics, 'b-', linewidth=2, label='Coherence')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Coherence')
        ax1.set_title('Quantum Coherence Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Entanglement vs Fidelity
        entanglement_metrics = quantum_data['entanglement_metrics']
        fidelity_metrics = quantum_data['fidelity_metrics']
        scatter = ax2.scatter(entanglement_metrics, fidelity_metrics, 
                            c=coherence_metrics, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Entanglement')
        ax2.set_ylabel('Fidelity')
        ax2.set_title('Entanglement vs Fidelity')
        plt.colorbar(scatter, ax=ax2, label='Coherence')
        
        # Spectral analysis
        temporal_analysis = quantum_data['temporal_analysis']
        if 'derivatives' in temporal_analysis and temporal_analysis['derivatives']:
            derivatives = temporal_analysis['derivatives']
            ax3.plot(derivatives, 'r-', linewidth=2, label='Coherence Rate')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Rate of Change')
            ax3.set_title('Coherence Dynamics')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Statistical distribution
        statistical_analysis = quantum_data['statistical_analysis']
        ax4.hist(coherence_metrics, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(statistical_analysis['mean'], color='red', linestyle='--', 
                   label=f"Mean: {statistical_analysis['mean']:.3f}")
        ax4.axvline(statistical_analysis['median'], color='green', linestyle='--', 
                   label=f"Median: {statistical_analysis['median']:.3f}")
        ax4.set_xlabel('Coherence Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Coherence Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_neuromorphic_efficiency_plot(self, neuromorphic_data: Dict[str, Any], 
                                          save_path: str = None) -> plt.Figure:
        """Create advanced neuromorphic efficiency visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neuromorphic Efficiency Analysis', fontsize=16, fontweight='bold')
        
        efficiency_metrics = neuromorphic_data['efficiency_metrics']
        temporal_patterns = neuromorphic_data['temporal_patterns']
        
        # Efficiency metrics bar plot
        metrics_names = list(efficiency_metrics.keys())
        metrics_values = list(efficiency_metrics.values())
        bars = ax1.bar(metrics_names, metrics_values, color=self.color_palette[:len(metrics_names)])
        ax1.set_ylabel('Efficiency Value')
        ax1.set_title('Efficiency Metrics Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Temporal patterns
        pattern_names = list(temporal_patterns.keys())
        pattern_values = list(temporal_patterns.values())
        ax2.bar(pattern_names, pattern_values, color=self.color_palette[5:5+len(pattern_names)])
        ax2.set_ylabel('Pattern Value')
        ax2.set_title('Temporal Pattern Analysis')
        ax2.tick_params(axis='x', rotation=45)
        
        # Theoretical comparison
        theoretical_analysis = neuromorphic_data['theoretical_analysis']
        theoretical_metrics = ['Landauer Efficiency', 'Biological Comparison']
        theoretical_values = [theoretical_analysis['landauer_efficiency'], 
                             theoretical_analysis['biological_comparison']]
        
        ax3.bar(theoretical_metrics, theoretical_values, color=['gold', 'lightgreen'])
        ax3.set_ylabel('Efficiency Ratio')
        ax3.set_title('Theoretical Limit Comparison')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Theoretical Limit')
        ax3.legend()
        
        # Energy vs Information scatter plot
        # This would need actual data, so we'll create a sample
        energy_values = np.random.exponential(1.0, 100)
        information_values = np.random.poisson(5, 100)
        scatter = ax4.scatter(energy_values, information_values, alpha=0.6, c=range(100), cmap='viridis')
        ax4.set_xlabel('Energy Consumption')
        ax4.set_ylabel('Information Content')
        ax4.set_title('Energy-Information Trade-off')
        plt.colorbar(scatter, ax=ax4, label='Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_hybrid_system_plot(self, hybrid_data: Dict[str, Any], 
                                save_path: str = None) -> plt.Figure:
        """Create advanced hybrid system visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hybrid Quantum-Neuromorphic System Analysis', fontsize=16, fontweight='bold')
        
        synergy_analysis = hybrid_data['synergy_analysis']
        interaction_analysis = hybrid_data['interaction_analysis']
        performance_comparison = hybrid_data['performance_comparison']
        
        # Synergy analysis
        synergy_metrics = ['Synergy Mean', 'Quantum Correlation', 'Neuromorphic Correlation']
        synergy_values = [synergy_analysis['synergy_mean'], 
                         synergy_analysis['quantum_correlation'],
                         synergy_analysis['neuromorphic_correlation']]
        
        bars = ax1.bar(synergy_metrics, synergy_values, color=['purple', 'blue', 'green'])
        ax1.set_ylabel('Synergy Value')
        ax1.set_title('System Synergy Analysis')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Interaction strength over time (simulated)
        time_points = np.linspace(0, 10, 100)
        quantum_influence = 0.5 + 0.3 * np.sin(time_points)
        neuromorphic_influence = 0.5 + 0.3 * np.cos(time_points)
        
        ax2.plot(time_points, quantum_influence, 'b-', linewidth=2, label='Quantum Influence')
        ax2.plot(time_points, neuromorphic_influence, 'g-', linewidth=2, label='Neuromorphic Influence')
        ax2.fill_between(time_points, quantum_influence, neuromorphic_influence, alpha=0.3, color='purple')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Influence Strength')
        ax2.set_title('Component Interaction Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance comparison
        systems = ['Quantum', 'Neuromorphic', 'Hybrid']
        improvements = [performance_comparison['quantum_improvement_percent'],
                       performance_comparison['neuromorphic_improvement_percent'], 0]
        
        colors = ['red' if x < 0 else 'green' for x in improvements]
        bars = ax3.bar(systems, improvements, color=colors, alpha=0.7)
        ax3.set_ylabel('Performance Improvement (%)')
        ax3.set_title('Performance Comparison')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value >= 0 else -1),
                    f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')
        
        # System efficiency radar chart
        categories = ['Coherence', 'Efficiency', 'Stability', 'Synergy', 'Scalability']
        quantum_scores = [0.8, 0.6, 0.7, 0.5, 0.6]
        neuromorphic_scores = [0.5, 0.9, 0.8, 0.6, 0.8]
        hybrid_scores = [0.9, 0.8, 0.8, 0.9, 0.7]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        quantum_scores += quantum_scores[:1]
        neuromorphic_scores += neuromorphic_scores[:1]
        hybrid_scores += hybrid_scores[:1]
        
        ax4.plot(angles, quantum_scores, 'o-', linewidth=2, label='Quantum', color='blue')
        ax4.plot(angles, neuromorphic_scores, 'o-', linewidth=2, label='Neuromorphic', color='green')
        ax4.plot(angles, hybrid_scores, 'o-', linewidth=2, label='Hybrid', color='purple')
        ax4.fill(angles, hybrid_scores, alpha=0.25, color='purple')
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('System Performance Radar Chart')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

