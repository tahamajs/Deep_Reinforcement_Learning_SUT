"""
Network Architecture Comparison and Analysis
===========================================

This module provides tools for comparing and analyzing different
DQN network architectures, including parameter counts, memory usage,
and computational costs.

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class DQNArchitectureComparison:
    """Compare different DQN architectures and their properties"""
    
    def __init__(self):
        self.architectures = {}
        
    def create_dqn(self, state_size, action_size, hidden_sizes=[512, 256], dropout=0.1):
        """Create a fully connected DQN network"""
        
        class DQNNet(nn.Module):
            def __init__(self):
                super(DQNNet, self).__init__()
                self.state_size = state_size
                self.action_size = action_size
                
                layers = []
                input_size = state_size
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    input_size = hidden_size
                
                layers.append(nn.Linear(input_size, action_size))
                self.network = nn.Sequential(*layers)
                self.apply(self._init_weights)
            
            def _init_weights(self, layer):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0.01)
            
            def forward(self, state):
                return self.network(state)
        
        return DQNNet()
    
    def create_conv_dqn(self, action_size, input_channels=4):
        """Create a convolutional DQN network"""
        
        class ConvDQNNet(nn.Module):
            def __init__(self):
                super(ConvDQNNet, self).__init__()
                self.action_size = action_size
                
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
                
                conv_out_size = 64 * 7 * 7
                self.fc1 = nn.Linear(conv_out_size, 512)
                self.fc2 = nn.Linear(512, action_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return ConvDQNNet()
        
    def create_networks(self):
        """Create different network architectures for comparison"""
        self.architectures['Small_FC'] = self.create_dqn(4, 2, [64, 32])
        self.architectures['Medium_FC'] = self.create_dqn(100, 4, [512, 256, 128])
        self.architectures['Large_FC'] = self.create_dqn(1000, 10, [1024, 512, 256])
        self.architectures['Conv_Atari'] = self.create_conv_dqn(4, input_channels=4)
        
        return self.architectures
    
    def analyze_architectures(self):
        """Analyze network architectures and parameters"""
        networks = self.create_networks()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate parameter counts
        param_counts = {}
        for name, net in networks.items():
            param_count = sum(p.numel() for p in net.parameters())
            param_counts[name] = param_count
        
        names = list(param_counts.keys())
        counts = list(param_counts.values())
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange']
        
        # Plot 1: Parameter counts
        bars = axes[0,0].bar(names, counts, color=colors, alpha=0.8)
        axes[0,0].set_title('Parameter Count by Architecture', fontsize=12, fontweight='bold')
        axes[0,0].set_ylabel('Number of Parameters')
        axes[0,0].set_yscale('log')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height*1.1,
                          f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        # Calculate memory usage
        memory_usage = {}
        for name, net in networks.items():
            params = sum(p.numel() for p in net.parameters())
            if 'Conv' in name:
                activation_memory = 84*84*4 + 20*20*32 + 9*9*64 + 7*7*64 + 512
            else:
                activation_memory = sum([layer.out_features for layer in net.network 
                                        if isinstance(layer, nn.Linear)])
            
            memory_usage[name] = (params * 4 + activation_memory * 4) / 1024 / 1024  # MB
        
        # Plot 2: Memory usage
        names = list(memory_usage.keys())
        usage = list(memory_usage.values())
        
        axes[0,1].bar(names, usage, color=colors, alpha=0.8)
        axes[0,1].set_title('Estimated Memory Usage', fontsize=12, fontweight='bold')
        axes[0,1].set_ylabel('Memory (MB)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: FLOPs estimate
        flops_estimate = {}
        for name, count in param_counts.items():
            if 'Conv' in name:
                flops_estimate[name] = count * 10
            else:
                flops_estimate[name] = count * 2
        
        names = list(flops_estimate.keys())
        flops = list(flops_estimate.values())
        
        axes[1,0].bar(names, flops, color=colors, alpha=0.8)
        axes[1,0].set_title('Estimated FLOPs per Forward Pass', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('FLOPs')
        axes[1,0].set_yscale('log')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Application domains
        applications = {
            'Small_FC': ['CartPole', 'MountainCar', 'Simple Control'],
            'Medium_FC': ['LunarLander', 'Acrobot', 'Complex Control'],
            'Large_FC': ['High-dim State', 'Sensor Arrays', 'Complex Obs'],
            'Conv_Atari': ['Atari Games', 'Visual Tasks', 'Image-based RL']
        }
        
        axes[1,1].axis('off')
        text_content = "Suitable Applications:\\n\\n"
        for arch, apps in applications.items():
            text_content += f"{arch}:\\n"
            for app in apps:
                text_content += f"  • {app}\\n"
            text_content += "\\n"
        
        axes[1,1].text(0.05, 0.95, text_content, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1,1].set_title('Application Domains', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return param_counts, memory_usage, flops_estimate


def analyze_dqn_architectures():
    """Run comprehensive architecture analysis"""
    print("Analyzing Different DQN Architectures...")
    print("=" * 50)
    
    comparison = DQNArchitectureComparison()
    param_counts, memory_usage, flops_estimate = comparison.analyze_architectures()
    
    print("\\nArchitecture Analysis Summary:")
    print("=" * 50)
    for name in param_counts.keys():
        print(f"{name}:")
        print(f"  Parameters: {param_counts[name]:,}")
        print(f"  Memory: {memory_usage[name]:.2f} MB")
        print(f"  FLOPs: {flops_estimate[name]:,}")
        print()
    
    print("✓ DQN architectures implemented and analyzed")
    print("✓ Parameter efficiency and computational costs evaluated")
    
    return comparison, param_counts, memory_usage, flops_estimate


if __name__ == "__main__":
    analyze_dqn_architectures()
