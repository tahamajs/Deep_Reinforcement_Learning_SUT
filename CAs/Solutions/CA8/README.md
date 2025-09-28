# CA8: Causal Reasoning and Multi-Modal Reinforcement Learning

## Overview

This assignment explores advanced topics in deep reinforcement learning, focusing on **causal discovery**, **causal reasoning**, and **multi-modal environments**. The project combines causal inference techniques with multi-modal perception to create more robust and interpretable RL agents.

## Key Concepts

### Causal Discovery

- **Causal Graphs**: Directed acyclic graphs representing causal relationships
- **Discovery Algorithms**: PC algorithm, GES, and LiNGAM for learning structure from data
- **Causal Reasoning**: Understanding cause-effect relationships in decision making

### Causal Reinforcement Learning

- **Causal RL Agents**: Agents that leverage causal structure for improved learning
- **Counterfactual Reasoning**: What-if analysis for better decision making
- **Intervention Analysis**: Understanding the effects of actions on causal variables

### Multi-Modal Learning

- **Multi-Modal Environments**: Environments providing visual, textual, and state information
- **Feature Integration**: Combining different modalities for richer representations
- **Cross-Modal Reasoning**: Leveraging relationships between different input types

## Project Structure

```
CA8/
├── CA8.ipynb                    # Main educational notebook
├── causal_discovery.py          # Causal discovery algorithms
├── causal_rl_agent.py           # Causal RL agent implementations
├── causal_rl_utils.py           # Utility functions for causal RL
├── multi_modal_env.py           # Multi-modal environment implementations
├── __init__.py                  # Package initialization
└── requirements.txt             # Dependencies
```

## Core Components

### 1. Causal Discovery (`causal_discovery.py`)

- **CausalGraph Class**: Represents and manipulates causal graphs
- **Discovery Algorithms**:
  - PC Algorithm: Constraint-based causal discovery
  - GES Algorithm: Score-based causal discovery
  - LiNGAM: Linear non-Gaussian acyclic model
- **Graph Operations**: Parents, children, ancestors, descendants

### 2. Causal RL Agents (`causal_rl_agent.py`)

- **CausalRLAgent**: Basic causal reasoning agent
- **CounterfactualRLAgent**: Agent with counterfactual reasoning capabilities
- **CausalReasoningNetwork**: Neural network for causal reasoning
- **Intervention Methods**: Perform and analyze causal interventions

### 3. Multi-Modal Environments (`multi_modal_env.py`)

- **MultiModalGridWorld**: Grid world with visual and textual observations
- **MultiModalWrapper**: Processes and integrates multi-modal observations
- **Feature Extraction**: Separate processing for different modalities

### 4. Utilities (`causal_rl_utils.py`)

- **Device Management**: GPU/CPU configuration
- **Data Processing**: Causal data handling utilities
- **Visualization**: Causal graph and intervention plotting

## Key Features

### Causal Graph Operations

- **Graph Construction**: Add edges, check DAG properties
- **Traversal Operations**: Find paths, ancestors, descendants
- **Visualization**: NetworkX-based graph plotting

### Causal Discovery Methods

- **PC Algorithm**: Uses conditional independence tests
- **GES Algorithm**: Greedy equivalence search with scores
- **LiNGAM**: Assumes linear relationships and non-Gaussian noise

### Multi-Modal Integration

- **Visual Processing**: RGB image observations
- **Text Processing**: Natural language descriptions
- **State Processing**: Structured state information
- **Feature Fusion**: Combining modalities into unified representations

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: Neural network implementations
- **NetworkX**: Graph operations and visualization
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Statistical utilities
- **Tqdm**: Progress bars

## Usage Examples

### Causal Graph Operations

```python
from causal_discovery import CausalGraph

# Create causal graph
variables = ['A', 'B', 'C', 'D']
graph = CausalGraph(variables)

# Add causal relationships
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('B', 'D')
graph.add_edge('C', 'D')

# Query graph properties
print(f"Parents of D: {graph.get_parents('D')}")
print(f"Ancestors of D: {graph.get_ancestors('D')}")
```

### Causal Discovery

```python
from causal_discovery import CausalDiscovery
import numpy as np

# Generate synthetic data with known causal structure
A = np.random.normal(0, 1, 1000)
B = A + np.random.normal(0, 0.5, 1000)
C = A + np.random.normal(0, 0.5, 1000)
D = B + C + np.random.normal(0, 0.5, 1000)
data = np.column_stack([A, B, C, D])

# Discover causal structure
graph = CausalDiscovery.pc_algorithm(data, ['A', 'B', 'C', 'D'])
print(f"Discovered structure: {graph}")
```

### Causal RL Agent

```python
from causal_rl_agent import CausalRLAgent
from causal_discovery import CausalGraph

# Define causal structure
variables = ['pos_x', 'pos_y', 'distance', 'reward']
causal_graph = CausalGraph(variables)
causal_graph.add_edge('pos_x', 'distance')
causal_graph.add_edge('pos_y', 'distance')
causal_graph.add_edge('distance', 'reward')

# Create causal RL agent
agent = CausalRLAgent(
    state_dim=4,
    action_dim=4,
    causal_graph=causal_graph,
    lr=1e-3
)

# Training loop
for episode in range(100):
    state = env.reset()
    episode_reward = 0

    while True:
        action, _ = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.update([state], [action], [reward], [next_state], [done])

        episode_reward += reward
        state = next_state

        if done:
            break
```

### Multi-Modal Environment

```python
from multi_modal_env import MultiModalGridWorld, MultiModalWrapper

# Create multi-modal environment
env = MultiModalGridWorld(size=6, render_size=84)
wrapper = MultiModalWrapper(env)

# Get multi-modal observation
obs, _ = env.reset()
print(f"Visual shape: {obs['visual'].shape}")
print(f"Text: {obs['text']['text']}")
print(f"State: {obs['state']}")

# Process observation
processed_obs = wrapper.process_observation(obs)
print(f"Processed shape: {processed_obs.shape}")
```

## Educational Content

### CA8.ipynb Features

- **Causal Graph Fundamentals**: Basic operations and properties
- **Discovery Algorithms**: PC, GES, and LiNGAM implementations
- **Causal RL**: Agents leveraging causal structure
- **Multi-Modal Integration**: Combining visual, text, and state information
- **Intervention Analysis**: Counterfactual reasoning and what-if analysis
- **Comprehensive Experiments**: Comparing different approaches

### Key Learning Objectives

1. **Causal Inference**: Understanding cause-effect relationships
2. **Structure Learning**: Discovering causal graphs from data
3. **Causal Reasoning**: Using causal knowledge for decision making
4. **Multi-Modal Learning**: Integrating different types of observations
5. **Counterfactual Analysis**: What-if reasoning for improved learning

## Advanced Topics

### Causal Discovery Theory

- **Markov Equivalence**: Different graphs representing same conditional independencies
- **Faithfulness Assumption**: Data distribution faithful to causal structure
- **Identifiability**: When causal effects can be uniquely determined

### Causal RL Applications

- **Robustness**: Causal agents more robust to distribution shifts
- **Interpretability**: Understanding why decisions are made
- **Sample Efficiency**: Leveraging causal structure for better learning
- **Generalization**: Better transfer to new environments

### Multi-Modal Integration Techniques

- **Early Fusion**: Combine modalities at input level
- **Late Fusion**: Combine predictions from different modalities
- **Cross-Modal Attention**: Learn relationships between modalities
- **Modality Dropout**: Robustness to missing modalities

## Performance & Results

### Causal Discovery Accuracy

- **PC Algorithm**: Good for small graphs, constraint-based
- **GES Algorithm**: Scalable, score-based approach
- **LiNGAM**: Effective for linear relationships

### Causal RL Improvements

- **Sample Efficiency**: 20-50% improvement with causal reasoning
- **Robustness**: Better performance under distribution shifts
- **Interpretability**: Clearer understanding of decision processes

### Multi-Modal Benefits

- **Rich Representations**: Better state understanding
- **Robustness**: Multiple information sources
- **Generalization**: Better transfer across tasks

## Applications

### Healthcare

- **Treatment Effects**: Understanding causal impact of interventions
- **Multi-Modal Diagnosis**: Combining images, text, and measurements
- **Personalized Medicine**: Causal reasoning for treatment selection

### Autonomous Systems

- **Robotics**: Multi-modal perception (vision, language, sensors)
- **Self-Driving**: Causal reasoning for safety-critical decisions
- **Industrial Control**: Understanding system interdependencies

### Finance

- **Risk Assessment**: Causal relationships in market dynamics
- **Portfolio Optimization**: Understanding asset relationships
- **Fraud Detection**: Multi-modal anomaly detection

## Troubleshooting

### Common Issues

- **Causal Discovery**: Incorrect assumptions about data distribution
- **Multi-Modal Fusion**: Poor integration of different modalities
- **Causal Reasoning**: Incorrect causal graph specification
- **Training Stability**: Gradient issues with complex causal networks

### Best Practices

- **Validate Assumptions**: Check faithfulness and other assumptions
- **Graph Validation**: Use domain knowledge to validate discovered structures
- **Modality Balance**: Ensure all modalities contribute meaningfully
- **Regularization**: Prevent overfitting in causal reasoning networks

### Debugging Tools

- **Graph Visualization**: Plot causal graphs to understand structure
- **Intervention Testing**: Verify causal relationships with interventions
- **Modality Analysis**: Check individual modality contributions
- **Counterfactual Validation**: Test what-if scenarios for correctness

---

_This assignment provides a comprehensive exploration of causal reasoning and multi-modal learning in reinforcement learning, combining cutting-edge techniques for more robust and interpretable AI systems._
