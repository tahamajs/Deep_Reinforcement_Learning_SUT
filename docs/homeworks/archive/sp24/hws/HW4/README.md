# HW4: Practical Reinforcement Learning

## 📋 Overview

This capstone assignment focuses on practical RL applications, combining everything learned to build, tune, and deploy RL agents on real-world inspired tasks. You'll compare algorithms, tune hyperparameters, and tackle applied challenges.

## 📂 Contents

```
HW4/
├── SP24_RL_HW4/
│   ├── RL_HW4.pdf             # Assignment description
│   └── RL_HW4_Practical.ipynb # Implementation notebook
└── README.md                    # This file
```

## 🎯 Learning Objectives

- ✅ Apply RL to practical problems
- ✅ Compare multiple algorithms systematically
- ✅ Master hyperparameter tuning
- ✅ Handle real-world challenges (sparse rewards, partial observability)
- ✅ Build end-to-end RL solutions
- ✅ Evaluate and deploy trained agents

## 📚 Assignment Structure

### Part 1: Algorithm Comparison (30%)

**Task:** Compare DQN, PPO, and SAC on multiple environments

**Environments:**

1. **CartPole-v1** (easy, discrete)
2. **LunarLander-v2** (moderate, discrete/continuous)
3. **BipedalWalker-v3** (hard, continuous)

**Metrics to Compare:**

- Sample efficiency (steps to solve)
- Final performance (max reward)
- Training stability (variance across seeds)
- Computational cost (wall-clock time)
- Robustness (performance across environments)

**Implementation:**

```python
class AlgorithmComparison:
    def __init__(self, algorithms, environments):
        self.algorithms = algorithms  # [DQN, PPO, SAC]
        self.environments = environments
        self.results = defaultdict(dict)

    def run_comparison(self, num_seeds=5):
        for algo in self.algorithms:
            for env_name in self.environments:
                env = gym.make(env_name)

                # Train with multiple seeds
                results = []
                for seed in range(num_seeds):
                    set_seed(seed)
                    metrics = train_and_evaluate(algo, env, seed)
                    results.append(metrics)

                # Aggregate results
                self.results[algo][env_name] = {
                    'mean_reward': np.mean([r['final_reward'] for r in results]),
                    'std_reward': np.std([r['final_reward'] for r in results]),
                    'mean_steps': np.mean([r['steps_to_solve'] for r in results]),
                    'training_time': np.mean([r['time'] for r in results])
                }

    def plot_results(self):
        # Learning curves
        # Bar charts comparing metrics
        # Statistical significance tests
        pass
```

### Part 2: Hyperparameter Tuning (25%)

**Task:** Systematically tune a chosen algorithm on a challenging environment

**Hyperparameters to Tune:**

**For PPO:**

- Learning rate: [1e-4, 3e-4, 1e-3]
- Clip epsilon: [0.1, 0.2, 0.3]
- GAE lambda: [0.9, 0.95, 0.99]
- Number of epochs: [5, 10, 20]
- Batch size: [32, 64, 128]

**For DQN/DDQN:**

- Learning rate: [1e-4, 1e-3, 1e-2]
- Batch size: [32, 64, 128]
- Buffer size: [10k, 50k, 100k]
- Target update freq: [100, 1000, 5000]
- Network size: [64, 128, 256]

**Methodology:**

1. **Grid Search** (exhaustive, expensive)
2. **Random Search** (more efficient)
3. **Bayesian Optimization** (most efficient)

```python
from sklearn.model_selection import ParameterGrid

def hyperparameter_search(algorithm_class, env, param_grid):
    best_params = None
    best_reward = -float('inf')

    for params in ParameterGrid(param_grid):
        # Train with these parameters
        agent = algorithm_class(**params)
        rewards = train(agent, env, num_episodes=500)
        mean_reward = np.mean(rewards[-100:])

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params

    return best_params, best_reward
```

### Part 3: Practical Challenges (25%)

**Challenge 1: Sparse Rewards**

- Environment with delayed rewards
- Solutions: Reward shaping, curiosity, hindsight experience replay

**Challenge 2: Partial Observability**

- POMDP environment (hidden state information)
- Solutions: Recurrent policies (LSTM), frame stacking

**Challenge 3: Safety Constraints**

- Must avoid certain states/actions
- Solutions: Constrained RL, safe exploration

**Example - Reward Shaping:**

```python
class ShapedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Add shaped reward
        shaped_reward = reward + self._potential(next_state) - self._potential(self.state)

        self.state = next_state
        return next_state, shaped_reward, done, info

    def _potential(self, state):
        # Potential function (e.g., distance to goal)
        return -np.linalg.norm(state - self.goal)
```

### Part 4: Custom Application (20%)

**Task:** Build an RL solution for a real-world inspired problem

**Example Projects:**

1. **Portfolio Management**: Stock trading with RL
2. **Resource Allocation**: Cloud computing optimization
3. **Robotics Control**: Simulated robot navigation
4. **Game AI**: Custom game environment
5. **Recommendation Systems**: Sequential recommendation

**Requirements:**

- Define problem as MDP
- Justify algorithm choice
- Implement and train agent
- Evaluate performance
- Discuss limitations and future work

```python
class CustomEnvironment(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def step(self, action):
        # Apply action, compute reward, check termination
        next_state = self._compute_next_state(action)
        reward = self._compute_reward(action, next_state)
        done = self._check_termination(next_state)
        info = {}
        return next_state, reward, done, info

    def reset(self):
        # Reset to initial state
        return initial_state

    def render(self):
        # Visualization (optional)
        pass
```

## 🔧 Advanced Techniques

### 1. Transfer Learning

```python
def transfer_learning(source_env, target_env, pretrained_agent):
    # Fine-tune on target environment
    agent = copy.deepcopy(pretrained_agent)

    # Reduce learning rate for fine-tuning
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] *= 0.1

    # Train on target
    train(agent, target_env, num_episodes=200)
    return agent
```

### 2. Curriculum Learning

```python
def curriculum_learning(agent, environments):
    # Train on progressively harder tasks
    for env in sorted(environments, key=lambda e: e.difficulty):
        train(agent, env, num_episodes=500)
        if not is_solved(agent, env):
            print(f"Failed to solve {env.name}, stopping curriculum")
            break
    return agent
```

### 3. Ensemble Methods

```python
class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents

    def act(self, state):
        # Vote or average actions
        actions = [agent.act(state) for agent in self.agents]
        return self._aggregate(actions)

    def _aggregate(self, actions):
        # Majority vote for discrete, average for continuous
        if discrete:
            return Counter(actions).most_common(1)[0][0]
        else:
            return np.mean(actions, axis=0)
```

## 📊 Evaluation Metrics

### Performance Metrics

- **Mean Return**: Average cumulative reward
- **Success Rate**: % of episodes reaching goal
- **Sample Efficiency**: Steps to reach threshold
- **Wall-Clock Time**: Total training duration

### Robustness Metrics

- **Variance Across Seeds**: Stability measure
- **Worst-Case Performance**: Min reward over runs
- **Transfer Performance**: Performance on similar tasks

### Analysis Tools

```python
def comprehensive_evaluation(agent, env, num_episodes=100):
    rewards = []
    success_count = 0
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while not done:
            action = agent.act(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state

        rewards.append(episode_reward)
        episode_lengths.append(steps)
        if info.get('is_success', episode_reward > threshold):
            success_count += 1

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': success_count / num_episodes,
        'mean_length': np.mean(episode_lengths)
    }
```

## 💡 Best Practices

### Training

- ✅ Use multiple random seeds (5+)
- ✅ Save checkpoints regularly
- ✅ Monitor training metrics (loss, entropy, KL)
- ✅ Use TensorBoard for visualization
- ✅ Validate on held-out episodes

### Debugging

- ✅ Test on simple environments first
- ✅ Verify random agent baseline
- ✅ Check gradient magnitudes
- ✅ Visualize learned policies
- ✅ Compare with published baselines

### Deployment

- ✅ Save best model (not last)
- ✅ Document hyperparameters
- ✅ Create reproducible setup
- ✅ Provide usage examples
- ✅ Include performance benchmarks

## 📖 References

### Tools and Libraries

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [RLlib](https://docs.ray.io/en/latest/rllib/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Weights & Biases](https://wandb.ai/)

### Papers

- **Henderson et al. (2018)** - Deep RL Reproducibility
- **Engstrom et al. (2020)** - Implementation Matters
- **Plappert et al. (2018)** - Multi-Goal RL

## 🎯 Deliverables

1. **Code**: Clean, documented, reproducible
2. **Report**:
   - Algorithm comparison analysis
   - Hyperparameter tuning results
   - Challenge solutions
   - Custom application description
3. **Plots**: Learning curves, comparisons, ablations
4. **Video** (optional): Trained agent demonstration

## ⏱️ Time Estimate

- Algorithm Comparison: 8-10 hours
- Hyperparameter Tuning: 6-8 hours
- Practical Challenges: 6-8 hours
- Custom Application: 10-15 hours
- Report & Analysis: 5-7 hours
- **Total**: 35-48 hours

---

**Difficulty**: ⭐⭐⭐⭐⭐ (Advanced/Capstone)  
**Prerequisites**: All previous HWs  
**Key Skills**: End-to-end RL, practical deployment

This is your culminating project - showcase everything you've learned!
