# HW10: Multi-Agent Reinforcement Learning

[![Deep RL](https://img.shields.io/badge/Deep-RL-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Multi-Agent](https://img.shields.io/badge/Type-Multi--Agent-green.svg)](.)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](.)

## 📋 Overview

This assignment explores multi-agent reinforcement learning (MARL), where multiple agents learn and interact simultaneously. MARL introduces unique challenges including non-stationarity, credit assignment, communication, and emergent behaviors.

## 🎯 Learning Objectives

1. **Multi-Agent Fundamentals**: Understand game theory and Nash equilibria
2. **Cooperative MARL**: Learn algorithms for teamwork and coordination
3. **Competitive MARL**: Master self-play and opponent modeling
4. **Mixed Settings**: Handle both cooperation and competition
5. **Communication**: Learn emergent and explicit communication protocols
6. **Scalability**: Address challenges in large-scale multi-agent systems

## 📚 Core Concepts

### 1. Multi-Agent Settings

**Classifications:**

#### Fully Cooperative
- Agents share same reward
- Team vs environment
- Example: Robot swarm, multiplayer cooperative games

#### Fully Competitive  
- Zero-sum games
- One agent's gain is another's loss
- Example: Chess, Go, poker

#### Mixed (General-Sum)
- Mix of cooperation and competition
- Complex social dynamics
- Example: Trading, negotiation, social dilemmas

**Key Challenges:**

1. **Non-Stationarity**
```
From agent i's perspective:
Environment changes as other agents learn
→ "Moving target" problem
```

2. **Credit Assignment**
```
Team gets reward, but which agent deserves credit?
→ Multiagent credit assignment problem
```

3. **Scalability**
```
Joint action space grows exponentially: |A|^n
Joint state space also explodes
```

4. **Partial Observability**
```
Each agent has limited view
Must infer others' states and intentions
```

### 2. Game Theory Foundations

**Normal Form Games:**
```
      Agent 2
      C    D
A  C  3,3  0,5
g  D  5,0  1,1
e
n     (Prisoner's Dilemma)
t
1
```

**Key Concepts:**

#### Nash Equilibrium
```
No agent can improve by unilaterally changing strategy
π* is Nash if: Vi(π*) ≥ Vi(πi', π*_{-i}) for all i, πi'
```

#### Pareto Optimality
```
No allocation where everyone can be better off
```

#### Dominated Strategies
```
Strategy that is always worse than another
```

**Solution Concepts:**
- Pure Nash Equilibrium
- Mixed Nash Equilibrium  
- Correlated Equilibrium
- Stackelberg Equilibrium

### 3. Independent Learning

**Simplest Approach:** Each agent treats others as part of environment

```python
class IndependentQLearning:
    def __init__(self, num_agents, state_dim, action_dim):
        self.agents = [
            QLearning(state_dim, action_dim) 
            for _ in range(num_agents)
        ]
    
    def update(self, obs, actions, rewards, next_obs):
        for i, agent in enumerate(self.agents):
            agent.update(
                obs[i], 
                actions[i], 
                rewards[i], 
                next_obs[i]
            )
```

**Advantages:**
- Simple, scalable
- No communication needed
- Works with existing single-agent algorithms

**Disadvantages:**
- Ignores non-stationarity
- No convergence guarantees
- Suboptimal in cooperative settings

### 4. Centralized Training, Decentralized Execution (CTDE)

**Key Idea:** Use global information during training, local at execution

#### QMIX
```python
class QMIX(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim):
        super().__init__()
        
        # Individual Q-networks
        self.agent_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ) for _ in range(num_agents)
        ])
        
        # Mixing network (monotonic)
        self.hyper_w1 = nn.Linear(state_dim, num_agents * 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, obs, state):
        # Individual Q-values
        q_vals = [net(obs[i]) for i, net in enumerate(self.agent_networks)]
        q_vals = torch.stack(q_vals, dim=1)  # [batch, n_agents, n_actions]
        
        # Mix Q-values (ensure monotonicity with abs)
        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)
        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)
        
        # Mixing
        hidden = F.elu(torch.matmul(q_vals, w1) + b1)
        q_tot = torch.matmul(hidden, w2) + b2
        
        return q_tot, q_vals
```

**Key Property:** Monotonicity
```
∂Q_tot/∂Q_i ≥ 0 for all i

Ensures: argmax Q_i = argmax Q_tot
→ Decentralized execution optimal!
```

#### MADDPG (Multi-Agent DDPG)
```python
class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim):
        self.actors = [Actor(obs_dim, action_dim) 
                      for _ in range(num_agents)]
        
        # Centralized critics see all observations and actions
        self.critics = [Critic(num_agents * (obs_dim + action_dim)) 
                       for _ in range(num_agents)]
    
    def act(self, obs):
        # Decentralized: each agent uses only local obs
        actions = [actor(obs[i]) for i, actor in enumerate(self.actors)]
        return actions
    
    def update(self, obs, actions, rewards, next_obs):
        # Centralized critic update
        all_obs = torch.cat(obs, dim=-1)
        all_actions = torch.cat(actions, dim=-1)
        
        for i in range(self.num_agents):
            # Critic update
            Q = self.critics[i](all_obs, all_actions)
            Q_target = compute_target(rewards[i], next_obs)
            critic_loss = F.mse_loss(Q, Q_target)
            
            # Actor update (policy gradient)
            actor_loss = -self.critics[i](
                all_obs, 
                self.replace_action(actions, i, self.actors[i](obs[i]))
            ).mean()
```

**Advantages:**
- Addresses non-stationarity (centralized training)
- Decentralized execution (scalable)
- Proven effective empirically

### 5. Communication

**Emergent Communication:**
```python
class CommNet(nn.Module):
    def __init__(self, obs_dim, action_dim, comm_dim):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, comm_dim)
        self.comm = nn.GRU(comm_dim, comm_dim)
        self.decoder = nn.Linear(comm_dim, action_dim)
    
    def forward(self, observations):
        # Encode observations
        hidden = [self.encoder(obs) for obs in observations]
        
        # Communication rounds
        for _ in range(num_comm_rounds):
            # Average communication
            comm_input = torch.stack(hidden).mean(dim=0)
            hidden, _ = self.comm(comm_input, hidden)
        
        # Decode to actions
        actions = [self.decoder(h) for h in hidden]
        return actions
```

**Targeted Communication (TarMAC):**
```python
class TarMAC:
    def forward(self, obs):
        # Generate query, key, value for each agent
        queries = [self.query_net(obs[i]) for i in range(n)]
        keys = [self.key_net(obs[i]) for i in range(n)]
        values = [self.value_net(obs[i]) for i in range(n)]
        
        # Attention-based communication
        for i in range(n):
            # Compute attention weights
            scores = [torch.dot(queries[i], keys[j]) 
                     for j in range(n) if j != i]
            weights = F.softmax(scores)
            
            # Aggregate messages
            message = sum(w * values[j] 
                         for w, j in zip(weights, range(n)))
            
            # Update agent representation
            hidden[i] = self.update(obs[i], message)
```

### 6. Population-Based Training

**Self-Play:**
```python
class SelfPlay:
    def __init__(self, policy):
        self.policy = policy
        self.opponent = copy.deepcopy(policy)
    
    def train_step(self):
        # Play against past self
        trajectory = play_episode(self.policy, self.opponent)
        
        # Update policy
        self.policy.update(trajectory)
        
        # Periodically update opponent
        if step % update_freq == 0:
            self.opponent = copy.deepcopy(self.policy)
```

**Fictitious Self-Play:**
- Maintain population of past policies
- Sample opponents from population
- Prevents cycling, maintains diversity

**League Training (AlphaStar):**
- Main agents
- Main exploiters (find weaknesses)
- League exploiters (general strategies)

### 7. Emergent Behaviors

**Examples:**
- **Prey-predator**: Coordinated hunting, evasion
- **Communication**: Emergent languages
- **Tool use**: Using environment creatively
- **Social learning**: Imitation, teaching

**Facilitating Emergence:**
- Appropriate reward structure
- Sufficient environmental complexity
- Population diversity
- Long training times

## 📊 Topics Covered

1. **Game Theory**: Nash equilibria, solution concepts
2. **Independent Learning**: IQL, independent PPO
3. **CTDE**: QMIX, MADDPG, COMA
4. **Communication**: CommNet, TarMAC, RIAL
5. **Self-Play**: Fictitious play, league training
6. **Applications**: Robotics, games, traffic, economics

## 📖 Key References

1. **Lowe, R., et al. (2017)** - "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" - NIPS (MADDPG)

2. **Rashid, T., et al. (2018)** - "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL" - ICML

3. **Foerster, J., et al. (2016)** - "Learning to Communicate with Deep Multi-Agent RL" - NIPS (CommNet)

4. **Vinyals, O., et al. (2019)** - "Grandmaster level in StarCraft II using multi-agent RL" - Nature (AlphaStar)

5. **Littman, M. L. (1994)** - "Markov games as a framework for multi-agent RL" - ICML

## 💡 Discussion Questions

1. Why is non-stationarity a fundamental challenge in MARL?
2. How does QMIX ensure decentralized execution is optimal?
3. What are trade-offs between centralized and decentralized approaches?
4. When might emergent communication be preferred over designed protocols?
5. How does self-play lead to increasing sophistication?

## 🎓 Extensions

- Implement multi-agent particle environments
- Try hierarchical multi-agent RL
- Explore opponent modeling
- Study social dilemmas (tragedy of commons)
- Implement graph neural networks for communication

---

**Course:** Deep Reinforcement Learning  
**Last Updated:** 2024
