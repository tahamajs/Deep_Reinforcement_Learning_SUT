import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from world_models.world_models import RSSMCore, WorldModel, MPCPlanner
from multi_agent_rl.multi_agent_rl import MADDPGAgent, MultiAgentEnvironment
from causal_rl.causal_rl import CausalGraph, CausalDiscovery, CausalWorldModel
from quantum_rl.quantum_rl import QuantumState, QuantumCircuit, QuantumQLearning
from federated_rl.federated_rl import FederatedAgent, FederatedServer
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_integrated_environment():
    """Create an environment that combines multiple paradigms"""

    class IntegratedEnvironment:
        def __init__(self):
            # Multi-agent setup
            self.n_agents = 2
            self.obs_dim = 6  # Shared state + private observations
            self.action_dim = 2

            # Causal structure
            self.causal_graph = CausalGraph(['x1', 'x2', 'y1', 'y2', 'z'])

            # Quantum state representation
            self.quantum_state = QuantumState(n_qubits=3)

            self.max_steps = 100
            self.reset()

        def reset(self):
            # Initialize with causal dependencies
            self.shared_state = np.random.normal(0, 1, 2)
            self.agent_states = np.random.normal(0, 1, (self.n_agents, 2))

            # Update causal graph
            self.causal_graph.update_state({
                'x1': self.shared_state[0],
                'x2': self.shared_state[1],
                'y1': self.agent_states[0, 0],
                'y2': self.agent_states[1, 0],
                'z': np.mean(self.agent_states[:, 1])
            })

            self.steps = 0
            return self.get_global_observation()

        def get_global_observation(self):
            """Get observation for all agents"""
            obs = []
            for i in range(self.n_agents):
                agent_obs = np.concatenate([
                    self.shared_state,
                    self.agent_states[i],
                    [self.causal_graph.get_effect('z', f'y{i+1}')]
                ])
                obs.append(agent_obs)
            return np.array(obs)

        def step(self, actions):
            actions = np.array(actions)

            # Apply actions with causal dependencies
            for i in range(self.n_agents):
                action_effect = actions[i] * 0.1

                # Causal influence on shared state
                self.shared_state += action_effect * 0.05

                # Agent-specific dynamics
                self.agent_states[i] += action_effect + np.random.normal(0, 0.1, 2)

            # Update causal graph
            self.causal_graph.update_state({
                'x1': self.shared_state[0],
                'x2': self.shared_state[1],
                'y1': self.agent_states[0, 0],
                'y2': self.agent_states[1, 0],
                'z': np.mean(self.agent_states[:, 1])
            })

            # Compute rewards (cooperative objective)
            coordination_reward = -np.linalg.norm(self.shared_state)
            individual_rewards = [-np.linalg.norm(actions[i]) * 0.1 for i in range(self.n_agents)]

            rewards = np.array([coordination_reward + individual_rewards[i]
                              for i in range(self.n_agents)])

            self.steps += 1
            done = self.steps >= self.max_steps

            return self.get_global_observation(), rewards, done, {}

    return IntegratedEnvironment()

def demonstrate_paradigm_integration():
    """Demonstrate integration of multiple RL paradigms"""

    print("🔗 Demonstrating Paradigm Integration")
    print("="*50)

    # Create integrated environment
    env = create_integrated_environment()
    print(f"Integrated Environment: {env.n_agents} agents, {env.obs_dim}D obs, {env.action_dim}D actions")

    # Initialize agents with different paradigms
    agents = []

    # Agent 1: Multi-agent with causal reasoning
    agent1 = MADDPGAgent(
        agent_idx=0,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        n_agents=env.n_agents,
        use_attention=True,
        use_communication=True
    )
    agents.append(('MADDPG+Causal', agent1))

    # Agent 2: Quantum-enhanced agent
    agent2 = QuantumQLearning(
        n_qubits=2,
        action_dim=env.action_dim,
        learning_rate=0.1,
        discount_factor=0.95
    )
    agents.append(('Quantum Q-Learning', agent2))

    # Training loop
    n_episodes = 50
    results = {name: [] for name, _ in agents}

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            actions = []
            for i, (name, agent) in enumerate(agents):
                if name == 'MADDPG+Causal':
                    # Multi-agent action selection
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(device)
                    action_tensor, _ = agent.act(obs_tensor, explore=True)
                    action = action_tensor.cpu().numpy()[0]
                else:  # Quantum agent
                    # Convert observation to quantum state
                    quantum_obs = obs[i][:2]  # Use first 2 dimensions
                    action = agent.select_action(quantum_obs)

                actions.append(action)

            next_obs, rewards, done, _ = env.step(actions)

            # Update agents
            for i, (name, agent) in enumerate(agents):
                if name == 'MADDPG+Causal':
                    # Store experience for MADDPG
                    pass  # Would need full replay buffer implementation
                else:  # Quantum agent
                    agent.update(obs[i][:2], actions[i], rewards[i], next_obs[i][:2], done)

            episode_reward += np.mean(rewards)
            obs = next_obs

        # Record results
        for i, (name, _) in enumerate(agents):
            results[name].append(episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: Agent1={results['MADDPG+Causal'][-1]:.2f}, Agent2={results['Quantum Q-Learning'][-1]:.2f}")

    return results

def demonstrate_federated_learning():
    """Demonstrate federated RL across distributed agents"""

    print("\n🌐 Demonstrating Federated Reinforcement Learning")
    print("="*50)

    # Create federated setup
    n_clients = 3
    clients = []

    for i in range(n_clients):
        client = FederatedAgent(
            client_id=i,
            obs_dim=4,
            action_dim=2,
            local_epochs=5
        )
        clients.append(client)

    server = FederatedServer(
        global_model_dim=100,  # Simplified
        n_clients=n_clients
    )

    # Simulate federated training
    n_rounds = 10
    global_rewards = []

    for round_num in range(n_rounds):
        print(f"\nFederated Round {round_num + 1}")

        # Client local training
        client_updates = []
        client_rewards = []

        for client in clients:
            # Simulate local environment (different for each client)
            local_env = create_integrated_environment()
            local_env.n_agents = 1  # Single agent per client

            # Local training
            local_reward = client.train_local(local_env, episodes=20)
            client_rewards.append(local_reward)

            # Generate update
            update = client.generate_update()
            client_updates.append(update)

        # Server aggregation
        global_model = server.aggregate_updates(client_updates)

        # Distribute to clients
        for client in clients:
            client.receive_global_model(global_model)

        avg_reward = np.mean(client_rewards)
        global_rewards.append(avg_reward)

        print(f"Average client reward: {avg_reward:.3f}")

    return global_rewards

def create_hybrid_agent():
    """Create a hybrid agent combining multiple paradigms"""

    class HybridAgent:
        def __init__(self, obs_dim, action_dim):
            self.obs_dim = obs_dim
            self.action_dim = action_dim

            # Components from different paradigms
            self.world_model = WorldModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                state_dim=16,
                hidden_dim=32,
                embed_dim=64
            )

            self.causal_reasoner = CausalDiscovery(
                variables=[f'obs_{i}' for i in range(obs_dim)] +
                         [f'action_{i}' for i in range(action_dim)],
                alpha=0.1
            )

            self.quantum_processor = QuantumCircuit(n_qubits=2)

            # Classical policy network
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim + 16, 64),  # obs + world model state
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            ).to(device)

        def select_action(self, obs, explore=True):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # Get world model prediction
            with torch.no_grad():
                wm_output = self.world_model.observe_sequence(obs_tensor,
                    torch.zeros(1, self.action_dim).to(device))
                wm_state = wm_output['states'][:, -1]  # Last state

            # Combine with observation
            combined_input = torch.cat([obs_tensor, wm_state], dim=-1)

            # Get action
            with torch.no_grad():
                action = self.policy_net(combined_input).cpu().numpy()[0]

            if explore:
                action += np.random.normal(0, 0.1, self.action_dim)
                action = np.clip(action, -1, 1)

            return action

        def update_causal_model(self, experience_batch):
            """Update causal understanding from experience"""
            # Extract causal relationships from recent experiences
            data = np.array([exp['obs'] + exp['action'] for exp in experience_batch])
            self.causal_reasoner.discover_structure(data)

    return HybridAgent(obs_dim=6, action_dim=2)