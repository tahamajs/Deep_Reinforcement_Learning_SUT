# Meta-Learning and Adaptation Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from game_theory import MultiAgentEnvironment


class MAMLAgent(nn.Module):
    """Model-Agnostic Meta-Learning agent."""

    def __init__(
        self, obs_dim, action_dim, hidden_dim=64, inner_lr=0.01, meta_lr=0.001
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value network
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=meta_lr)

    def inner_update(self, obs, actions, rewards, next_obs, dones):
        """Perform inner loop update for adaptation."""
        # Compute policy loss
        action_logits = self.policy(obs)
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)

        # Compute value loss
        values = self.value(obs).squeeze()
        next_values = self.value(next_obs).squeeze()
        targets = rewards + 0.99 * next_values * (1 - dones)
        value_loss = F.mse_loss(values, targets.detach())

        # Compute policy loss (REINFORCE)
        advantages = targets - values.detach()
        policy_loss = -(log_probs * advantages).mean()

        total_loss = policy_loss + 0.5 * value_loss

        # Get current parameters
        old_params = {name: param.clone() for name, param in self.named_parameters()}

        # Perform gradient descent
        grads = torch.autograd.grad(total_loss, self.parameters(), create_graph=True)
        for param, grad in zip(self.parameters(), grads):
            param.data = param.data - self.inner_lr * grad

        return old_params

    def meta_update(self, task_losses):
        """Perform meta-update across tasks."""
        meta_loss = torch.stack(task_losses).mean()

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

    def adapt_to_task(self, task_data, n_inner_steps=5):
        """Adapt to a new task using MAML."""
        obs, actions, rewards, next_obs, dones = task_data

        # Store original parameters
        original_params = {
            name: param.clone() for name, param in self.named_parameters()
        }

        # Perform inner updates
        for _ in range(n_inner_steps):
            self.inner_update(obs, actions, rewards, next_obs, dones)

        # Compute adapted loss
        adapted_logits = self.policy(obs)
        adapted_dist = Categorical(logits=adapted_logits)
        adapted_log_probs = adapted_dist.log_prob(actions)

        adapted_values = self.value(obs).squeeze()
        next_adapted_values = self.value(next_obs).squeeze()
        adapted_targets = rewards + 0.99 * next_adapted_values * (1 - dones)
        adapted_loss = -(
            adapted_log_probs * (adapted_targets - adapted_values.detach())
        ).mean()

        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name]

        return adapted_loss


class OpponentModel(nn.Module):
    """Model of opponent behavior for strategic adaptation."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Opponent policy model
        self.opponent_policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Opponent value model
        self.opponent_value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Belief update network
        self.belief_update = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim),  # obs + action + reward
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # belief confidence
        )

    def predict_opponent_action(self, obs):
        """Predict opponent's next action."""
        action_logits = self.opponent_policy(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

    def update_belief(self, obs, opponent_action, reward):
        """Update belief about opponent based on observed behavior."""
        # Combine observation, action, and reward
        belief_input = torch.cat(
            [
                obs,
                F.one_hot(opponent_action, self.action_dim).float(),
                reward.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Update belief confidence
        belief_confidence = self.belief_update(belief_input)

        return belief_confidence

    def strategic_adaptation(self, obs, opponent_belief):
        """Adapt strategy based on opponent model."""
        opponent_probs = self.predict_opponent_action(obs)

        # Compute best response considering opponent model
        # This is a simplified strategic adaptation
        expected_opponent_action = torch.argmax(opponent_probs)

        # Adjust own strategy based on opponent belief
        strategic_adjustment = opponent_belief * 0.1  # Small adjustment factor

        return expected_opponent_action, strategic_adjustment


class PopulationBasedTraining:
    """Population-Based Training for hyperparameter optimization."""

    def __init__(self, population_size=10, mutation_rate=0.1, mutation_strength=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = []
        self.fitness_scores = []

    def initialize_population(self, agent_class, agent_kwargs):
        """Initialize population of agents."""
        self.population = []
        for _ in range(self.population_size):
            # Randomly initialize hyperparameters
            lr = 10 ** np.random.uniform(-4, -2)  # Learning rate
            gamma = np.random.uniform(0.9, 0.999)  # Discount factor
            hidden_dim = np.random.choice([32, 64, 128, 256])  # Hidden dimension

            agent = agent_class(
                **agent_kwargs, lr=lr, gamma=gamma, hidden_dim=hidden_dim
            )
            self.population.append(
                {
                    "agent": agent,
                    "hyperparams": {"lr": lr, "gamma": gamma, "hidden_dim": hidden_dim},
                    "fitness": 0.0,
                }
            )

    def mutate_hyperparams(self, hyperparams):
        """Mutate hyperparameters."""
        mutated = hyperparams.copy()

        for key, value in mutated.items():
            if np.random.random() < self.mutation_rate:
                if key == "lr":
                    # Log-space mutation for learning rate
                    mutated[key] = value * (
                        1 + np.random.normal(0, self.mutation_strength)
                    )
                    mutated[key] = np.clip(mutated[key], 1e-5, 1e-1)
                elif key == "gamma":
                    # Clip mutation for gamma
                    mutated[key] = value + np.random.normal(0, self.mutation_strength)
                    mutated[key] = np.clip(mutated[key], 0.8, 0.999)
                elif key == "hidden_dim":
                    # Discrete mutation for hidden dimension
                    options = [32, 64, 128, 256]
                    current_idx = options.index(value) if value in options else 1
                    new_idx = np.clip(
                        current_idx + np.random.choice([-1, 0, 1]), 0, len(options) - 1
                    )
                    mutated[key] = options[new_idx]

        return mutated

    def evolve_population(self, fitness_scores):
        """Evolve population based on fitness."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order

        # Keep top performers
        elite_size = max(1, self.population_size // 5)
        elites = [self.population[i] for i in sorted_indices[:elite_size]]

        # Create new population
        new_population = elites.copy()

        while len(new_population) < self.population_size:
            # Select parent (tournament selection)
            parent_idx = np.random.choice(sorted_indices[: len(sorted_indices) // 2])
            parent = self.population[parent_idx]

            # Mutate hyperparameters
            mutated_hyperparams = self.mutate_hyperparams(parent["hyperparams"])

            # Create new agent with mutated hyperparameters
            new_agent = type(parent["agent"])(
                **{
                    k: v
                    for k, v in parent["agent"].__dict__.items()
                    if k not in ["lr", "gamma", "hidden_dim"]
                },
                **mutated_hyperparams,
            )

            new_population.append(
                {"agent": new_agent, "hyperparams": mutated_hyperparams, "fitness": 0.0}
            )

        self.population = new_population

    def get_best_agent(self):
        """Get the best performing agent."""
        if not self.population:
            return None

        best_agent = max(self.population, key=lambda x: x["fitness"])
        return best_agent["agent"], best_agent["hyperparams"]


class SelfPlayTraining:
    """Self-play training framework."""

    def __init__(self, agent_class, agent_kwargs, n_opponents=5):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.n_opponents = n_opponents

        # Initialize main agent
        self.main_agent = agent_class(**agent_kwargs)

        # Initialize opponent pool
        self.opponent_pool = []
        for _ in range(n_opponents):
            opponent = agent_class(**agent_kwargs)
            self.opponent_pool.append(opponent)

        # Training history
        self.training_history = []

    def update_opponent_pool(self, new_agent, performance_threshold=0.6):
        """Update opponent pool with improved agents."""
        # Evaluate new agent against current opponents
        total_score = 0
        for opponent in self.opponent_pool:
            score = self.evaluate_agents(new_agent, opponent)
            total_score += score

        avg_score = total_score / len(self.opponent_pool)

        if avg_score > performance_threshold:
            # Replace worst opponent
            worst_opponent_idx = np.argmin([opp.fitness for opp in self.opponent_pool])
            self.opponent_pool[worst_opponent_idx] = new_agent
            print(f"Updated opponent pool with new agent (avg score: {avg_score:.3f})")

    def evaluate_agents(self, agent1, agent2, n_episodes=10):
        """Evaluate two agents against each other."""
        total_score = 0

        for _ in range(n_episodes):
            # Create environment (simplified)
            env = MultiAgentEnvironment(n_agents=2, obs_dim=10, action_dim=4)

            obs = env.reset()
            done = False
            episode_score = 0

            while not done:
                # Get actions
                action1 = agent1.select_action(obs[0])
                action2 = agent2.select_action(obs[1])

                # Step environment
                next_obs, rewards, done, _ = env.step([action1, action2])

                episode_score += rewards[0]  # Score for agent1
                obs = next_obs

            total_score += episode_score

        return total_score / n_episodes

    def train_self_play(self, n_iterations=100, n_episodes_per_iter=20):
        """Train using self-play."""
        for iteration in range(n_iterations):
            iteration_scores = []

            for episode in range(n_episodes_per_iter):
                # Select random opponent
                opponent = np.random.choice(self.opponent_pool)

                # Train main agent against opponent
                episode_data = self.collect_episode_data(self.main_agent, opponent)
                self.main_agent.update(episode_data)

                iteration_scores.append(episode_data["total_reward"])

            avg_score = np.mean(iteration_scores)

            # Update opponent pool periodically
            if iteration % 10 == 0:
                self.update_opponent_pool(self.main_agent)

            self.training_history.append(
                {
                    "iteration": iteration,
                    "avg_score": avg_score,
                    "opponent_pool_size": len(self.opponent_pool),
                }
            )

            print(f"Iteration {iteration}: Avg Score = {avg_score:.3f}")

    def collect_episode_data(self, agent, opponent):
        """Collect episode data for training."""
        env = MultiAgentEnvironment(n_agents=2, obs_dim=10, action_dim=4)

        obs = env.reset()
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "total_reward": 0,
        }

        done = False
        while not done:
            # Get actions
            action_agent = agent.select_action(obs[0])
            action_opponent = opponent.select_action(obs[1])

            # Step environment
            next_obs, rewards, done, _ = env.step([action_agent, action_opponent])

            # Store data
            episode_data["observations"].append(obs[0])
            episode_data["actions"].append(action_agent)
            episode_data["rewards"].append(rewards[0])
            episode_data["next_observations"].append(next_obs[0])
            episode_data["dones"].append(done)
            episode_data["total_reward"] += rewards[0]

            obs = next_obs

        # Convert to tensors
        for key in episode_data:
            if key != "total_reward":
                episode_data[key] = torch.stack(episode_data[key])

        return episode_data


class CurriculumLearning:
    """Curriculum learning for gradual difficulty increase."""

    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0, difficulty_step=0.1):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_step = difficulty_step
        self.performance_threshold = 0.8  # Threshold to increase difficulty

    def should_increase_difficulty(self, recent_performance):
        """Check if difficulty should be increased."""
        avg_performance = np.mean(recent_performance)
        return avg_performance >= self.performance_threshold

    def increase_difficulty(self):
        """Increase task difficulty."""
        self.current_difficulty = min(
            self.current_difficulty + self.difficulty_step, self.max_difficulty
        )
        print(f"Increased difficulty to {self.current_difficulty:.2f}")

    def get_current_difficulty(self):
        """Get current difficulty level."""
        return self.current_difficulty

    def adapt_environment(self, env, difficulty):
        """Adapt environment based on difficulty."""
        # Example adaptation: scale rewards, change dynamics, etc.
        env.reward_scale = 1.0 + difficulty
        env.noise_level = difficulty * 0.1
        env.complexity = int(difficulty * 10)

        return env


# Demonstration functions
def demonstrate_meta_learning():
    """Demonstrate meta-learning capabilities."""
    print("🧠 Meta-Learning Demo")

    # Create MAML agent
    maml_agent = MAMLAgent(obs_dim=10, action_dim=4)

    # Generate sample task data
    obs = torch.randn(20, 10)
    actions = torch.randint(0, 4, (20,))
    rewards = torch.randn(20)
    next_obs = torch.randn(20, 10)
    dones = torch.zeros(20)

    task_data = (obs, actions, rewards, next_obs, dones)

    # Adapt to task
    adapted_loss = maml_agent.adapt_to_task(task_data, n_inner_steps=3)
    print(f"MAML adaptation loss: {adapted_loss.item():.3f}")

    return maml_agent


def demonstrate_opponent_modeling():
    """Demonstrate opponent modeling."""
    print("\n🎭 Opponent Modeling Demo")

    # Create opponent model
    opponent_model = OpponentModel(obs_dim=10, action_dim=4)

    # Generate observation
    obs = torch.randn(10)

    # Predict opponent action
    opponent_probs = opponent_model.predict_opponent_action(obs)
    predicted_action = torch.argmax(opponent_probs)
    print(f"Predicted opponent action: {predicted_action.item()}")

    # Update belief
    reward = torch.tensor(1.0)
    belief_confidence = opponent_model.update_belief(obs, predicted_action, reward)
    print(f"Belief confidence: {belief_confidence.item():.3f}")

    return opponent_model


def demonstrate_population_training():
    """Demonstrate population-based training."""
    print("\n👥 Population-Based Training Demo")

    # Create PBT instance
    pbt = PopulationBasedTraining(population_size=5)

    # Initialize population (simplified)
    class DummyAgent:
        def __init__(self, lr=0.001, gamma=0.99, hidden_dim=64):
            self.lr = lr
            self.gamma = gamma
            self.hidden_dim = hidden_dim
            self.fitness = np.random.random()

    pbt.initialize_population(DummyAgent, {})

    # Simulate evolution
    fitness_scores = [agent["fitness"] for agent in pbt.population]
    print(f"Initial fitness scores: {fitness_scores}")

    pbt.evolve_population(fitness_scores)

    best_agent, best_hyperparams = pbt.get_best_agent()
    print(f"Best hyperparameters: {best_hyperparams}")

    return pbt


def demonstrate_self_play():
    """Demonstrate self-play training."""
    print("\n🤖 Self-Play Training Demo")

    # Create self-play training instance
    class DummyMultiAgent:
        def __init__(self, obs_dim=10, action_dim=4):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.fitness = 0.5

        def select_action(self, obs):
            return np.random.randint(0, self.action_dim)

        def update(self, data):
            self.fitness += 0.01  # Simulate improvement

    self_play = SelfPlayTraining(
        DummyMultiAgent, {"obs_dim": 10, "action_dim": 4}, n_opponents=3
    )

    # Run short training
    self_play.train_self_play(n_iterations=5, n_episodes_per_iter=2)

    print(f"Training completed. Main agent fitness: {self_play.main_agent.fitness:.3f}")

    return self_play


# Run demonstrations
print("🔄 Meta-Learning and Adaptation Systems")
meta_demo = demonstrate_meta_learning()
opponent_demo = demonstrate_opponent_modeling()
pbt_demo = demonstrate_population_training()
self_play_demo = demonstrate_self_play()

print("\n🚀 Meta-learning and adaptation implementations ready!")
print("✅ MAML, opponent modeling, population training, and self-play implemented!")
