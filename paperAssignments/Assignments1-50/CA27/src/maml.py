"""Model-Agnostic Meta-Learning (MAML) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import numpy as np
from .config import MAMLConfig
from .tasks import Task
from .utils import Trajectory, collect_trajectory, compute_returns


class PolicyNetwork(nn.Module):
    """Simple MLP policy for MAML."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MAML:
    """MAML agent for meta-learning."""
    def __init__(self, config: MAMLConfig):
        self.config = config
        self.policy = PolicyNetwork(config.obs_dim, config.action_dim, config.hidden_dim)
        self.meta_optimizer = optim.Adam(
            self.policy.parameters(), lr=config.meta_lr
        )

    def compute_policy_loss(self, trajectory: Dict[str, torch.Tensor], policy: nn.Module) -> torch.Tensor:
        """Compute policy gradient loss for a trajectory."""
        states = trajectory['states']
        actions = trajectory['actions']
        returns = trajectory['returns']

        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        loss = -(log_probs * returns).mean()
        return loss

    def collect_trajectory(self, env, policy: nn.Module, max_steps: int = 200) -> Dict[str, torch.Tensor]:
        """Collect one trajectory using a policy network.

        This method handles different Gym APIs for ``reset()`` and ``step()`` and
        returns a dictionary of tensors suitable for computing policy gradients.
        """
        states, actions, rewards, log_probs = [], [], [], []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get action logits
            logits = policy(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Support different step() return signatures
            step_result = env.step(action.item())
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                next_state, reward, done = step_result[0], step_result[1], bool(step_result[2])

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            log_probs.append(log_prob)

            state = next_state
            if done:
                break

        returns = compute_returns(rewards, self.config.gamma)

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.stack(actions),
            'returns': returns,
            'log_probs': torch.stack(log_probs)
        }

    def inner_loop_update(self, task_env, policy: nn.Module) -> nn.Module:
        """Perform inner loop adaptation."""
        # Clone current parameters
        adapted_policy = PolicyNetwork(
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim
        )
        adapted_policy.load_state_dict(policy.state_dict())

        for step in range(self.config.inner_steps):
            # Collect trajectories
            trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(5)  # 5 trajectories per inner step
            ]

            # Compute loss
            total_loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in trajectories
            ) / len(trajectories)

            # Compute gradients
            grads = torch.autograd.grad(
                total_loss, adapted_policy.parameters(),
                create_graph=True  # Enable second-order derivatives
            )

            # Manual SGD update
            with torch.no_grad():
                for param, grad in zip(adapted_policy.parameters(), grads):
                    param.data = param.data - self.config.inner_lr * grad.data

        return adapted_policy

    def meta_train_step(self, task_envs: List[Any]) -> float:
        """One meta-training step."""
        meta_loss = 0

        for task_env in task_envs:
            # Inner loop: adapt to task
            adapted_policy = self.inner_loop_update(task_env, self.policy)

            # Collect test trajectories with adapted policy
            test_trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(10)  # More trajectories for meta-loss
            ]

            # Compute meta-loss
            task_loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in test_trajectories
            ) / len(test_trajectories)

            meta_loss += task_loss

        meta_loss = meta_loss / len(task_envs)

        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_new_task(self, task_env, num_adapt_steps: int = 5) -> nn.Module:
        """Adapt to new task at test time."""
        # Clone current policy
        adapted_policy = PolicyNetwork(
            self.config.obs_dim, self.config.action_dim, self.config.hidden_dim
        )
        adapted_policy.load_state_dict(self.policy.state_dict())

        optimizer = optim.SGD(
            adapted_policy.parameters(), lr=self.config.inner_lr
        )

        for _ in range(num_adapt_steps):
            trajectories = [
                self.collect_trajectory(task_env, adapted_policy)
                for _ in range(3)
            ]

            loss = sum(
                self.compute_policy_loss(traj, adapted_policy)
                for traj in trajectories
            ) / len(trajectories)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_policy

    def adapt_and_evaluate(self, task: Task, adaptation_steps: int = 1, eval_steps: int = 200) -> float:
        """Adapt and evaluate on a task."""
        adapted_policy = self.adapt_to_new_task(task.env, adaptation_steps)
        trajectory = collect_trajectory(task.env, adapted_policy, max_steps=eval_steps)
        return sum(trajectory.rewards)

    def train(self, task_distribution, num_meta_iterations: int = 100, meta_batch_size: int = 5) -> List[float]:
        """Train the meta-learner."""
        losses = []
        for iteration in range(num_meta_iterations):
            # Sample batch of tasks
            task_batch = task_distribution.sample(meta_batch_size)

            # Meta-training step
            loss = self.meta_train_step([task.env for task in task_batch])
            losses.append(loss)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{num_meta_iterations}, Loss: {loss:.4f}")

        return losses