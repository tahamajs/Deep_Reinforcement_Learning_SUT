# BootstrapDQN Implementation Guide

This guide will help you complete all the TODO sections in the notebook.

## ✅ EpsGreedyDQNAgent (Already Complete)

The EpsGreedyDQNAgent is already fully implemented in the notebook. No changes needed.

## Cell 26: MultiHeadQNet + BootstrapDQNAgent

Replace the entire cell content with:

```python
class MultiHeadQNet(nn.Module):
    """
    Multi-head Q-network for Bootstrap DQN.
    """

    def __init__(self, input_dim, output_dim, num_heads=10, hidden_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multiple heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_heads)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, x, head_idx=None):
        """
        Forward pass through the network.
        If head_idx is provided, use only that head.
        Otherwise, return outputs from all heads.
        """
        shared_features = self.shared_layers(x)

        if head_idx is not None:
            return self.heads[head_idx](shared_features)
        else:
            # Return outputs from all heads
            outputs = []
            for head in self.heads:
                outputs.append(head(shared_features))
            return torch.stack(outputs, dim=1)  # Shape: (batch_size, num_heads, output_dim)


class BootstrapDQNAgent(EpsGreedyDQNAgent):
    """
    Bootstrap DQN agent.
    """

    def __init__(self, k: int = 10, bernoulli_p: float = 0.5, **kwargs):
        self.k = k
        self.bernoulli_p = bernoulli_p
        super().__init__(**kwargs)

        # Initialize Bernoulli distribution for mask generation
        self.mask_dist = torch.distributions.Bernoulli(probs=bernoulli_p)

        # Current head to use for action selection
        self.current_head = 0

    def _create_network(self):
        """Create multi-head Q-network and target network."""
        self.q_network = MultiHeadQNet(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n,
            num_heads=self.k,
            hidden_dim=256
        ).to(self.device)

        self.target_network = MultiHeadQNet(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n,
            num_heads=self.k,
            hidden_dim=256
        ).to(self.device)

    def _create_replay_buffer(self, max_size=1000000):
        """Create replay buffer with masks for each head."""
        self.replay_buffer = ReplayBuffer(
            [
                ("state", (self.env.observation_space.shape[0],), torch.float32),
                ("action", (), torch.int64),
                ("reward", (), torch.float32),
                ("next_state", (self.env.observation_space.shape[0],), torch.float32),
                ("done", (), torch.float32),
                ("mask", (self.k,), torch.float32),  # Bootstrap mask for each head
            ],
            max_size=max_size,
            device=self.device,
        )

    def _preprocess_add(self, state, action, reward, next_state, done):
        """Generate bootstrap masks before adding to replay buffer."""
        # Generate binary mask using Bernoulli distribution
        mask = self.mask_dist.sample((self.k,)).to(self.device)
        return state, action, reward, next_state, done, mask

    def _compute_loss(self, batch):
        """
        Compute the loss for Bootstrap DQN.
        Each head is trained only on experiences where its mask is 1.
        """
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]
        masks = batch["mask"]

        batch_size = states.shape[0]

        # Get Q-values for all heads: (batch_size, k, num_actions)
        q_values_all_heads = self.q_network(states)

        # Get Q-values for selected actions for each head
        # actions: (batch_size,) -> (batch_size, 1, 1) -> (batch_size, k, 1)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, self.k, 1)
        q_values = q_values_all_heads.gather(2, actions_expanded).squeeze(2)  # (batch_size, k)

        # Get target Q-values
        with torch.no_grad():
            next_q_values_all_heads = self.target_network(next_states)
            next_q_values = next_q_values_all_heads.max(2)[0]  # (batch_size, k)

            # Compute expected Q-values for each head
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Apply masks: only compute loss for experiences where mask is 1
        loss = nn.SmoothL1Loss(reduction='none')(q_values, expected_q_values)
        masked_loss = (loss * masks).sum() / masks.sum()

        return masked_loss

    def _act_in_training(self, state):
        """
        Select an action during training using the current head.
        """
        self._decay_eps()
        if torch.rand(1).item() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor, head_idx=self.current_head)
                return q_values.argmax().item()

    def _act_in_eval(self, state):
        """
        Select an action during evaluation.
        Use the mean Q-values across all heads.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values_all_heads = self.q_network(state_tensor)  # (1, k, num_actions)
            mean_q_values = q_values_all_heads.mean(dim=1)  # (1, num_actions)
            return mean_q_values.argmax().item()

    def _episode(self):
        """
        Override to randomly select a head at the start of each episode.
        """
        # Randomly select a head for this episode
        self.current_head = random.randint(0, self.k - 1)
        super()._episode()

    def _save_dict(self):
        save_dict = super()._save_dict()
        save_dict["k"] = self.k
        save_dict["bernoulli_p"] = self.bernoulli_p
        save_dict["current_head"] = self.current_head
        return save_dict
```

## Cell 32: PriorMultiHeadQNet + RPFBootstrapDQNAgent

Replace the entire cell content with:

```python
class PriorMultiHeadQNet(MultiHeadQNet):
    """
    Multi-head Q-network with a prior network for RPF Bootstrap DQN.
    The prior network is shallower and provides exploration bonuses.
    """

    def __init__(self, input_dim, output_dim, num_heads=10, hidden_dim=256, prior_hidden_dim=128):
        super().__init__(input_dim, output_dim, num_heads, hidden_dim)

        # Prior network (shallower)
        self.prior_shared = nn.Sequential(
            nn.Linear(input_dim, prior_hidden_dim),
            nn.ReLU(),
        )

        self.prior_heads = nn.ModuleList([
            nn.Linear(prior_hidden_dim, output_dim) for _ in range(num_heads)
        ])

        # Initialize prior network weights
        for m in list(self.prior_shared.modules()) + list(self.prior_heads.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, x, head_idx=None, return_prior=False):
        """
        Forward pass through the network.
        If return_prior is True, also return prior network outputs.
        """
        if return_prior:
            # Get trainable network output
            trainable_output = super().forward(x, head_idx)

            # Get prior network output
            prior_features = self.prior_shared(x)
            if head_idx is not None:
                prior_output = self.prior_heads[head_idx](prior_features)
            else:
                prior_outputs = []
                for head in self.prior_heads:
                    prior_outputs.append(head(prior_features))
                prior_output = torch.stack(prior_outputs, dim=1)

            return trainable_output, prior_output
        else:
            return super().forward(x, head_idx)


class RPFBootstrapDQNAgent(BootstrapDQNAgent):
    """
    Randomized Prior Functions (RPF) Bootstrap DQN agent.
    Uses a fixed prior network to provide exploration bonuses.
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        self.beta = beta  # Weight for prior network
        super().__init__(**kwargs)

    def _create_network(self):
        """Create multi-head Q-network with prior and target network."""
        self.q_network = PriorMultiHeadQNet(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n,
            num_heads=self.k,
            hidden_dim=256,
            prior_hidden_dim=128
        ).to(self.device)

        self.target_network = PriorMultiHeadQNet(
            input_dim=self.env.observation_space.shape[0],
            output_dim=self.env.action_space.n,
            num_heads=self.k,
            hidden_dim=256,
            prior_hidden_dim=128
        ).to(self.device)

        # Freeze prior network parameters
        for param in self.q_network.prior_shared.parameters():
            param.requires_grad = False
        for head in self.q_network.prior_heads:
            for param in head.parameters():
                param.requires_grad = False

        for param in self.target_network.prior_shared.parameters():
            param.requires_grad = False
        for head in self.target_network.prior_heads:
            for param in head.parameters():
                param.requires_grad = False

    def _act_in_training(self, state):
        """
        Select an action during training using the current head.
        Combines trainable network with prior network.
        """
        self._decay_eps()
        if torch.rand(1).item() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                trainable_q, prior_q = self.q_network(state_tensor, head_idx=self.current_head, return_prior=True)
                combined_q = trainable_q + self.beta * prior_q
                return combined_q.argmax().item()

    def _act_in_eval(self, state):
        """
        Select an action during evaluation.
        Use the mean Q-values across all heads (trainable + prior).
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            trainable_q, prior_q = self.q_network(state_tensor, return_prior=True)
            combined_q = trainable_q + self.beta * prior_q
            mean_q_values = combined_q.mean(dim=1)
            return mean_q_values.argmax().item()

    def _compute_loss(self, batch):
        """
        Compute the loss for RPF Bootstrap DQN.
        Only the trainable network is updated; prior network is fixed.
        """
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]
        masks = batch["mask"]

        batch_size = states.shape[0]

        # Get Q-values for all heads (only trainable part)
        q_values_all_heads, _ = self.q_network(states, return_prior=True)

        # Get Q-values for selected actions for each head
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(batch_size, self.k, 1)
        q_values = q_values_all_heads.gather(2, actions_expanded).squeeze(2)

        # Get target Q-values (trainable + prior)
        with torch.no_grad():
            next_trainable_q, next_prior_q = self.target_network(next_states, return_prior=True)
            next_combined_q = next_trainable_q + self.beta * next_prior_q
            next_q_values = next_combined_q.max(2)[0]

            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Apply masks
        loss = nn.SmoothL1Loss(reduction='none')(q_values, expected_q_values)
        masked_loss = (loss * masks).sum() / masks.sum()

        return masked_loss

    def _save_dict(self):
        save_dict = super()._save_dict()
        save_dict["beta"] = self.beta
        return save_dict
```

## Cell 37: UEBootstrapDQNAgent

Replace the entire cell content with:

```python
class UEBootstrapDQNAgent(RPFBootstrapDQNAgent):
    """
    Uncertainty-based Exploration (UE) Bootstrap DQN agent.
    Uses uncertainty (variance across heads) for exploration.
    """

    def __init__(self, xi: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.xi = xi  # Weight for uncertainty bonus

    def _act_in_training(self, state):
        """
        Select an action during training using uncertainty-based exploration.
        Action value = mean Q + xi * std(Q across heads)
        """
        self._decay_eps()
        if torch.rand(1).item() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                trainable_q, prior_q = self.q_network(state_tensor, return_prior=True)
                combined_q = trainable_q + self.beta * prior_q  # (1, k, num_actions)

                # Compute mean and std across heads
                mean_q = combined_q.mean(dim=1)  # (1, num_actions)
                std_q = combined_q.std(dim=1)    # (1, num_actions)

                # Add uncertainty bonus
                action_values = mean_q + self.xi * std_q
                return action_values.argmax().item()

    def _act_in_eval(self, state):
        """
        Select an action during evaluation.
        Use the mean Q-values across all heads without uncertainty bonus.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            trainable_q, prior_q = self.q_network(state_tensor, return_prior=True)
            combined_q = trainable_q + self.beta * prior_q
            mean_q_values = combined_q.mean(dim=1)
            return mean_q_values.argmax().item()

    def _save_dict(self):
        save_dict = super()._save_dict()
        save_dict["xi"] = self.xi
        return save_dict

    def _wandb_train_step_dict(self):
        """Add uncertainty metrics to logging."""
        log_dict = super()._wandb_train_step_dict()

        # Log average uncertainty if we have recent states
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer) > 0:
            # Sample a small batch to compute average uncertainty
            try:
                sample_batch = self.replay_buffer.sample(min(32, len(self.replay_buffer)))
                with torch.no_grad():
                    trainable_q, prior_q = self.q_network(sample_batch["state"], return_prior=True)
                    combined_q = trainable_q + self.beta * prior_q
                    std_q = combined_q.std(dim=1).mean().item()
                    log_dict["train_step/avg_uncertainty"] = std_q
            except:
                pass

        return log_dict
```

## Summary

You need to replace 3 code cells in total:

1. **Cell 26**: MultiHeadQNet + BootstrapDQNAgent
2. **Cell 32**: PriorMultiHeadQNet + RPFBootstrapDQNAgent
3. **Cell 37**: UEBootstrapDQNAgent

All other cells (EpsGreedyDQNAgent, training cells, etc.) are already complete and should not be modified.

After making these changes, you can run the notebook to train and evaluate all four algorithms!
