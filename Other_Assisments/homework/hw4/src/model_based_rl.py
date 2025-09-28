"""
Model-Based Reinforcement Learning Agent

This module contains the model-based RL agent with dynamics model and MPC policy.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
import tensorflow as tf
from collections import namedtuple


class Dataset:
    """Dataset for storing transition tuples."""

    def __init__(self):
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []

    def add(self, state, action, next_state, reward, done):
        """Add a transition to the dataset."""
        self._states.append(state)
        self._actions.append(action)
        self._next_states.append(next_state)
        self._rewards.append(reward)
        self._dones.append(done)

    def size(self):
        """Return the size of the dataset."""
        return len(self._states)

    def get_all(self):
        """Get all transitions as numpy arrays."""
        return {
            'states': np.array(self._states),
            'actions': np.array(self._actions),
            'next_states': np.array(self._next_states),
            'rewards': np.array(self._rewards),
            'dones': np.array(self._dones)
        }

    def random_iterator(self, batch_size):
        """Create a random iterator over the dataset."""
        indices = np.random.permutation(self.size())

        for i in range(0, self.size(), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield {
                'states': np.array(self._states)[batch_indices],
                'actions': np.array(self._actions)[batch_indices],
                'next_states': np.array(self._next_states)[batch_indices],
                'rewards': np.array(self._rewards)[batch_indices],
                'dones': np.array(self._dones)[batch_indices]
            }


class DynamicsModel:
    """Neural network dynamics model."""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """Initialize dynamics model.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build model
        self._build_model(hidden_dims)

    def _build_model(self, hidden_dims):
        """Build the neural network model."""
        self.input_state = tf.placeholder(tf.float32, [None, self.state_dim])
        self.input_action = tf.placeholder(tf.float32, [None, self.action_dim])
        self.target_next_state = tf.placeholder(tf.float32, [None, self.state_dim])

        # Concatenate state and action
        x = tf.concat([self.input_state, self.input_action], axis=1)

        # Hidden layers
        for dim in hidden_dims:
            x = tf.layers.dense(x, dim, activation=tf.nn.relu)

        # Output layer (delta prediction)
        self.predicted_delta = tf.layers.dense(x, self.state_dim, activation=None)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.predicted_delta - (self.target_next_state - self.input_state)))

        # Training
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, states, actions, sess):
        """Predict next states.

        Args:
            states: Current states
            actions: Actions taken
            sess: TensorFlow session

        Returns:
            Predicted next states
        """
        feed_dict = {
            self.input_state: states,
            self.input_action: actions
        }
        deltas = sess.run(self.predicted_delta, feed_dict=feed_dict)
        return states + deltas

    def train(self, states, actions, next_states, sess, epochs=1, batch_size=512):
        """Train the dynamics model.

        Args:
            states: Current states
            actions: Actions taken
            next_states: Next states
            sess: TensorFlow session
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training losses
        """
        losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            epoch_losses = []

            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i + batch_size]

                feed_dict = {
                    self.input_state: states[batch_indices],
                    self.input_action: actions[batch_indices],
                    self.target_next_state: next_states[batch_indices]
                }

                _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                epoch_losses.append(loss)

            losses.append(np.mean(epoch_losses))

        return losses


class MPCPolicy:
    """Model Predictive Control policy."""

    def __init__(self, dynamics_model, horizon=15, num_random_actions=4096):
        """Initialize MPC policy.

        Args:
            dynamics_model: Trained dynamics model
            horizon: MPC planning horizon
            num_random_actions: Number of random actions to sample
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_random_actions = num_random_actions

    def get_action(self, state, sess):
        """Get the best action using MPC.

        Args:
            state: Current state
            sess: TensorFlow session

        Returns:
            Best action
        """
        # Sample random action sequences
        action_dim = self.dynamics_model.action_dim
        action_sequences = np.random.uniform(-1, 1, (self.num_random_actions, self.horizon, action_dim))

        # Evaluate each action sequence
        best_reward = -np.inf
        best_action = None

        for action_seq in action_sequences:
            total_reward = 0
            current_state = state.copy()

            for action in action_seq:
                # Predict next state
                next_state = self.dynamics_model.predict(
                    current_state.reshape(1, -1),
                    action.reshape(1, -1),
                    sess
                )[0]

                # Simple reward function (can be customized)
                reward = -np.sum(np.square(action))  # Penalize large actions
                total_reward += reward
                current_state = next_state

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action_seq[0]  # First action in sequence

        return best_action


class ModelBasedRLAgent:
    """Model-Based Reinforcement Learning Agent."""

    def __init__(self, env, num_init_random_rollouts=10, max_rollout_length=500,
                 num_onpolicy_iters=10, num_onpolicy_rollouts=10, training_epochs=60,
                 training_batch_size=512, mpc_horizon=15, num_random_action_selection=4096,
                 nn_layers=1):
        """Initialize MBRL agent.

        Args:
            env: Environment
            num_init_random_rollouts: Number of initial random rollouts
            max_rollout_length: Maximum rollout length
            num_onpolicy_iters: Number of on-policy iterations
            num_onpolicy_rollouts: Number of on-policy rollouts per iteration
            training_epochs: Number of training epochs for dynamics model
            training_batch_size: Training batch size
            mpc_horizon: MPC planning horizon
            num_random_action_selection: Number of random actions for MPC
            nn_layers: Number of layers in dynamics model
        """
        self.env = env
        self.num_init_random_rollouts = num_init_random_rollouts
        self.max_rollout_length = max_rollout_length
        self.num_onpolicy_iters = num_onpolicy_iters
        self.num_onpolicy_rollouts = num_onpolicy_rollouts
        self.training_epochs = training_epochs
        self.training_batch_size = training_batch_size
        self.mpc_horizon = mpc_horizon
        self.num_random_action_selection = num_random_action_selection
        self.nn_layers = nn_layers

        # Get environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Initialize components
        self.dataset = Dataset()
        self.dynamics_model = None
        self.policy = None

        # TensorFlow session
        self.sess = None

    def init_tf_sess(self):
        """Initialize TensorFlow session."""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def gather_rollouts(self, policy, num_rollouts, render=False):
        """Gather rollouts using a policy.

        Args:
            policy: Policy to use for action selection
            num_rollouts: Number of rollouts to gather
            render: Whether to render episodes

        Returns:
            Dataset of transitions
        """
        dataset = Dataset()

        for _ in range(num_rollouts):
            state = self.env.reset()
            done = False
            t = 0

            while not done and t < self.max_rollout_length:
                if render:
                    self.env.render()

                action = policy.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                dataset.add(state, action, next_state, reward, done)
                state = next_state
                t += 1

        return dataset

    def train_dynamics_model(self):
        """Train the dynamics model."""
        # Get training data
        data = self.dataset.get_all()

        # Create dynamics model
        hidden_dims = [256] * self.nn_layers
        self.dynamics_model = DynamicsModel(self.state_dim, self.action_dim, hidden_dims)

        # Train model
        losses = self.dynamics_model.train(
            data['states'], data['actions'], data['next_states'],
            self.sess, self.training_epochs, self.training_batch_size
        )

        return losses

    def create_mpc_policy(self):
        """Create MPC policy using trained dynamics model."""
        self.policy = MPCPolicy(
            self.dynamics_model,
            horizon=self.mpc_horizon,
            num_random_actions=self.num_random_action_selection
        )

    def run_q1(self):
        """Run question 1: Gather random data and train dynamics model."""
        print("Running Q1: Gathering random data and training dynamics model")

        # Gather initial random data
        random_policy = RandomPolicy(self.env)
        random_dataset = self.gather_rollouts(random_policy, self.num_init_random_rollouts)
        self.dataset = random_dataset

        # Train dynamics model
        losses = self.train_dynamics_model()
        print(f"Training completed. Final loss: {losses[-1]:.6f}")

    def run_q2(self):
        """Run question 2: MPC with random shooting."""
        print("Running Q2: MPC with random shooting")

        # Ensure we have a trained dynamics model
        if self.dynamics_model is None:
            self.run_q1()

        # Create MPC policy
        self.create_mpc_policy()

        # Test the policy
        test_rollouts = self.gather_rollouts(self.policy, 5, render=True)
        print(f"Collected {test_rollouts.size()} transitions with MPC policy")

    def run_q3(self):
        """Run question 3: On-policy MBRL."""
        print("Running Q3: On-policy MBRL")

        # Start with random data
        if self.dataset.size() == 0:
            self.run_q1()

        # On-policy iterations
        for iteration in range(self.num_onpolicy_iters):
            print(f"On-policy iteration {iteration + 1}/{self.num_onpolicy_iters}")

            # Train dynamics model on current dataset
            losses = self.train_dynamics_model()

            # Create MPC policy
            self.create_mpc_policy()

            # Gather on-policy data
            onpolicy_dataset = self.gather_rollouts(self.policy, self.num_onpolicy_rollouts)

            # Add to dataset
            onpolicy_data = onpolicy_dataset.get_all()
            for i in range(onpolicy_dataset.size()):
                self.dataset.add(
                    onpolicy_data['states'][i],
                    onpolicy_data['actions'][i],
                    onpolicy_data['next_states'][i],
                    onpolicy_data['rewards'][i],
                    onpolicy_data['dones'][i]
                )

            print(f"Iteration {iteration + 1} completed. Dataset size: {self.dataset.size()}")


class RandomPolicy:
    """Random action policy."""

    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        """Sample random action."""
        return self.env.action_space.sample()