# Author: Taha Majlesi - 810101504, University of Tehran

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from src.models import make_model
from src.utils import generate_episode, generate_dagger_episode, TV_distance

class Imitation:
    def __init__(self, env, expert_policy, lr=1e-3):
        self.env = env
        self.expert_policy = expert_policy
        self.lr = lr
        self.policy = make_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def train(self, num_rollouts, num_epochs=100, dagger=False):
        # Collect initial expert data
        expert_states = []
        expert_actions = []
        for _ in range(num_rollouts):
            states, actions, _ = generate_episode(self.env, self.expert_policy)
            expert_states.extend(states)
            expert_actions.extend(actions)
        expert_states = np.array(expert_states)
        expert_actions = np.array(expert_actions)

        # Train policy
        for epoch in range(num_epochs):
            expert_states, expert_actions = shuffle(expert_states, expert_actions)
            with tf.GradientTape() as tape:
                predictions = self.policy(expert_states)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(expert_actions, predictions))
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

            if dagger:
                # Collect more data with current policy
                student_states = []
                student_actions = []
                for _ in range(num_rollouts):
                    states, actions, _ = generate_dagger_episode(self.env, self.policy, self.expert_policy)
                    student_states.extend(states)
                    student_actions.extend(actions)
                expert_states = np.concatenate([expert_states, np.array(student_states)])
                expert_actions = np.concatenate([expert_actions, np.array(student_actions)])

        return self.policy

    def evaluate(self, num_rollouts=50):
        total_rewards = []
        for _ in range(num_rollouts):
            _, _, rewards = generate_episode(self.env, self.policy)
            total_rewards.append(np.sum(rewards))
        return np.mean(total_rewards), np.std(total_rewards)

    def evaluate_convergence(self, expert_states, num_rollouts=50):
        student_states = []
        for _ in range(num_rollouts):
            states, _, _ = generate_episode(self.env, self.policy)
            student_states.extend(states)
        return TV_distance(expert_states, student_states)