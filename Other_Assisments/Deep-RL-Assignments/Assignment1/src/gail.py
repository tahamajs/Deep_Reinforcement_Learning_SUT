# Author: Taha Majlesi - 810101504, University of Tehran

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from src.models import make_model
from src.utils import generate_episode, generate_GAIL_episode


class GAIL:
    def __init__(self, env, expert_policy, lr=1e-3):
        self.env = env
        self.expert_policy = expert_policy
        self.lr = lr
        self.policy = make_model()
        self.discriminator = make_model()
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def train(self, num_rollouts, num_epochs=100):
        # Collect expert data
        expert_states = []
        expert_actions = []
        for _ in range(num_rollouts):
            states, actions, _ = generate_episode(self.env, self.expert_policy)
            expert_states.extend(states)
            expert_actions.extend(actions)
        expert_states = np.array(expert_states)
        expert_actions = np.array(expert_actions)

        for epoch in range(num_epochs):
            # Train discriminator
            student_states = []
            student_actions = []
            for _ in range(num_rollouts):
                states, actions, _ = generate_GAIL_episode(self.env, self.policy)
                student_states.extend(states)
                student_actions.extend(actions)
            student_states = np.array(student_states)
            student_actions = np.array(student_actions)

            expert_labels = np.ones(len(expert_states))
            student_labels = np.zeros(len(student_states))

            all_states = np.concatenate([expert_states, student_states])
            all_actions = np.concatenate([expert_actions, student_actions])
            all_labels = np.concatenate([expert_labels, student_labels])

            all_states, all_actions, all_labels = shuffle(
                all_states, all_actions, all_labels
            )

            with tf.GradientTape() as tape:
                predictions = self.discriminator(
                    tf.concat([all_states, all_actions], axis=1)
                )
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(all_labels, predictions)
                )
            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

            # Train policy
            policy_states = []
            policy_actions = []
            for _ in range(num_rollouts):
                states, actions, _ = generate_GAIL_episode(
                    self.env, self.policy, self.discriminator
                )
                policy_states.extend(states)
                policy_actions.extend(actions)
            policy_states = np.array(policy_states)
            policy_actions = np.array(policy_actions)

            with tf.GradientTape() as tape:
                predictions = self.policy(policy_states)
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        policy_actions, predictions
                    )
                )
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(
                zip(gradients, self.policy.trainable_variables)
            )

        return self.policy

    def evaluate(self, num_rollouts=50):
        total_rewards = []
        for _ in range(num_rollouts):
            _, _, rewards = generate_episode(self.env, self.policy)
            total_rewards.append(np.sum(rewards))
        return np.mean(total_rewards), np.std(total_rewards)
