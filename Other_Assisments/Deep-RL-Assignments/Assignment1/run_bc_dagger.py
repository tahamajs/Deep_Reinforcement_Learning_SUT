import numpy as np
import tensorflow as tf
import gymnasium as gym
from src.bc_dagger import Imitation
from src.utils import generate_episode
def main():
    env = gym.make("CartPole-v1")
    expert_policy = tf.keras.models.load_model("expert.h5")
    bc = Imitation(env, expert_policy)
    bc_policy = bc.train(num_rollouts=10, num_epochs=100, dagger=False)
    bc_mean, bc_std = bc.evaluate()
    print(f"BC Mean Reward: {bc_mean}, Std: {bc_std}")
    dagger = Imitation(env, expert_policy)
    dagger_policy = dagger.train(num_rollouts=10, num_epochs=100, dagger=True)
    dagger_mean, dagger_std = dagger.evaluate()
    print(f"DAgger Mean Reward: {dagger_mean}, Std: {dagger_std}")
    expert_states = []
    for _ in range(50):
        states, _, _ = generate_episode(env, expert_policy)
        expert_states.extend(states)
    expert_states = np.array(expert_states)

    bc_convergence = bc.evaluate_convergence(expert_states)
    dagger_convergence = dagger.evaluate_convergence(expert_states)
    print(f"BC TV Distance: {bc_convergence}")
    print(f"DAgger TV Distance: {dagger_convergence}")
if __name__ == "__main__":
    main()