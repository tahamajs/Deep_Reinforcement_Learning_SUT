# Author: Taha Majlesi - 810101504, University of Tehran

import gymnasium as gym
from src.cmaes import CMAES


def main():
    env = gym.make("CartPole-v1")
    num_params = (
        env.observation_space.shape[0] * env.action_space.n + env.action_space.n
    )
    cmaes = CMAES(env, num_params)
    best_params = cmaes.train(num_rollouts=5, num_epochs=10)
    mean_reward, std_reward = cmaes.evaluate(best_params)
    print(f"CMA-ES Mean Reward: {mean_reward}, Std: {std_reward}")


if __name__ == "__main__":
    main()
