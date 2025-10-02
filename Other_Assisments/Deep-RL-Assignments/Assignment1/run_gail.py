import tensorflow as tf
import gymnasium as gym
from src.gail import GAIL
def main():
    env = gym.make("CartPole-v1")
    expert_policy = tf.keras.models.load_model("expert.h5")
    gail = GAIL(env, expert_policy)
    gail_policy = gail.train(num_rollouts=10, num_epochs=100)
    mean_reward, std_reward = gail.evaluate()
    print(f"GAIL Mean Reward: {mean_reward}, Std: {std_reward}")
if __name__ == "__main__":
    main()