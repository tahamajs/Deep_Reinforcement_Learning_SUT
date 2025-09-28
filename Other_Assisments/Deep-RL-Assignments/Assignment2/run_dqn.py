# Author: Taha Majlesi - 810101504, University of Tehran

import argparse
import tensorflow as tf
import keras
from src.dqn_agent import DQN_Agent

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', action="store_true", default=False)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--double_dqn', dest='double_dqn', type=int, default=0)
    parser.add_argument('--frameskip', dest='frameskip', type=int, default=1)
    parser.add_argument('--update_freq', dest='network_update_freq', type=int, default=10)
    parser.add_argument('--log_freq', dest='log_freq', type=int, default=25)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=100)
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=500)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=1.0)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()

# Author: Taha Majlesi - 810101504, University of Tehran

import argparse
import os
import numpy as np
import tensorflow as tf
import keras
import gymnasium as gym
from src.dqn_agent import DQN_Agent

def test_video(agent, env_name, episodes):
    # Usage:
    #   you can pass the arguments within agent.train() as:
    #       if episode % int(self.num_episodes/3) == 0:
    #           test_video(self, self.environment_name, episode)
    save_path = "%s/video-%s" % (env_name, episodes)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    print("Video recording the agent with epsilon {0:.4f}".format(agent.epsilon))
    while not done:
        q_values = agent.q_network.model.predict(state.reshape(1, -1))
        action = agent.greedy_policy(q_values)
        i = 0
        while (i < agent.args.frameskip) and not done:
            env.render()
            next_state, reward, done, info = env.step(action)
            reward_total.append(reward)
            i += 1
        state = next_state
    print("reward_total: {}".format(np.sum(reward_total)))
    env.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', action="store_true", default=False)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--double_dqn', dest='double_dqn', type=int, default=0)
    parser.add_argument('--frameskip', dest='frameskip', type=int, default=1)
    parser.add_argument('--update_freq', dest='network_update_freq', type=int, default=10)
    parser.add_argument('--log_freq', dest='log_freq', type=int, default=25)
    parser.add_argument('--test_freq', dest='test_freq', type=int, default=100)
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=500)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=1.0)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    q_agent = DQN_Agent(args)

    # Render output videos using the model loaded from file.
    if args.render:
        test_video(q_agent, args.env, 0)  # Using 0 as default episode
    else:
        q_agent.train()  # Train the model.

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()