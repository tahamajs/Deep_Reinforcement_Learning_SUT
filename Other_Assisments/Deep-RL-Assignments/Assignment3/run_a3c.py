# Author: Taha Majlesi - 810101504, University of Tehran

import argparse
from src.a3c_agent import A3C


def parse_arguments():
    """Command-line flags are defined here."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--policy_lr', dest='policy_lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic_lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")
    parser.add_argument('--reward_norm', dest='reward_normalizer', type=float,
                        default=100.0, help='Normalize rewards by.')
    parser.add_argument('--random_seed', dest='random_seed', type=int,
                        default=999, help='Random Seed')
    parser.add_argument('--test_episodes', dest='test_episodes', type=int,
                        default=25, help='Number of episodes to test on.')
    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=1000, help='Weights save interval.')
    parser.add_argument('--test_interval', dest='test_interval', type=int,
                        default=250, help='Test interval.')
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        default=25, help='Log interval.')
    parser.add_argument('--weights_path', dest='weights_path', type=str,
                        default=None, help='Pretrained weights file.')
    parser.add_argument('--det_eval', action="store_true", default=False,
                        help='Deterministic policy for testing.')
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main():
    """Parse command-line arguments and run A3C."""
    args = parse_arguments()

    # Create the environment.
    actor_critic = A3C(args, env='Breakout-v0')
    if not args.render: actor_critic.train()


if __name__ == '__main__':
    main()