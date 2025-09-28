"""
DDPG Training Script

This script provides a modular implementation of DDPG, TD3, and HER algorithms
for continuous control tasks.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import argparse
import gymnasium as gym
import envs  # Custom environments
from src.ddpg_agent import DDPGAgent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DDPG Training Script")

    # Training parameters
    parser.add_argument(
        "--num-episodes",
        dest="num_episodes",
        type=int,
        default=50000,
        help="Number of episodes to train on.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ddpg",
        choices=["ddpg", "td3", "her"],
        help="Training algorithm (ddpg | td3 | her)",
    )

    # Network parameters
    parser.add_argument(
        "--actor_lr",
        dest="actor_lr",
        type=float,
        default=1e-4,
        help="The actor's learning rate.",
    )
    parser.add_argument(
        "--critic_lr",
        dest="critic_lr",
        type=float,
        default=1e-3,
        help="The critic's learning rate.",
    )
    parser.add_argument(
        "--tau",
        dest="tau",
        type=float,
        default=0.05,
        help="The update parameter for soft updates",
    )
    parser.add_argument(
        "--gamma", dest="gamma", type=float, default=0.98, help="Discount factor"
    )

    # Training hyperparameters
    parser.add_argument(
        "--buffer_size",
        dest="buffer_size",
        type=int,
        default=1000000,
        help="Replay buffer size",
    )
    parser.add_argument(
        "--burn_in",
        dest="burn_in",
        type=int,
        default=5000,
        help="Number of transitions to collect before training starts",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1024,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epsilon",
        dest="epsilon",
        type=float,
        default=0.1,
        help="Epsilon for action noise",
    )

    # TD3 specific parameters
    parser.add_argument(
        "--target_action_sigma",
        dest="target_action_sigma",
        type=float,
        default=0.05,
        help="Action smoothing noise for TD3",
    )
    parser.add_argument(
        "--clip",
        dest="clip",
        type=float,
        default=0.05,
        help="Clip value for action smoothing in TD3",
    )
    parser.add_argument(
        "--policy_update_frequency",
        dest="policy_update_frequency",
        type=int,
        default=2,
        help="How often to update policy wrt critic in TD3",
    )

    # Training updates
    parser.add_argument(
        "--num_update_iters",
        dest="num_update_iters",
        type=int,
        default=4,
        help="How many times to update networks per episode",
    )

    # Logging and saving
    parser.add_argument(
        "--save_interval",
        dest="save_interval",
        type=int,
        default=2000,
        help="Model save interval",
    )
    parser.add_argument(
        "--test_interval",
        dest="test_interval",
        type=int,
        default=100,
        help="Evaluation interval",
    )
    parser.add_argument(
        "--log_interval",
        dest="log_interval",
        type=int,
        default=100,
        help="Logging interval",
    )
    parser.add_argument(
        "--weights_path",
        dest="weights_path",
        type=str,
        default=None,
        help="Path to pretrained weights",
    )

    # Training mode
    parser.add_argument(
        "--train", action="store_true", default=True, help="Do training"
    )

    # Network initialization
    parser.add_argument(
        "--custom_init",
        action="store_true",
        default=False,
        help="Use custom weight initialization",
    )

    # Rendering
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument(
        "--render", dest="render", action="store_true", help="Render the environment"
    )
    parser_group.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Don't render the environment",
    )
    parser.set_defaults(render=False)

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    # Create environment
    env = gym.make("Pushing2D-v0")

    # Create output filename
    outfile_name = f"{args.algorithm}_log.txt"

    # Initialize DDPG agent
    agent = DDPGAgent(args, env, outfile_name)

    # Train the agent
    agent.train(args.num_episodes)


if __name__ == "__main__":
    main()
