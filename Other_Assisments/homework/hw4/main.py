import os
import argparse
import time
import sys

try:
    import tensorflow.compat.v1 as tf  # type: ignore
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # type: ignore

# MuJoCo environments that require mujoco-py
MUJOCO_ENVS = {'HalfCheetah', 'InvertedPendulum', 'Hopper', 'Walker2d', 'Ant', 'Humanoid'}

def check_mujoco_available():
    """Check if MuJoCo is available."""
    try:
        import mujoco_py
        return True
    except (ImportError, Exception):
        return False

parser = argparse.ArgumentParser()
parser.add_argument('question', type=str, choices=('q1', 'q2', 'q3'))
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--env', type=str, default='HalfCheetah', choices=('HalfCheetah',))
parser.add_argument('--render', action='store_true')
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--nn_layers', type=int, default=1)
args = parser.parse_args()

# Check for MuJoCo dependency BEFORE importing environment
if args.env in MUJOCO_ENVS and not check_mujoco_available():
    print(f"\n❌ ERROR: Environment '{args.env}' requires MuJoCo, but it's not installed.")
    print("\n📦 To install MuJoCo:")
    print("   1. Download MuJoCo 2.1.0 from: https://github.com/deepmind/mujoco/releases")
    print("   2. Extract to ~/.mujoco/mujoco210")
    print("   3. Install mujoco-py: pip install mujoco-py")
    print("   4. On macOS: brew install gcc")
    print("\nExiting HW4 (all experiments require MuJoCo).\n")
    sys.exit(1)

# Import MuJoCo-dependent modules AFTER the check
from half_cheetah_env import HalfCheetahEnv
from logger import logger
from model_based_rl import ModelBasedRL

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format(args.env,
                                args.question,
                                args.exp_name if args.exp_name else time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = os.path.join(data_dir, exp_name)
assert not os.path.exists(exp_dir),\
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(exp_dir)
os.makedirs(exp_dir, exist_ok=True)
logger.setup(exp_name, os.path.join(exp_dir, 'log.txt'), 'debug')

env = {
    'HalfCheetah': HalfCheetahEnv()
}[args.env]

mbrl = ModelBasedRL(env=env,
                    render=args.render,
                    mpc_horizon=args.mpc_horizon,
                    num_random_action_selection=args.num_random_action_selection,
                    nn_layers=args.nn_layers)

run_func = {
    'q1': mbrl.run_q1,
    'q2': mbrl.run_q2,
    'q3': mbrl.run_q3
}[args.question]
run_func()