"""
Main script to test LQR and iLQR on TwoLinkArmEnv-v0
"""

import os
import sys
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import deeprl_hw6
from controllers import calc_lqr_input
from ilqr import calc_ilqr_input


class Agent:
    def __init__(self, env_name="TwoLinkArm-v0", policy="LQR"):
        self.env = gym.make(env_name)
        self.sim_env = deepcopy(self.env)
        self.env_name = env_name
        self.algo = policy
        self.folder = os.path.join(env_name, policy)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def run_LQR(self, max_steps=2000):  # Added max_steps to prevent infinite loop
        rewards = []
        # gymnasium.reset() returns (obs, info)
        init = self.env.reset()
        init_state = init[0] if isinstance(init, (list, tuple)) else init
        states, actions = [init_state], []
        self.sim_env.reset()

        count = 1
        print(f"\nRunning LQR (Max Steps: {max_steps})...")

        while count <= max_steps:
            action = calc_lqr_input(self.env, self.sim_env)
            res = self.env.step(action)
            if isinstance(res, (list, tuple)) and len(res) == 5:
                state, reward, terminated, truncated, info = res
                done = terminated or truncated
            else:
                state, reward, done, info = res

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # --- FIX 1: Use %.2f to handle floats and avoid crash ---
            sys.stdout.write("\rSteps: %04d | Reward: %.2f\t" % (count, reward))
            sys.stdout.flush()

            if terminated or truncated:
                print(f"\nGoal Reached at step {count}!")
                break

            if done:
                break
            count += 1

        # Check if we ran out of steps
        if count > max_steps:
            print(
                f"\n[INFO] LQR stopped due to max_steps limit ({max_steps}). Proceeding..."
            )

        print("\nRewards Sum:", np.sum(rewards))
        print("Rewards Mean:", np.mean(rewards))

        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        }

        # Plot and save images for LQR as well
        self.plot(trajectory)

        return np.array(actions).T, states

    def run_iLQR(self):
        tN = 1000

        print("Initializing with LQR...")
        # Get an initial state from LQR (short run)
        U, LQR_X = self.run_LQR(max_steps=2000)

        # Ensure U matches tN size
        if U.shape[1] >= tN:
            U = U[:, :tN]
        else:
            padding = np.zeros((U.shape[0], tN - U.shape[1]))
            U = np.hstack((U, padding))

        # set underlying env state (unwrap wrappers)
        real_env = getattr(self.env, "unwrapped", self.env)
        real_env.state = LQR_X[0]

        # --- FIX: Initialize rewards list here ---
        states, actions = [LQR_X[0]], []
        rewards = []  # <--- THIS WAS MISSING
        # -----------------------------------------

        self.sim_env = deepcopy(self.env)

        count = 0

        print("\nStarting iLQR Optimization...")
        # Run the optimization
        U, costs = calc_ilqr_input(self.env, self.sim_env, U, tN=tN)

        print("\nExecuting Optimized Trajectory...")
        while count < tN:
            # Execute the optimized controls
            res = self.env.step(U[:, count])
            if isinstance(res, (list, tuple)) and len(res) == 5:
                state, reward, terminated, truncated, info = res
                done = terminated or truncated
            else:
                state, reward, done, info = res

            time.sleep(0.05)
            try:
                self.env.render()
            except Exception:
                pass

            states.append(state)
            rewards.append(reward)  # This will now work
            actions.append(U[:, count])

            sys.stdout.write("\rSteps: %04d | Reward: %.2f\t" % (count, reward))
            sys.stdout.flush()

            if done:
                print("\nGoal Reached!")
                break
            count += 1

        print("\nRewards Sum:", np.sum(rewards))
        print("Rewards Mean:", np.mean(rewards))

        trajectory = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        }

        self.plot(trajectory, costs)

    def run(self):
        if self.algo == "LQR":
            self.run_LQR()
        elif self.algo == "iLQR":
            print("Starting iLQR Pipeline")
            self.run_iLQR()
        else:
            print("Wrong Algorithm selected: {}".format(self.algo))

    def plot(self, trajectory, costs=None):
        total = len(trajectory["rewards"])
        plt.title(r"%s: Joint Angles (q)" % self.env_name)
        plt.plot(trajectory["states"][:, 0], label=r"$q_1$")
        plt.plot(trajectory["states"][:, 1], label=r"$q_2$")
        plt.xlabel("Steps (Total: %d)" % total)
        plt.ylabel("Joint Angles (rad)")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, "joint_angles.png"), dpi=300)

        plt.figure()
        plt.title(r"%s: Joint Velocities $(\dot{q})$" % self.env_name)
        plt.plot(trajectory["states"][:, 2], label=r"$\dot{q}_1$")
        plt.plot(trajectory["states"][:, 3], label=r"$\dot{q}_2$")
        plt.xlabel("Steps (Total: %d)" % total)
        plt.ylabel("Joint Velocities (rad/s)")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, "joint_velocities.png"), dpi=300)

        plt.figure()
        plt.title(r"%s: Control Inputs (u)" % self.env_name)
        # Handle case where actions might be shorter than states by 1
        u_steps = trajectory["actions"].shape[0]
        plt.plot(range(u_steps), trajectory["actions"][:, 0], label=r"$u_1$")
        plt.plot(range(u_steps), trajectory["actions"][:, 1], label=r"$u_2$")
        plt.xlabel("Steps (Total: %d)" % total)
        plt.ylabel("Control Inputs")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.folder, "control_inputs.png"), dpi=300)

        if costs is not None:
            plt.figure()
            plt.title(r"%s: Cost (u)" % self.env_name)
            plt.plot(costs[:], label=r"Cost")
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.folder, "cost.png"), dpi=300)


if __name__ == "__main__":
    agent = Agent(policy="iLQR")
    agent.run()
