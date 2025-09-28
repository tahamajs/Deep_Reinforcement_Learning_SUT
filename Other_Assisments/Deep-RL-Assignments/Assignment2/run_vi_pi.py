# Author: Taha Majlesi - 810101504, University of Tehran

import gymnasium as gym
from src.environment import env_wrapper
from src.policy_iteration import policy_iteration_sync, policy_iteration_async_ordered, policy_iteration_async_randperm
from src.value_iteration import value_iteration_sync, value_iteration_async_ordered, value_iteration_async_randperm

def main():
    # Create environment
    env = env_wrapper('Deterministic-4x4-FrozenLake-v0')
    gamma = 0.9

    print("Running Policy Iteration (Synchronous):")
    policy_pi_sync, value_func_pi_sync, policy_iters_pi_sync, value_iters_pi_sync = policy_iteration_sync(env, gamma)
    print(f"Policy Iters: {policy_iters_pi_sync}, Value Iters: {value_iters_pi_sync}")

    print("\nRunning Policy Iteration (Async Ordered):")
    policy_pi_async_ordered, value_func_pi_async_ordered, policy_iters_pi_async_ordered, value_iters_pi_async_ordered = policy_iteration_async_ordered(env, gamma)
    print(f"Policy Iters: {policy_iters_pi_async_ordered}, Value Iters: {value_iters_pi_async_ordered}")

    print("\nRunning Policy Iteration (Async Randperm):")
    policy_pi_async_randperm, value_func_pi_async_randperm, policy_iters_pi_async_randperm, value_iters_pi_async_randperm = policy_iteration_async_randperm(env, gamma)
    print(f"Policy Iters: {policy_iters_pi_async_randperm}, Value Iters: {value_iters_pi_async_randperm}")

    print("\nRunning Value Iteration (Synchronous):")
    value_func_vi_sync, iters_vi_sync = value_iteration_sync(env, gamma)
    print(f"Iters: {iters_vi_sync}")

    print("\nRunning Value Iteration (Async Ordered):")
    value_func_vi_async_ordered, iters_vi_async_ordered = value_iteration_async_ordered(env, gamma)
    print(f"Iters: {iters_vi_async_ordered}")

    print("\nRunning Value Iteration (Async Randperm):")
    value_func_vi_async_randperm, iters_vi_async_randperm = value_iteration_async_randperm(env, gamma)
    print(f"Iters: {iters_vi_async_randperm}")

if __name__ == "__main__":
    main()