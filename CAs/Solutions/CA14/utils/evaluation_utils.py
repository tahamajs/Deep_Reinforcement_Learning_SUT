"""Utilities module for CA14 advanced RL implementations."""

import numpy as np
from robust_rl.environment import RobustEnvironment


def create_evaluation_environments():
    """Create diverse test environments for robustness evaluation."""
    environments = {
        "standard": RobustEnvironment(base_size=6, uncertainty_level=0.0),
        "noisy": RobustEnvironment(base_size=6, uncertainty_level=0.2),
        "large": RobustEnvironment(base_size=8, uncertainty_level=0.1),
        "obstacles": RobustEnvironment(
            base_size=6, uncertainty_level=0.1, dynamic_obstacles=True
        ),
    }
    return environments


def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all advanced RL methods."""
    print("🔍 Starting Comprehensive Evaluation of Advanced Deep RL Methods")
    print("=" * 70)

    from evaluation.advanced_evaluator import ComprehensiveEvaluator

    evaluator = ComprehensiveEvaluator()

    test_environments = create_evaluation_environments()

    all_agents = {}

    if "offline_agents" in globals():
        for name, agent in offline_agents.items():
            all_agents[f"Offline_{name}"] = agent

    if "safe_agents" in globals():
        for name, agent in safe_agents.items():
            all_agents[f"Safe_{name}"] = agent

    if "ma_agents" in globals():
        for name, agent_list in ma_agents.items():
            if isinstance(agent_list, list) and len(agent_list) > 0:
                all_agents[f"MultiAgent_{name}"] = agent_list[0]
            else:
                all_agents[f"MultiAgent_{name}"] = agent_list

    if "robust_agents" in globals():
        for name, agent in robust_agents.items():
            all_agents[f"Robust_{name}"] = agent

    training_curves = {}

    if "offline_results" in globals():
        for name in offline_results.keys():
            if "episode_rewards" in offline_results[name]:
                training_curves[f"Offline_{name}"] = offline_results[name][
                    "episode_rewards"
                ]

    if "safe_results" in globals():
        for name in safe_results.keys():
            if "rewards" in safe_results[name]:
                training_curves[f"Safe_{name}"] = safe_results[name]["rewards"]

    if "ma_results" in globals():
        for name in ma_results.keys():
            if "episode_rewards" in ma_results[name]:
                training_curves[f"MultiAgent_{name}"] = ma_results[name][
                    "episode_rewards"
                ]

    if "robust_results" in globals():
        for name in robust_results.keys():
            if (
                "rewards" in robust_results[name]
                and "low_uncertainty" in robust_results[name]["rewards"]
            ):
                training_curves[f"Robust_{name}"] = robust_results[name]["rewards"][
                    "low_uncertainty"
                ]

    print(
        f"📊 Evaluating {len(all_agents)} methods across {len(test_environments)} environments"
    )

    evaluation_results = {}

    print("⚡ Evaluating sample efficiency...")
    efficiency_scores = evaluator.evaluate_sample_efficiency(training_curves)
    evaluation_results["sample_efficiency"] = efficiency_scores

    print("🎯 Evaluating asymptotic performance...")
    asymptotic_scores = evaluator.evaluate_asymptotic_performance(training_curves)
    evaluation_results["asymptotic_performance"] = asymptotic_scores

    print("🛡️ Evaluating robustness...")
    robustness_scores = evaluator.evaluate_robustness(all_agents, test_environments)
    evaluation_results["robustness_score"] = robustness_scores

    print("🚨 Evaluating safety...")
    if "safe_envs" in globals():
        safe_env = (
            list(safe_envs.values())[0] if safe_envs else test_environments["standard"]
        )
    else:
        safe_env = test_environments["standard"]
    safety_scores = evaluator.evaluate_safety(all_agents, safe_env)
    evaluation_results["safety_score"] = safety_scores

    print("🤝 Evaluating coordination...")
    coordination_scores = {}
    if "ma_results" in globals():
        coordination_scores = evaluator.evaluate_coordination(ma_results)
    evaluation_results["coordination_effectiveness"] = coordination_scores

    print("📈 Computing comprehensive scores...")
    comprehensive_scores, normalized_scores = evaluator.compute_comprehensive_score(
        evaluation_results
    )

    return evaluation_results, comprehensive_scores, normalized_scores


__all__ = ["create_evaluation_environments", "run_comprehensive_evaluation"]
