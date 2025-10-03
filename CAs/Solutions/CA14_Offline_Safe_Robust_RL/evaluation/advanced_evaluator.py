import inspect
import numpy as np
import matplotlib.pyplot as plt


class ComprehensiveEvaluator:
    """Framework for evaluating advanced Deep RL methods across multiple dimensions."""

    def __init__(self):
        self.evaluation_metrics = {
            "sample_efficiency": [],
            "asymptotic_performance": [],
            "robustness_score": [],
            "safety_violations": [],
            "coordination_effectiveness": [],
            "computational_cost": [],
        }

        self.method_results = {}

    def evaluate_sample_efficiency(self, training_curves, convergence_threshold=0.8):
        """Evaluate how quickly methods reach target performance."""
        efficiency_scores = {}

        for method_name, rewards in training_curves.items():
            if not rewards:
                efficiency_scores[method_name] = float("inf")
                continue

            max_reward = max(rewards)
            target_reward = convergence_threshold * max_reward

            convergence_episode = len(rewards)  # Default to end
            for i, reward in enumerate(rewards):
                if reward >= target_reward:
                    convergence_episode = i
                    break

            efficiency_scores[method_name] = convergence_episode

        return efficiency_scores

    def evaluate_asymptotic_performance(self, training_curves, final_episodes=50):
        """Evaluate final performance after convergence."""
        asymptotic_scores = {}

        for method_name, rewards in training_curves.items():
            if len(rewards) >= final_episodes:
                asymptotic_scores[method_name] = np.mean(rewards[-final_episodes:])
            else:
                asymptotic_scores[method_name] = np.mean(rewards) if rewards else 0.0

        return asymptotic_scores

    def evaluate_robustness(self, agents, test_environments, num_episodes=50):
        """Evaluate robustness across different environments."""
        robustness_scores = {}

        for agent_name, agent in agents.items():
            environment_performances = []

            for env_name, env in test_environments.items():
                episode_rewards = []

                for episode in range(num_episodes):
                    obs = env.reset()
                    total_reward = 0
                    done = False

                    while not done:
                        if hasattr(agent, "get_action"):
                            act_out = agent.get_action(obs)
                            if isinstance(act_out, (list, tuple)):
                                action = act_out[0]
                            else:
                                action = act_out
                        else:
                            action = np.random.randint(getattr(env, "action_space", 4))

                        obs, reward, done, _ = env.step(action)
                        total_reward += reward

                    episode_rewards.append(total_reward)

                environment_performances.append(np.mean(episode_rewards))

            if environment_performances:
                min_perf = min(environment_performances)
                max_perf = max(environment_performances)
                robustness_scores[agent_name] = (
                    min_perf / max_perf if max_perf > 0 else 0.0
                )
            else:
                robustness_scores[agent_name] = 0.0

        return robustness_scores

    def evaluate_safety(self, agents, safe_environment, num_episodes=100):
        """Evaluate safety constraint satisfaction."""
        safety_scores = {}

        for agent_name, agent in agents.items():
            violations = 0
            total_steps = 0

            for episode in range(num_episodes):
                obs = safe_environment.reset()
                done = False

                while not done:
                    if hasattr(agent, "get_action"):
                        act_out = agent.get_action(obs)
                        if isinstance(act_out, (list, tuple)):
                            action = act_out[0]
                        else:
                            action = act_out
                    else:
                        action = np.random.randint(
                            getattr(safe_environment, "action_space", 4)
                        )

                    obs, reward, done, info = safe_environment.step(action)
                    total_steps += 1

                    if hasattr(safe_environment, "constraint_violation"):
                        if safe_environment.constraint_violation:
                            violations += 1
                    elif "constraint_violation" in info:
                        if info["constraint_violation"]:
                            violations += 1
                    elif (
                        reward < -1.0
                    ):  # Assume large negative reward indicates violation
                        violations += 1

            safety_scores[agent_name] = (
                violations / total_steps if total_steps > 0 else 1.0
            )

        return safety_scores

    def evaluate_coordination(self, multi_agent_results):
        """Evaluate multi-agent coordination effectiveness.

        Accepts different result shapes. If `cooperation_rate` exists, uses its
        final-episodes mean. Otherwise, falls back to coordination rewards if present.
        """
        coordination_scores = {}

        for method_name, results in multi_agent_results.items():
            score = 0.0
            if "cooperation_rate" in results and len(results["cooperation_rate"]) > 0:
                vals = results["cooperation_rate"]
                score = (
                    float(np.mean(vals[-50:]))
                    if len(vals) >= 50
                    else float(np.mean(vals))
                )
            elif (
                "coordination_rewards" in results
                and len(results["coordination_rewards"]) > 0
            ):
                individual_perf = results.get("individual_performance", 0.0)
                coordinated_perf = (
                    float(np.mean(results["coordination_rewards"][-50:]))
                    if len(results["coordination_rewards"]) >= 50
                    else float(np.mean(results["coordination_rewards"]))
                )
                score = coordinated_perf - float(individual_perf)

            coordination_scores[method_name] = score

        return coordination_scores

    def compute_comprehensive_score(self, method_results):
        """Compute overall score combining all metrics."""
        comprehensive_scores = {}

        metrics = [
            "sample_efficiency",
            "asymptotic_performance",
            "robustness_score",
            "safety_score",
            "coordination_effectiveness",
        ]

        normalized_scores = {}
        for metric in metrics:
            if metric in method_results:
                values = list(method_results[metric].values())
                if values:
                    if metric == "sample_efficiency":  # Lower is better
                        min_val, max_val = min(values), max(values)
                        normalized_scores[metric] = {
                            method: (
                                1 - (score - min_val) / (max_val - min_val)
                                if max_val > min_val
                                else 1.0
                            )
                            for method, score in method_results[metric].items()
                        }
                    elif metric == "safety_score":  # Lower is better (fewer violations)
                        min_val, max_val = min(values), max(values)
                        normalized_scores[metric] = {
                            method: (
                                1 - (score - min_val) / (max_val - min_val)
                                if max_val > min_val
                                else 1.0
                            )
                            for method, score in method_results[metric].items()
                        }
                    else:  # Higher is better
                        min_val, max_val = min(values), max(values)
                        normalized_scores[metric] = {
                            method: (
                                (score - min_val) / (max_val - min_val)
                                if max_val > min_val
                                else 1.0
                            )
                            for method, score in method_results[metric].items()
                        }

        weights = {
            "sample_efficiency": 0.2,
            "asymptotic_performance": 0.25,
            "robustness_score": 0.25,
            "safety_score": 0.2,
            "coordination_effectiveness": 0.1,
        }

        methods = set()
        for metric_scores in normalized_scores.values():
            methods.update(metric_scores.keys())

        for method in methods:
            score = 0
            weight_sum = 0

            for metric, weight in weights.items():
                if metric in normalized_scores and method in normalized_scores[metric]:
                    score += weight * normalized_scores[metric][method]
                    weight_sum += weight

            comprehensive_scores[method] = score / weight_sum if weight_sum > 0 else 0.0

        return comprehensive_scores, normalized_scores
