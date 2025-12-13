import numpy as np

class GridWorldConfig:
    SIZE = 4
    GOAL_REWARD = 10
    STEP_REWARD = -0.1
    OBSTACLE_REWARD = -5
    OBSTACLES = [(1, 1), (2, 1), (1, 2)]
    START_STATE = (0, 0)
    GOAL_STATE = (3, 3)

class PolicyIterationConfig:
    GAMMA = 0.9
    THETA = 1e-6
    MAX_ITERATIONS = 100

class ValueIterationConfig:
    GAMMA = 0.9
    THETA = 1e-6
    MAX_ITERATIONS = 100

class QLearningConfig:
    NUM_EPISODES = 1000
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.1

class ExperimentConfig:
    DISCOUNT_FACTORS = [0.1, 0.5, 0.9, 0.99]
    POLICY_COMPARISON_GAMMA = 0.9
    Q_LEARNING_EVAL_EPISODES = 1000
    Q_LEARNING_EVAL_WINDOW = 50
    ENVIRONMENT_MODIFICATIONS = [
        {"obstacles": [(1, 1), (2, 1), (1, 2)], "name": "Standard"},
        {"obstacles": [(1, 1)], "name": "Easy (Few Obstacles)"},
        {
            "obstacles": [(1, 1), (2, 1), (1, 2), (2, 2)],
            "name": "Hard (Many Obstacles)",
        },
        {"obstacles": [], "name": "No Obstacles"},
    ]

class VisualizationConfig:
    FIGURE_SIZE = (12, 8)
    FONT_SIZE = 12
    PLOT_DPI = 300
    SAVE_PATH = "../pictures/"
    # Define specific plot filenames
    VALUE_FUNCTION_PLOT = "value_function.png"
    POLICY_PLOT = "policy.png"
    Q_VALUES_PLOT = "q_values.png"
    LEARNING_CURVE_PLOT = "learning_curve.png"
    VALUE_ITERATION_CONVERGENCE_PLOT = "value_iteration_convergence.png"
    POLICY_COMPARISON_PLOT = "policy_comparison.png"
    ALGORITHM_CONVERGENCE_PLOT = "algorithm_convergence.png"
    DISCOUNT_FACTOR_EFFECT_PLOT = "discount_factor_effect.png"
    ENVIRONMENT_MODIFICATION_PLOT = "environment_modification.png"

# Global settings
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

