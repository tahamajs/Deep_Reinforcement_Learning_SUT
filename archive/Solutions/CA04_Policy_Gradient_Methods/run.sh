#!/bin/bash

# CA4: Advanced Policy Gradient Methods - Complete Run Script
# این اسکریپت تمام فایل‌های پروژه پیشرفته را اجرا می‌کند و نتایج را در فولدر visualizations ذخیره می‌کند

echo "=========================================="
echo "🚀 CA4: Advanced Policy Gradient Methods"
echo "=========================================="

# تنظیم متغیرهای محیطی
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MPLBACKEND="Agg"  # برای ذخیره نمودارها بدون نمایش

# ایجاد فولدرهای مورد نیاز
echo "📁 ایجاد فولدرهای مورد نیاز..."
mkdir -p visualizations
mkdir -p evaluation/results
mkdir -p models/saved_models
mkdir -p logs
mkdir -p experiments/advanced
mkdir -p benchmarks/results
mkdir -p analysis/reports

# نصب وابستگی‌ها
echo "📦 نصب وابستگی‌ها..."
if [ ! -d "venv" ]; then
    echo "ایجاد محیط مجازی..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# نصب وابستگی‌های اضافی برای ویژگی‌های پیشرفته
echo "📦 نصب وابستگی‌های پیشرفته..."
pip install plotly networkx scikit-learn opencv-python

# بررسی وجود فایل‌های مورد نیاز
echo "🔍 بررسی فایل‌های پروژه..."
if [ ! -f "CA4.ipynb" ]; then
    echo "❌ فایل CA4.ipynb یافت نشد!"
    exit 1
fi

if [ ! -f "training_examples.py" ]; then
    echo "❌ فایل training_examples.py یافت نشد!"
    exit 1
fi

# اجرای تست‌های سریع
echo "🧪 اجرای تست‌های سریع..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import run_quick_test
print('تست REINFORCE...')
run_quick_test('CartPole-v1', 'reinforce', 50)
print('تست Actor-Critic...')
run_quick_test('CartPole-v1', 'actor_critic', 50)
print('✅ تست‌های سریع تکمیل شدند')
"

# اجرای تست‌های الگوریتم‌های پیشرفته
echo "🔬 اجرای تست‌های الگوریتم‌های پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from agents.advanced_algorithms import TRPOAgent, SACAgent, DDPGAgent
from environments.advanced_environments import CustomMountainCarEnv, CustomPendulumEnv
import numpy as np

print('🧪 تست TRPO...')
env = CustomMountainCarEnv()
agent = TRPOAgent(env.observation_space.shape[0], env.action_space.n)
results = agent.train(env, num_episodes=100, print_every=50)
print(f'TRPO - میانگین امتیاز نهایی: {np.mean(results[\"scores\"][-10:]):.2f}')

print('🧪 تست SAC...')
env = CustomPendulumEnv()
agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0])
results = agent.train(env, num_episodes=100, print_every=50)
print(f'SAC - میانگین امتیاز نهایی: {np.mean(results[\"scores\"][-10:]):.2f}')

print('🧪 تست DDPG...')
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0])
results = agent.train(env, num_episodes=100, print_every=50)
print(f'DDPG - میانگین امتیاز نهایی: {np.mean(results[\"scores\"][-10:]):.2f}')

print('✅ تست‌های الگوریتم‌های پیشرفته تکمیل شدند')
"

# اجرای تست‌های شبکه‌های عصبی پیشرفته
echo "🧠 اجرای تست‌های شبکه‌های عصبی پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from agents.advanced_networks import (
    CNNPolicyNetwork, LSTMPolicyNetwork, TransformerPolicyNetwork,
    DeepResidualPolicyNetwork, AttentionPolicyNetwork, EnsemblePolicyNetwork
)
import torch
import numpy as np

print('🧠 تست CNN Policy Network...')
cnn_net = CNNPolicyNetwork(input_channels=3, action_size=4)
test_input = torch.randn(1, 3, 84, 84)
output = cnn_net(test_input)
print(f'CNN Output shape: {output.shape}')

print('🧠 تست LSTM Policy Network...')
lstm_net = LSTMPolicyNetwork(state_size=4, action_size=2)
test_input = torch.randn(1, 4)
output = lstm_net(test_input)
print(f'LSTM Output shape: {output.shape}')

print('🧠 تست Transformer Policy Network...')
transformer_net = TransformerPolicyNetwork(state_size=4, action_size=2)
test_input = torch.randn(1, 4)
output = transformer_net(test_input)
print(f'Transformer Output shape: {output.shape}')

print('🧠 تست Deep Residual Policy Network...')
residual_net = DeepResidualPolicyNetwork(state_size=4, action_size=2)
test_input = torch.randn(1, 4)
output = residual_net(test_input)
print(f'Residual Output shape: {output.shape}')

print('🧠 تست Attention Policy Network...')
attention_net = AttentionPolicyNetwork(state_size=4, action_size=2)
test_input = torch.randn(1, 4)
output = attention_net(test_input)
print(f'Attention Output shape: {output.shape}')

print('🧠 تست Ensemble Policy Network...')
ensemble_net = EnsemblePolicyNetwork(state_size=4, action_size=2)
test_input = torch.randn(1, 4)
output = ensemble_net(test_input)
print(f'Ensemble Output shape: {output.shape}')

print('✅ تست‌های شبکه‌های عصبی پیشرفته تکمیل شدند')
"

# اجرای تست‌های سیستم‌های Multi-Agent
echo "👥 اجرای تست‌های سیستم‌های Multi-Agent..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from agents.multi_agent_systems import MultiAgentPolicyGradient, MetaLearningAgent
from environments.advanced_environments import MultiAgentWrapper
import gymnasium as gym

print('👥 تست Multi-Agent System...')
env = gym.make('CartPole-v1')
env = MultiAgentWrapper(env, num_agents=2)
multi_agent = MultiAgentPolicyGradient(num_agents=2, state_size=4, action_size=2)
multi_agent.train(env, num_episodes=50, print_every=25)

print('🧠 تست Meta-Learning Agent...')
meta_agent = MetaLearningAgent(state_size=4, action_size=2)
# Simulate task adaptation
task_experiences = [{'state': [0.1, 0.2, 0.3, 0.4], 'action': 0, 'reward': 1.0, 'next_state': [0.2, 0.3, 0.4, 0.5], 'done': False}]
adapted_network = meta_agent.adapt_to_task('task_1', task_experiences)
print('Meta-learning adaptation completed')

print('✅ تست‌های سیستم‌های Multi-Agent تکمیل شدند')
"

# اجرای تست‌های محیط‌های پیشرفته
echo "🌍 اجرای تست‌های محیط‌های پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from environments.advanced_environments import (
    AtariWrapper, CurriculumWrapper, NoisyObservationWrapper, RewardShapingWrapper
)
import gymnasium as gym

print('🌍 تست Atari Wrapper...')
env = gym.make('Breakout-v4')
env = AtariWrapper(env)
obs, info = env.reset()
print(f'Atari observation shape: {obs.shape}')

print('🌍 تست Curriculum Wrapper...')
env = gym.make('CartPole-v1')
env = CurriculumWrapper(env)
obs, info = env.reset()
print(f'Curriculum level: {info[\"level_name\"]}')

print('🌍 تست Noisy Observation Wrapper...')
env = gym.make('CartPole-v1')
env = NoisyObservationWrapper(env, noise_std=0.1)
obs, info = env.reset()
print(f'Noisy observation shape: {obs.shape}')

print('🌍 تست Reward Shaping Wrapper...')
def shaping_func(prev_state, action, next_state, reward):
    return 0.1 * (next_state[0] - prev_state[0])  # Position-based shaping
env = gym.make('CartPole-v1')
env = RewardShapingWrapper(env, shaping_function=shaping_func)
obs, info = env.reset()
print('Reward shaping wrapper initialized')

print('✅ تست‌های محیط‌های پیشرفته تکمیل شدند')
"

# اجرای تست‌های تجسم پیشرفته
echo "📊 اجرای تست‌های تجسم پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from utils.advanced_visualization import AdvancedPolicyVisualizer, AdvancedTrainingVisualizer
import matplotlib.pyplot as plt
import numpy as np

print('📊 تست Advanced Policy Visualizer...')
viz = AdvancedPolicyVisualizer()

# Create dummy policy network for testing
import torch.nn as nn
policy_net = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

state_space = np.array([[-2, 2], [-2, 2]])
action_space = np.array([0, 1])

print('Creating policy landscape visualization...')
viz.plot_policy_landscape(policy_net, state_space, action_space)

print('📊 تست Advanced Training Visualizer...')
train_viz = AdvancedTrainingVisualizer()

# Create dummy training metrics
metrics = {
    'scores': np.random.randn(1000).cumsum(),
    'policy_losses': np.random.randn(1000),
    'value_losses': np.random.randn(1000),
    'entropy_losses': np.random.randn(1000)
}

print('Creating training metrics visualization...')
train_viz.plot_training_metrics(metrics)

print('✅ تست‌های تجسم پیشرفته تکمیل شدند')
"

# اجرای مثال‌های آموزشی
echo "📚 اجرای مثال‌های آموزشی..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from training_examples import (
    hyperparameter_sensitivity_study,
    curriculum_learning_example,
    performance_comparison_study,
    train_with_monitoring
)

print('🔬 مطالعه حساسیت هایپرپارامترها...')
hp_results = hyperparameter_sensitivity_study()
print(f'نتایج مطالعه هایپرپارامترها: {len(hp_results)} رکورد')

print('📈 مثال یادگیری تدریجی...')
curriculum_results = curriculum_learning_example()
print(f'نتایج یادگیری تدریجی: {curriculum_results[\"final_performance\"]:.2f}')

print('⚖️ مطالعه مقایسه عملکرد...')
comparison_results = performance_comparison_study()
print(f'بهترین الگوریتم: {comparison_results[\"best_algorithm\"]}')

print('📊 آموزش با مانیتورینگ...')
monitoring_results = train_with_monitoring('CartPole-v1', 200)
print(f'میانگین نهایی: {sum(monitoring_results[\"scores\"][-10:])/10:.2f}')
"

# اجرای آزمایش‌های کامل پیشرفته
echo "🔬 اجرای آزمایش‌های کامل پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import PolicyGradientExperiment, BenchmarkSuite
from agents.advanced_algorithms import TRPOAgent, SACAgent, DDPGAgent
from agents.advanced_networks import create_advanced_policy_network
from environments.advanced_environments import create_advanced_environment
from utils.advanced_visualization import AdvancedTrainingVisualizer, AdvancedAnalysisTools
import matplotlib.pyplot as plt
import numpy as np
import pickle

print('🔄 آزمایش مقایسه الگوریتم‌های پیشرفته...')

# Create advanced environments
envs = {
    'CartPole-v1': create_advanced_environment('CartPole-v1'),
    'CustomMountainCar': create_advanced_environment('CustomMountainCar'),
    'CustomPendulum': create_advanced_environment('CustomPendulum')
}

# Test different algorithms
algorithms = {
    'REINFORCE': 'reinforce',
    'Actor-Critic': 'actor_critic',
    'TRPO': 'trpo',
    'SAC': 'sac',
    'DDPG': 'ddpg'
}

results = {}

for env_name, env in envs.items():
    print(f'Testing on {env_name}...')
    env_results = {}
    
    for alg_name, alg_type in algorithms.items():
        print(f'  Testing {alg_name}...')
        
        if alg_type == 'trpo':
            agent = TRPOAgent(env.observation_space.shape[0], env.action_space.n)
        elif alg_type == 'sac':
            agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0])
        elif alg_type == 'ddpg':
            agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0])
        else:
            # Use basic algorithms for comparison
            experiment = PolicyGradientExperiment(env_name)
            env_results[alg_name] = experiment.run_single_algorithm(alg_type, num_episodes=200)
            continue
        
        # Train advanced algorithm
        train_results = agent.train(env, num_episodes=200, print_every=50)
        env_results[alg_name] = train_results
    
    results[env_name] = env_results

# Save comprehensive results
with open('evaluation/results/advanced_comparison_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print('📊 ایجاد نمودارهای پیشرفته...')
viz = AdvancedTrainingVisualizer()

# Create comprehensive comparison plots
for env_name, env_results in results.items():
    if len(env_results) > 1:
        viz.plot_learning_curves_comparison(env_results, window_size=50)
        plt.savefig(f'visualizations/{env_name}_advanced_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        viz.plot_performance_distribution(env_results)
        plt.savefig(f'visualizations/{env_name}_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

print('✅ آزمایش‌های کامل پیشرفته تکمیل شدند')
"

# اجرای تحلیل‌های پیشرفته
echo "📈 اجرای تحلیل‌های پیشرفته..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from utils.advanced_visualization import AdvancedAnalysisTools
import pickle
import numpy as np

print('📈 تحلیل‌های پیشرفته...')
analysis_tools = AdvancedAnalysisTools()

# Load results
with open('evaluation/results/advanced_comparison_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Generate comprehensive analysis report
report = 'analysis/reports/comprehensive_analysis_report.txt'
with open(report, 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('COMPREHENSIVE POLICY GRADIENT ANALYSIS REPORT\n')
    f.write('=' * 80 + '\n\n')
    
    for env_name, env_results in results.items():
        f.write(f'ENVIRONMENT: {env_name}\n')
        f.write('-' * 40 + '\n')
        
        for alg_name, alg_results in env_results.items():
            f.write(f'\\nAlgorithm: {alg_name}\n')
            
            if 'scores' in alg_results:
                scores = alg_results['scores']
                conv_analysis = analysis_tools.analyze_policy_convergence(scores)
                
                f.write(f'  Final Performance: {np.mean(scores[-100:]):.2f} ± {np.std(scores[-100:]):.2f}\n')
                f.write(f'  Best Performance: {np.max(scores):.2f}\n')
                f.write(f'  Convergence: {conv_analysis[\"converged\"]}\n')
                f.write(f'  Stability: {conv_analysis[\"stability\"]:.4f}\n')
                
                if conv_analysis['convergence_episode']:
                    f.write(f'  Convergence Episode: {conv_analysis[\"convergence_episode\"]}\n')
        
        f.write('\\n' + '=' * 40 + '\\n')

print(f'📊 گزارش تحلیل در {report} ذخیره شد')
print('✅ تحلیل‌های پیشرفته تکمیل شدند')
"

# ایجاد گزارش نهایی
echo "📋 ایجاد گزارش نهایی..."
source venv/bin/activate && python3 -c "
import os
import glob
import json
from datetime import datetime

# Collect all results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'visualizations': glob.glob('visualizations/*.png'),
    'results': glob.glob('evaluation/results/*.pkl'),
    'reports': glob.glob('analysis/reports/*.txt'),
    'logs': glob.glob('logs/*.log')
}

# Save summary
with open('evaluation/results/execution_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print('📋 خلاصه اجرا:')
print(f'  - تعداد نمودارها: {len(results_summary[\"visualizations\"])}')
print(f'  - تعداد نتایج: {len(results_summary[\"results\"])}')
print(f'  - تعداد گزارش‌ها: {len(results_summary[\"reports\"])}')
print(f'  - خلاصه در: evaluation/results/execution_summary.json')
"

echo ""
echo "🎉 تمام فایل‌های پروژه پیشرفته اجرا شدند!"
echo "📊 نتایج در فولدر visualizations ذخیره شدند"
echo "📈 گزارش‌های تحلیل در فولدر analysis/reports موجودند"
echo "💾 نتایج کامل در فولدر evaluation/results ذخیره شدند"
echo ""
echo "🚀 پروژه Policy Gradient Methods با موفقیت تکمیل شد!"
echo "=========================================="