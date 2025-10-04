# CA9: Advanced Policy Gradient Methods - راهنمای کامل اجرا

## 📋 فهرست مطالب

- [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
- [اسکریپت‌های اجرا](#اسکریپت‌های-اجرا)
- [اجرای کامل](#اجرای-کامل)
- [اجرای سریع](#اجرای-سریع)
- [اجرای جداگانه](#اجرای-جداگانه)
- [نتایج و خروجی‌ها](#نتایج-و-خروجی‌ها)
- [عیب‌یابی](#عیب‌یابی)

## 🚀 نصب و راه‌اندازی

### 1. فعال‌سازی Virtual Environment

```bash
cd /Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA09_Advanced_Policy_Gradients
python3 -m venv venv
source venv/bin/activate
```

### 2. نصب Dependencies

```bash
pip install -r requirements.txt
```

### 3. تست Setup

```bash
python3 test_setup.py
```

## 📜 اسکریپت‌های اجرا

### 1. `run.sh` - اسکریپت Bash کامل

```bash
chmod +x run.sh
./run.sh
```

**ویژگی‌ها:**

- اجرای کامل همه agent ها
- تولید همه visualization ها
- اجرای training examples
- ایجاد گزارش کامل

### 2. `final_run.py` - اسکریپت Python نهایی

```bash
python3 final_run.py
```

**ویژگی‌ها:**

- اجرای REINFORCE و Baseline REINFORCE
- تولید visualization های پیشرفته
- ایجاد گزارش کامل

### 3. `quick_run.py` - اسکریپت سریع

```bash
python3 quick_run.py
```

**ویژگی‌ها:**

- اجرای سریع REINFORCE
- تولید visualization های اصلی
- مناسب برای تست سریع

### 4. `run_all.py` - اسکریپت کامل Python

```bash
python3 run_all.py
```

**ویژگی‌ها:**

- اجرای همه algorithm ها
- تولید همه visualization ها
- hyperparameter tuning

## 🎯 اجرای کامل

### روش 1: اسکریپت Bash

```bash
# فعال‌سازی environment
source venv/bin/activate

# اجرای کامل
./run.sh
```

### روش 2: اسکریپت Python

```bash
# فعال‌سازی environment
source venv/bin/activate

# اجرای کامل
python3 final_run.py
```

## ⚡ اجرای سریع

برای تست سریع و تولید visualization های اصلی:

```bash
source venv/bin/activate
python3 quick_run.py
```

## 🔧 اجرای جداگانه

### 1. اجرای Agent های جداگانه

#### REINFORCE

```python
from agents.reinforce import REINFORCEAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = REINFORCEAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# آموزش
for episode in range(100):
    state, _ = env.reset()
    episode_rewards = []

    for step in range(200):
        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.store_reward(reward)
        episode_rewards.append(reward)
        state = next_state

        if terminated or truncated:
            break

    agent.update_policy()
    print(f"Episode {episode + 1}: Reward = {sum(episode_rewards)}")

env.close()
```

#### Baseline REINFORCE

```python
from agents.baseline_reinforce import BaselineREINFORCEAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = BaselineREINFORCEAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# آموزش مشابه REINFORCE
```

#### Actor-Critic

```python
from agents.actor_critic import ActorCriticAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = ActorCriticAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# آموزش Actor-Critic
```

#### PPO

```python
from agents.ppo import PPOAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# آموزش PPO
```

#### Continuous Control

```python
from agents.continuous_control import ContinuousPPOAgent
import gymnasium as gym

env = gym.make("Pendulum-v1")
agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# آموزش برای کنترل پیوسته
```

### 2. اجرای Visualization ها

#### Policy Gradient Visualizer

```python
from utils.policy_gradient_visualizer import PolicyGradientVisualizer

visualizer = PolicyGradientVisualizer()

# ایجاد visualization های اصلی
results = visualizer.demonstrate_policy_gradient_intuition()
visualizer.compare_value_vs_policy_methods()
visualizer.create_advanced_visualizations()
```

#### Training Examples

```python
from training_examples import (
    plot_policy_gradient_convergence_analysis,
    comprehensive_policy_gradient_comparison,
    policy_gradient_curriculum_learning,
    entropy_regularization_study,
    create_comprehensive_visualization_suite
)

# تحلیل همگرایی
fig1 = plot_policy_gradient_convergence_analysis("convergence_analysis.png")

# مقایسه جامع
results = comprehensive_policy_gradient_comparison("comprehensive_comparison.png")

# Curriculum Learning
curriculum_results = policy_gradient_curriculum_learning("curriculum_learning.png")

# Entropy Regularization
entropy_results = entropy_regularization_study("entropy_regularization.png")

# مجموعه کامل visualization ها
create_comprehensive_visualization_suite("visualizations/")
```

### 3. Hyperparameter Tuning

```python
from utils.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner("CartPole-v1")

# تنظیم learning rates
lr_results = tuner.tune_learning_rates()

# تنظیم پارامترهای PPO
ppo_results = tuner.tune_ppo_parameters()
```

## 📊 نتایج و خروجی‌ها

### پوشه‌های خروجی

#### `visualizations/`

- `policy_gradient_intuition.png` - درک policy gradient
- `value_vs_policy_comparison.png` - مقایسه value-based vs policy-based
- `convergence_analysis.png` - تحلیل همگرایی
- `comprehensive_comparison.png` - مقایسه جامع روش‌ها
- `curriculum_learning.png` - تحلیل curriculum learning
- `entropy_regularization.png` - مطالعه entropy regularization
- فایل‌های PDF با کیفیت بالا

#### `results/`

- `final_execution_report.md` - گزارش کامل اجرا
- `quick_execution_report.md` - گزارش اجرای سریع
- آمار عملکرد و نتایج

#### `logs/`

- لاگ‌های اجرای هر component
- اطلاعات debug و error ها

### نتایج کلیدی

#### Performance Metrics

- **REINFORCE**: میانگین reward آخرین 10 episode
- **Baseline REINFORCE**: بهبود variance و stability
- **Actor-Critic**: همگرایی سریع‌تر
- **PPO**: عملکرد state-of-the-art
- **Continuous Control**: کنترل پیوسته موفق

#### Visualizations

- نمودارهای همگرایی
- مقایسه روش‌ها
- تحلیل variance
- نمودارهای 3D و انیمیشن

## 🔍 عیب‌یابی

### مشکلات رایج

#### 1. خطای Import

```bash
ModuleNotFoundError: No module named 'torch'
```

**راه حل:**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. خطای IndentationError

```bash
IndentationError: expected an indented block
```

**راه حل:** فایل‌های Python را بررسی کنید و indentation را اصلاح کنید.

#### 3. خطای Environment

```bash
Environment test failed
```

**راه حل:**

```bash
pip install gymnasium
```

#### 4. خطای Visualization

```bash
Matplotlib backend error
```

**راه حل:**

```bash
pip install matplotlib
export MPLBACKEND=Agg  # برای server environments
```

### Debug Mode

برای اجرای debug:

```bash
python3 -c "
import sys
sys.path.append('.')
from test_setup import main
main()
"
```

## 📈 Performance Tips

### 1. GPU Support

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Memory Management

```python
torch.cuda.empty_cache()  # پاک کردن GPU memory
```

### 3. Parallel Training

```python
# برای environments متعدد
from multiprocessing import Pool
```

## 🎯 Next Steps

### 1. Experimentation

- تست روی environments مختلف
- تنظیم hyperparameters
- مقایسه performance

### 2. Advanced Features

- Multi-agent training
- Hierarchical policies
- Meta-learning

### 3. Research Extensions

- Offline RL
- Safe RL
- Robust RL

## 📞 Support

برای مشکلات و سوالات:

1. بررسی logs در پوشه `logs/`
2. اجرای `test_setup.py` برای diagnostic
3. بررسی documentation در فایل‌های markdown

---

**نکته:** همه اسکریپت‌ها برای اجرای کامل طراحی شده‌اند و نتایج را در پوشه‌های مناسب ذخیره می‌کنند. برای بهترین نتیجه، ابتدا `test_setup.py` را اجرا کنید تا مطمئن شوید همه چیز درست کار می‌کند.

