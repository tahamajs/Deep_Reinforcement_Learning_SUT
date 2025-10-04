# CA5 Advanced DQN Methods

این پروژه شامل پیاده‌سازی کامل روش‌های پیشرفته Deep Q-Network (DQN) است که شامل Double DQN، Dueling DQN، Prioritized Experience Replay و Rainbow DQN می‌باشد.

## ویژگی‌های پروژه

### 🤖 Agent های پیاده‌سازی شده

- **Vanilla DQN**: پیاده‌سازی پایه DQN
- **Double DQN**: حل مشکل overestimation bias
- **Dueling DQN**: جداسازی value و advantage streams
- **Prioritized DQN**: استفاده از Prioritized Experience Replay
- **Rainbow DQN**: ترکیب تمام بهبودها

### 🌍 Environment های سفارشی

- **GridWorld**: محیط ساده برای تست الگوریتم‌ها
- **MountainCarContinuous**: محیط Mountain Car پیوسته
- **LunarLander**: محیط Lunar Lander با reward shaping

### 🛠️ ابزارهای کمکی

- **Replay Buffer**: Experience replay معمولی و اولویت‌دار
- **Network Architectures**: معماری‌های مختلف شبکه عصبی
- **Training Analysis**: ابزارهای تحلیل آموزش
- **Performance Evaluation**: ارزیابی عملکرد agent ها

## نصب و راه‌اندازی

### پیش‌نیازها

```bash
Python 3.8+
PyTorch 1.9+
Gym/Gymnasium
```

### نصب

```bash
# کلون کردن پروژه
git clone <repository-url>
cd CA05_Advanced_DQN_Methods

# نصب dependencies
pip install -r requirements.txt

# اجرای کامل پروژه
./run.sh
```

## استفاده سریع

### آموزش یک Agent

```python
from agents import DQNAgent
from training_examples import train_dqn_agent

# آموزش Vanilla DQN
results = train_dqn_agent(
    env_name='CartPole-v1',
    agent_type='dqn',
    num_episodes=1000
)
```

### مقایسه Agent های مختلف

```python
from training_examples import dqn_variant_comparison

# مقایسه تمام variant ها
comparison_results = dqn_variant_comparison()
```

### اجرای آزمایشات

```python
from experiments import ExperimentRunner, get_dqn_configs
from agents import DQNAgent, DoubleDQNAgent

runner = ExperimentRunner()
configs = get_dqn_configs()
agent_classes = [DQNAgent, DoubleDQNAgent]

results = runner.run_comparison_experiment(
    configs, agent_classes, 'CartPole-v1'
)
```

## ساختار پروژه

```
CA05_Advanced_DQN_Methods/
├── agents/                 # پیاده‌سازی agent ها
│   ├── dqn_base.py        # DQN پایه
│   ├── double_dqn.py      # Double DQN
│   ├── dueling_dqn.py     # Dueling DQN
│   ├── prioritized_replay.py # Prioritized DQN
│   └── rainbow_dqn.py     # Rainbow DQN
├── environments/          # محیط‌های سفارشی
│   └── custom_envs.py     # تعریف محیط‌ها
├── utils/                # ابزارهای کمکی
│   ├── advanced_dqn_extensions.py
│   ├── network_architectures.py
│   ├── training_analysis.py
│   ├── analysis_tools.py
│   ├── ca5_helpers.py
│   └── ca5_main.py
├── experiments/          # تنظیمات آزمایشات
│   └── __init__.py       # Experiment runner
├── evaluation/           # ارزیابی عملکرد
│   └── __init__.py       # Performance evaluator
├── visualizations/       # نمودارها و تصاویر
├── models/              # مدل‌های ذخیره شده
├── results/             # نتایج آزمایشات
├── training_examples.py # مثال‌های آموزش
├── CA5.ipynb           # Jupyter notebook
├── run.sh              # اسکریپت اجرای کامل
└── requirements.txt    # Dependencies
```

## اجرای کامل پروژه

برای اجرای کامل تمام کامپوننت‌ها:

```bash
./run.sh
```

این اسکریپت:

1. ✅ محیط مجازی ایجاد می‌کند
2. ✅ Dependencies نصب می‌کند
3. ✅ محیط‌های سفارشی تست می‌کند
4. ✅ Agent ها تست می‌کند
5. ✅ مثال‌های آموزش اجرا می‌کند
6. ✅ آزمایشات مقایسه‌ای انجام می‌دهد
7. ✅ ارزیابی عملکرد انجام می‌دهد
8. ✅ نمودارها و تصاویر تولید می‌کند
9. ✅ گزارش خلاصه ایجاد می‌کند

## نتایج و خروجی‌ها

پس از اجرای کامل، نتایج در پوشه‌های زیر ذخیره می‌شوند:

- **visualizations/**: نمودارها و تصاویر

  - `q_value_landscape.png`: نقشه Q-values
  - `replay_analysis.png`: تحلیل experience replay
  - `agent_comparison.png`: مقایسه agent ها

- **results/**: نتایج آموزش و ارزیابی

  - `training_results.json`: نتایج آموزش
  - `summary_report.json`: گزارش خلاصه

- **experiments/**: نتایج آزمایشات
  - `comparison_results.json`: نتایج مقایسه‌ای

## مثال‌های استفاده

### آموزش Double DQN

```python
from agents import DoubleDQNAgent
import gym

env = gym.make('CartPole-v1')
agent = DoubleDQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,
    gamma=0.99
)

# آموزش
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > agent.batch_size:
            agent.update()
        state = next_state
```

### استفاده از Prioritized Experience Replay

```python
from agents import PrioritizedDQNAgent

agent = PrioritizedDQNAgent(
    state_dim=4,
    action_dim=2,
    alpha=0.6,  # اولویت‌دهی
    beta=0.4   # تصحیح bias
)
```

## تنظیمات پیشرفته

### پارامترهای مهم

- **Learning Rate**: `lr=1e-3` (پیش‌فرض)
- **Discount Factor**: `gamma=0.99`
- **Epsilon Decay**: `epsilon_decay=0.995`
- **Buffer Size**: `buffer_size=10000`
- **Batch Size**: `batch_size=32`

### محیط‌های پشتیبانی شده

- `CartPole-v1`: محیط کلاسیک
- `MountainCar-v0`: محیط چالش‌برانگیز
- `LunarLander-v2`: محیط پیچیده
- `GridWorld`: محیط سفارشی

## مشارکت

برای مشارکت در پروژه:

1. Fork کنید
2. Branch جدید ایجاد کنید
3. تغییرات را commit کنید
4. Pull request ارسال کنید

## مجوز

این پروژه تحت مجوز MIT منتشر شده است.

## تماس

برای سوالات و پشتیبانی، با تیم توسعه تماس بگیرید.

---

**نکته**: این پروژه برای اهداف آموزشی و تحقیقاتی طراحی شده است. برای استفاده در محیط‌های production، تست‌های بیشتری مورد نیاز است.
