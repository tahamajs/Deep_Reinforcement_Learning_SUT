# 🔍 راهنمای Debugging برای HW3

این راهنما شامل تمام تغییرات debugging اضافه شده به فایل‌های HW3 است.

## 📁 فایل‌های تغییر یافته

### 1. `src/actor_critic.py`

**تغییرات:**

- ✅ اضافه شدن logging system با فایل `actor_critic_debug.log`
- ✅ Debugging برای ساخت MLP networks
- ✅ Logging جزئیات initialization
- ✅ Debugging برای sampling trajectories
- ✅ Logging آمار critic و actor updates
- ✅ نمایش loss قبل و بعد از training

**اطلاعات قابل مشاهده:**

- 🏗️ ساختار شبکه‌ها
- 📊 آمار trajectories (rewards, steps)
- 🔄 Critic loss changes
- 🎭 Actor advantage statistics

### 2. `src/dqn.py`

**تغییرات:**

- ✅ اضافه شدن logging system با فایل `dqn_debug.log`
- ✅ Debugging برای ساخت Q-networks (CNN/MLP)
- ✅ Logging جزئیات agent initialization
- ✅ Debugging action selection (random vs greedy)
- ✅ Logging training progress و loss changes
- ✅ نمایش target network updates

**اطلاعات قابل مشاهده:**

- 🏗️ ساختار Q-networks
- 🎲 Action selection statistics
- 📉 Training loss progression
- 🎯 Target network updates

### 3. `run_ac.py`

**تغییرات:**

- ✅ اضافه شدن logging system با فایل `run_ac_debug.log`
- ✅ Logging جزئیات training setup
- ✅ Debugging برای هر iteration
- ✅ نمایش آمار paths و returns
- ✅ Logging performance metrics

**اطلاعات قابل مشاهده:**

- 🚀 Training setup details
- 📊 Iteration statistics
- 📈 Path performance metrics

### 4. `run_dqn_lander.py`

**تغییرات:**

- ✅ اضافه شدن logging system با فایل `run_dqn_lander_debug.log`
- ✅ Logging جزئیات environment setup
- ✅ Debugging training loop progress
- ✅ نمایش performance milestones
- ✅ Logging final training results

**اطلاعات قابل مشاهده:**

- 🎮 Environment configuration
- 📊 Training progress updates
- 🎯 Major milestones
- 🏆 Final performance metrics

## 📝 فایل‌های Log تولید شده

```
hw3/
├── actor_critic_debug.log      # Actor-Critic debugging
├── dqn_debug.log              # DQN debugging
├── run_ac_debug.log           # AC training debugging
└── run_dqn_lander_debug.log   # DQN training debugging
```

## 🚀 نحوه استفاده

### 1. اجرای Actor-Critic با Debugging

```bash
cd /Users/tahamajs/Documents/uni/DRL/Other_Assisments/homework/hw3
source /Users/tahamajs/Documents/uni/DRL/venv/bin/activate
python run_ac.py CartPole-v0 -n 100 -b 1000 --seed 1
```

### 2. اجرای DQN با Debugging

```bash
python run_dqn_lander.py LunarLander-v2 --num_timesteps 50000 --seed 1
```

### 3. مشاهده Logs در Real-time

```bash
# Actor-Critic logs
tail -f actor_critic_debug.log

# DQN logs
tail -f dqn_debug.log

# Training logs
tail -f run_ac_debug.log
tail -f run_dqn_lander_debug.log
```

## 📊 اطلاعات Debugging موجود

### Actor-Critic:

- 🎯 Agent initialization parameters
- 🏗️ Network architecture details
- 🚀 Trajectory sampling statistics
- 🔄 Critic update progress
- 🎭 Actor advantage calculations
- 📈 Performance metrics per iteration

### DQN:

- 🤖 Agent configuration
- 🏗️ Q-network architecture (CNN/MLP)
- 🎲 Action selection (epsilon-greedy)
- 📚 Replay buffer statistics
- 📉 Training loss progression
- 🎯 Target network updates
- 📊 Episode performance

## 🔧 تنظیمات Logging

### تغییر سطح Logging:

```python
# در هر فایل، این خط را تغییر دهید:
logging.basicConfig(level=logging.DEBUG)  # بیشترین جزئیات
logging.basicConfig(level=logging.INFO)   # اطلاعات متوسط (پیش‌فرض)
logging.basicConfig(level=logging.WARNING) # فقط هشدارها
```

### غیرفعال کردن Logging:

```python
logging.basicConfig(level=logging.CRITICAL)  # فقط خطاهای بحرانی
```

## 🎯 نکات مفید

1. **فایل‌های Log بزرگ می‌شوند** - مرتباً پاک کنید:

   ```bash
   rm *.log
   ```

2. **برای اجرای سریع** - سطح logging را کاهش دهید:

   ```python
   logging.basicConfig(level=logging.WARNING)
   ```

3. **برای تحلیل عمیق** - از DEBUG level استفاده کنید:

   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **مشاهده Real-time** - از `tail -f` استفاده کنید:
   ```bash
   tail -f run_ac_debug.log | grep "📊"
   ```

## 📈 مثال خروجی Log

```
2025-10-04 02:20:00 - __main__ - INFO - 🚀 Starting Actor-Critic training:
2025-10-04 02:20:00 - __main__ - INFO -   📋 Experiment: ac_CartPole-v0
2025-10-04 02:20:00 - __main__ - INFO -   🎮 Environment: CartPole-v0
2025-10-04 02:20:00 - __main__ - INFO -   🔄 Iterations: 100
2025-10-04 02:20:00 - __main__ - INFO - 🔧 Building MLP: scope=policy, input_shape=(?, 4), output_size=2, n_layers=2, size=64
2025-10-04 02:20:00 - __main__ - INFO - ✅ MLP built successfully: final_output_shape=(?, 2)
2025-10-04 02:20:01 - __main__ - INFO - 🚀 Starting new trajectory (animate=False)
2025-10-04 02:20:01 - __main__ - INFO - 🏁 Trajectory completed: steps=25, total_reward=25.000, avg_reward=1.000
```

## 🔍 Troubleshooting

### مشکل: فایل‌های Log خالی هستند

**راه‌حل:** بررسی کنید که logging level درست تنظیم شده باشد.

### مشکل: Logs خیلی پرجزئیات هستند

**راه‌حل:** سطح logging را به INFO یا WARNING تغییر دهید.

### مشکل: فایل‌های Log خیلی بزرگ می‌شوند

**راه‌حل:** از rotation یا compression استفاده کنید یا مرتباً پاک کنید.

---

## 📞 پشتیبانی

اگر سوالی دارید یا مشکلی پیش آمد، فایل‌های log را بررسی کنید و اطلاعات مربوطه را ارائه دهید.

**نکته:** این debugging system به شما کمک می‌کند تا:

- 🔍 مشکلات training را شناسایی کنید
- 📊 عملکرد agent را تحلیل کنید
- 🎯 نقاط بهبود را پیدا کنید
- 📈 پیشرفت training را رصد کنید
