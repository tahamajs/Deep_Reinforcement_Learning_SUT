# راهنمای اجرای پروژه CA13

## 🚀 اجرای سریع

### اجرای کامل تمام آزمایشات:

```bash
./run.sh
```

### اجرای سریع (فقط آزمایشات تک عامل):

```bash
./run.sh quick
```

### اجرای Jupyter Notebook:

```bash
./run.sh notebook
```

### اجرای نمایشی:

```bash
./run.sh demo
```

## 🔬 اجرای آزمایشات جداگانه

### آزمایشات تک عامل:

```bash
python3 run_individual_experiments.py single --episodes 100 --save --plot
```

### آزمایشات سلسله مراتبی:

```bash
python3 run_individual_experiments.py hierarchical --episodes 100 --save --plot
```

### آزمایشات چندعاملی:

```bash
python3 run_individual_experiments.py multi --episodes 100 --save --plot
```

### آزمایشات مدل جهانی:

```bash
python3 run_individual_experiments.py world --episodes 100 --save --plot
```

### ارزیابی جامع:

```bash
python3 run_individual_experiments.py comprehensive --save --plot
```

## 📁 ساختار فایل‌ها

```
CA13_Advanced_Model_Based_Hierarchical/
├── run.sh                           # اسکریپت اصلی اجرا
├── run_all_experiments.py           # اجرای کامل تمام آزمایشات
├── run_individual_experiments.py    # اجرای جداگانه آزمایشات
├── CA13.ipynb                       # نوت‌بوک اصلی
├── training_examples.py             # مثال‌های آموزش
├── requirements.txt                 # وابستگی‌ها
├── agents/                          # پیاده‌سازی عامل‌ها
│   ├── model_free.py               # عامل‌های مدل-آزاد
│   ├── model_based.py              # عامل‌های مدل-محور
│   ├── sample_efficient.py         # عامل‌های کارآمد
│   └── hierarchical.py             # عامل‌های سلسله مراتبی
├── environments/                    # محیط‌های آزمایش
│   └── grid_world.py               # محیط شبکه‌ای
├── models/                         # مدل‌های شبکه عصبی
│   └── world_model.py              # مدل جهانی
├── utils/                          # ابزارهای کمکی
│   └── visualization.py            # ابزارهای تصویرسازی
├── evaluation/                     # فریمورک ارزیابی
│   └── advanced_evaluator.py       # ارزیاب پیشرفته
├── buffers/                        # بافرهای تجربه
│   └── replay_buffer.py            # بافر تجربه
├── experiments/                    # آزمایشات نمونه
│   └── demo_comprehensive.py       # نمایش جامع
├── visualizations/                 # نتایج تصویری (ایجاد می‌شود)
├── results/                        # نتایج خام (ایجاد می‌شود)
└── logs/                          # فایل‌های لاگ (ایجاد می‌شود)
```

## 🎯 انواع آزمایشات

### 1. آزمایشات تک عامل (Single Agent)

- **DQN**: یادگیری Q عمیق با تجربه بازپخش
- **Model-Based**: یادگیری بر اساس مدل محیط
- **Sample-Efficient**: یادگیری کارآمد با اولویت‌دهی

### 2. آزمایشات سلسله مراتبی (Hierarchical)

- **Options-Critic**: یادگیری با گزینه‌ها
- **Feudal Networks**: شبکه‌های فئودالی

### 3. آزمایشات چندعاملی (Multi-Agent)

- **Independent DQN**: عوامل مستقل
- **Communication**: ارتباط بین عوامل

### 4. آزمایشات مدل جهانی (World Model)

- **Variational Autoencoder**: مدل‌سازی محیط
- **Imagination-Based Learning**: یادگیری مبتنی بر تخیل

### 5. ارزیابی جامع (Comprehensive Evaluation)

- **Sample Efficiency**: کارایی نمونه
- **Transfer Learning**: یادگیری انتقالی
- **Performance Comparison**: مقایسه عملکرد

## 📊 نتایج و تصاویر

پس از اجرای آزمایشات، نتایج در پوشه‌های زیر ذخیره می‌شوند:

- **visualizations/**: نمودارها و تصاویر
- **results/**: نتایج خام JSON
- **logs/**: فایل‌های لاگ

## 🛠️ رفع مشکلات

### خطای Import:

```bash
pip3 install -r requirements.txt
```

### خطای CUDA:

```bash
# استفاده از CPU به جای GPU
export CUDA_VISIBLE_DEVICES=""
```

### خطای Memory:

```bash
# کاهش اندازه batch
python3 run_individual_experiments.py single --episodes 50
```

## 📈 تفسیر نتایج

### نمودارهای مهم:

1. **Learning Curves**: منحنی‌های یادگیری
2. **Performance Comparison**: مقایسه عملکرد
3. **Sample Efficiency**: کارایی نمونه
4. **Transfer Learning**: قابلیت انتقال

### معیارهای ارزیابی:

- **Mean Reward**: میانگین پاداش
- **Episode Length**: طول قسمت
- **Training Steps**: مراحل آموزش
- **Sample Efficiency**: کارایی نمونه

## 🔧 تنظیمات پیشرفته

### تغییر پارامترهای محیط:

```python
env = SimpleGridWorld(size=10, goal_reward=20.0, step_penalty=-0.05)
```

### تغییر پارامترهای عامل:

```python
agent = DQNAgent(
    state_dim=2,
    action_dim=4,
    hidden_dim=256,
    learning_rate=5e-4
)
```

### تغییر پارامترهای آموزش:

```python
results = train_dqn_agent(
    env, agent,
    num_episodes=500,
    max_steps=1000,
    eval_interval=50
)
```

## 📚 منابع و مراجع

- **World Models** (Ha & Schmidhuber, 2018)
- **Dreamer** (Hafner et al., 2020)
- **Options-Critic** (Bacon et al., 2017)
- **Feudal Networks** (Vezhnevets et al., 2017)
- **Prioritized Experience Replay** (Schaul et al., 2016)

## 🤝 مشارکت

برای مشارکت در پروژه:

1. Fork کنید
2. Branch جدید ایجاد کنید
3. تغییرات را اعمال کنید
4. Pull Request ارسال کنید

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

**موفق باشید! 🎉**
