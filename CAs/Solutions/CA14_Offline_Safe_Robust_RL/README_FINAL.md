# CA14: Advanced Deep Reinforcement Learning

## 🎯 پروژه یادگیری تقویتی پیشرفته - تکمیل شده

این پروژه شامل پیاده‌سازی کامل الگوریتم‌های پیشرفته یادگیری تقویتی عمیق است:

### 📚 موضوعات پوشش داده شده

#### 1. یادگیری تقویتی آفلاین (Offline RL)

- **Conservative Q-Learning (CQL)**: جلوگیری از سوگیری بیش‌برآورد با جریمه‌های محافظه‌کارانه
- **Implicit Q-Learning (IQL)**: اجتناب از بهبود صریح سیاست از طریق رگرسیون انتظاری

#### 2. یادگیری تقویتی ایمن (Safe RL)

- **Constrained Policy Optimization (CPO)**: روش‌های منطقه اعتماد با رضایت محدودیت
- **Lagrangian Methods**: جریمه تطبیقی متعادل‌کننده عملکرد و ایمنی

#### 3. یادگیری تقویتی چندعاملی (Multi-Agent RL)

- **MADDPG**: آموزش متمرکز با اجرای غیرمتمرکز
- **QMIX**: فاکتورسازی تابع ارزش یکنواخت برای هماهنگی تیمی

#### 4. یادگیری تقویتی مقاوم (Robust RL)

- **Domain Randomization**: آموزش در پیکربندی‌های متنوع محیط
- **Adversarial Training**: مقاومت در برابر اختلالات ورودی و عدم قطعیت مدل

## 🚀 نحوه اجرا

### روش 1: اجرای کامل پروژه

```bash
# فعال‌سازی محیط مجازی
source /Users/tahamajs/Documents/uni/DRL/venv/bin/activate

# اجرای کامل پروژه
./run.sh
```

### روش 2: اجرای سریع

```bash
# فعال‌سازی محیط مجازی
source /Users/tahamajs/Documents/uni/DRL/venv/bin/activate

# اجرای اسکریپت سریع
python quick_start.py
```

### روش 3: اجرای جداگانه

```bash
# فعال‌سازی محیط مجازی
source /Users/tahamajs/Documents/uni/DRL/venv/bin/activate

# تست ماژول‌ها
python test_modules.py

# اجرای مثال‌های آموزشی
python training_examples.py

# باز کردن نوت‌بوک تعاملی
jupyter notebook CA14.ipynb
```

## 📁 ساختار پروژه

```
CA14_Offline_Safe_Robust_RL/
├── 📓 CA14.ipynb                 # نوت‌بوک اصلی تعاملی
├── 🎯 training_examples.py       # اسکریپت آموزشی کامل
├── 🧪 test_modules.py            # تست ماژول‌ها
├── ⚡ quick_start.py             # اجرای سریع
├── 🚀 run.sh                     # اسکریپت اجرای کامل
├── 📋 requirements.txt           # وابستگی‌ها
├── 📖 README.md                  # مستندات
├── 📂 offline_rl/                # پیاده‌سازی‌های Offline RL
│   ├── algorithms.py             # CQL, IQL
│   ├── dataset.py                # مدیریت دیتاست آفلاین
│   └── utils.py                  # ابزارهای تولید دیتاست
├── 📂 safe_rl/                   # پیاده‌سازی‌های Safe RL
│   ├── agents.py                 # CPO, Lagrangian
│   ├── environment.py            # محیط ایمن
│   └── utils.py                  # ابزارهای Safe RL
├── 📂 multi_agent/               # پیاده‌سازی‌های Multi-Agent RL
│   ├── agents.py                 # MADDPG, QMIX
│   ├── environment.py            # محیط چندعاملی
│   └── buffers.py                # بافرهای چندعاملی
├── 📂 robust_rl/                 # پیاده‌سازی‌های Robust RL
│   ├── agents.py                 # Domain Randomization, Adversarial
│   ├── environment.py            # محیط مقاوم
│   └── utils.py                  # ابزارهای Robust RL
├── 📂 evaluation/                # چارچوب ارزیابی
│   └── advanced_evaluator.py    # ارزیابی جامع
├── 📂 environments/              # پیاده‌سازی‌های محیط
│   └── grid_world.py             # محیط گرید ساده
├── 📂 utils/                     # توابع کمکی
│   └── evaluation_utils.py       # ابزارهای ارزیابی
├── 📂 visualizations/            # نمودارها و نتایج
│   └── CA14_comprehensive_results.png
├── 📂 results/                   # نتایج تحلیل
│   └── CA14_summary_report.md
└── 📂 logs/                      # لاگ‌های اجرا
```

## ✅ ویژگی‌های کلیدی

- ✅ پیاده‌سازی کامل تمام الگوریتم‌های اصلی
- ✅ چارچوب ارزیابی جامع
- ✅ تحلیل عملکرد چندبعدی
- ✅ ملاحظات استقرار در دنیای واقعی
- ✅ تجسم و گزارش‌دهی گسترده
- ✅ تست‌های کامل و خودکار
- ✅ مستندات کامل فارسی و انگلیسی

## 📊 نتایج

پروژه با موفقیت تکمیل شده و شامل:

- **ارزیابی جامع** در ابعاد مختلف
- **مقایسه عملکرد** تمام روش‌ها
- **تحلیل مقاومت و ایمنی**
- **اثربخشی هماهنگی چندعاملی**
- **تحلیل تجسمی و گزارش‌دهی**

## 🎉 وضعیت پروژه

**✅ تکمیل شده و آماده استفاده**

تمام فایل‌ها تکمیل شده، تست شده و آماده اجرا هستند. پروژه شامل:

1. **پیاده‌سازی کامل** تمام الگوریتم‌های پیشرفته RL
2. **اسکریپت اجرای خودکار** برای اجرای کامل پروژه
3. **تست‌های جامع** برای اطمینان از عملکرد صحیح
4. **مستندات کامل** به زبان فارسی
5. **نتایج تجسمی** و گزارش‌های تحلیلی

## 🔧 نیازمندی‌ها

- Python 3.8+
- PyTorch
- NumPy, Matplotlib, Seaborn
- Pandas, Plotly
- Gym/Gymnasium
- Jupyter Notebook

## 📞 پشتیبانی

در صورت بروز مشکل، لاگ‌های اجرا در پوشه `logs/` موجود است.

---

**تاریخ تکمیل**: ۴ اکتبر ۲۰۲۵  
**وضعیت**: ✅ تکمیل شده و آماده استفاده


