#!/bin/bash

# CA4: Policy Gradient Methods - Complete Run Script
# این اسکریپت تمام فایل‌های پروژه را اجرا می‌کند و نتایج را در فولدر visualizations ذخیره می‌کند

echo "=========================================="
echo "CA4: Policy Gradient Methods - اجرای کامل"
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

# نصب وابستگی‌ها
echo "📦 نصب وابستگی‌ها..."
if [ ! -d "venv" ]; then
    echo "ایجاد محیط مجازی..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

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

# اجرای آزمایش‌های کامل
echo "🔬 اجرای آزمایش‌های کامل..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import PolicyGradientExperiment, BenchmarkSuite
import matplotlib.pyplot as plt
import numpy as np

# آزمایش مقایسه الگوریتم‌ها
print('🔄 آزمایش مقایسه الگوریتم‌ها...')
experiment = PolicyGradientExperiment('CartPole-v1')
comparison_results = experiment.run_comparison_experiment(
    algorithms=['reinforce', 'actor_critic'], 
    num_episodes=300
)

# ذخیره نتایج
import pickle
with open('evaluation/results/comparison_results.pkl', 'wb') as f:
    pickle.dump(comparison_results, f)

# ایجاد نمودارها
from utils.visualization import TrainingVisualizer
viz = TrainingVisualizer()

# نمودار مقایسه
viz.plot_multiple_curves(
    {alg: results['scores'] for alg, results in comparison_results.items()},
    'مقایسه REINFORCE و Actor-Critic'
)
plt.savefig('visualizations/algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# نمودارهای جداگانه
for alg_name, results in comparison_results.items():
    viz.plot_learning_curves(results['scores'], f'منحنی یادگیری {alg_name}')
    plt.savefig(f'visualizations/{alg_name}_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'actor_losses' in results and 'critic_losses' in results:
        viz.plot_losses(results['actor_losses'], results['critic_losses'], f'تلفات {alg_name}')
        plt.savefig(f'visualizations/{alg_name}_losses.png', dpi=300, bbox_inches='tight')
        plt.close()

print('✅ آزمایش‌های کامل تکمیل شدند')
"

# اجرای آزمایش‌های اکتشاف
echo "🔍 اجرای آزمایش‌های اکتشاف..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import PolicyGradientExperiment
from agents.exploration import ExplorationScheduler
import matplotlib.pyplot as plt

print('🎯 آزمایش استراتژی‌های اکتشاف...')
experiment = PolicyGradientExperiment('CartPole-v1')

# آزمایش استراتژی‌های مختلف اکتشاف
exploration_results = experiment.run_exploration_experiment(
    base_algorithm='reinforce',
    exploration_strategies=['boltzmann', 'epsilon_greedy'],
    num_episodes=200
)

# ذخیره نتایج اکتشاف
import pickle
with open('evaluation/results/exploration_results.pkl', 'wb') as f:
    pickle.dump(exploration_results, f)

# ایجاد نمودارهای اکتشاف
from agents.exploration import ExplorationVisualizer
exp_viz = ExplorationVisualizer()

exp_viz.compare_exploration_strategies(
    exploration_results,
    'مقایسه استراتژی‌های اکتشاف'
)
plt.savefig('visualizations/exploration_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print('✅ آزمایش‌های اکتشاف تکمیل شدند')
"

# اجرای آزمایش‌های هایپرپارامتر
echo "⚙️ اجرای آزمایش‌های هایپرپارامتر..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import PolicyGradientExperiment
import matplotlib.pyplot as plt
import numpy as np

print('🔧 آزمایش‌های هایپرپارامتر...')
experiment = PolicyGradientExperiment('CartPole-v1')

# آزمایش نرخ یادگیری مختلف
lr_results = experiment.run_hyperparameter_sweep(
    algorithm='reinforce',
    param_name='lr',
    param_values=[0.0001, 0.001, 0.01, 0.1],
    num_episodes=150
)

# ذخیره نتایج هایپرپارامتر
import pickle
with open('evaluation/results/hyperparameter_results.pkl', 'wb') as f:
    pickle.dump(lr_results, f)

# ایجاد نمودار مقایسه هایپرپارامترها
fig, ax = plt.subplots(figsize=(12, 8))
for param_name, results in lr_results.items():
    scores = results['scores']
    if len(scores) >= 20:
        moving_avg = [np.mean(scores[i-20:i]) for i in range(20, len(scores))]
        ax.plot(range(20, len(scores)), moving_avg, label=param_name, linewidth=2)

ax.set_title('مقایسه نرخ‌های یادگیری مختلف')
ax.set_xlabel('اپیزود')
ax.set_ylabel('میانگین امتیاز (20 اپیزود)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('visualizations/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print('✅ آزمایش‌های هایپرپارامتر تکمیل شدند')
"

# اجرای آزمایش‌های محیط‌های مختلف
echo "🌍 اجرای آزمایش‌های محیط‌های مختلف..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
from experiments.experiments import BenchmarkSuite
import matplotlib.pyplot as plt

print('🏆 اجرای مجموعه آزمایش‌های جامع...')
benchmark = BenchmarkSuite()
benchmark_results = benchmark.run_benchmark(episodes_per_env=150)

# ذخیره نتایج مجموعه آزمایش‌ها
import pickle
with open('evaluation/results/benchmark_results.pkl', 'wb') as f:
    pickle.dump(benchmark_results, f)

# ایجاد گزارش
report = benchmark.create_report()
with open('evaluation/results/benchmark_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print('✅ آزمایش‌های محیط‌های مختلف تکمیل شدند')
"

# ایجاد گزارش نهایی
echo "📋 ایجاد گزارش نهایی..."
source venv/bin/activate && python3 -c "
import sys
sys.path.append('.')
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print('📊 ایجاد گزارش جامع...')

# خواندن نتایج
results = {}
try:
    with open('evaluation/results/comparison_results.pkl', 'rb') as f:
        results['comparison'] = pickle.load(f)
except:
    print('نتایج مقایسه یافت نشد')

try:
    with open('evaluation/results/exploration_results.pkl', 'rb') as f:
        results['exploration'] = pickle.load(f)
except:
    print('نتایج اکتشاف یافت نشد')

try:
    with open('evaluation/results/hyperparameter_results.pkl', 'rb') as f:
        results['hyperparameter'] = pickle.load(f)
except:
    print('نتایج هایپرپارامتر یافت نشد')

# ایجاد گزارش HTML
html_report = f'''
<!DOCTYPE html>
<html dir=\"rtl\" lang=\"fa\">
<head>
    <meta charset=\"UTF-8\">
    <title>گزارش CA4: Policy Gradient Methods</title>
    <style>
        body {{ font-family: 'Tahoma', Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .results {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class=\"header\">
        <h1>گزارش CA4: Policy Gradient Methods</h1>
        <p>تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class=\"section\">
        <h2>خلاصه نتایج</h2>
        <div class=\"results\">
'''

if 'comparison' in results:
    html_report += f'''
            <h3>مقایسه الگوریتم‌ها</h3>
            <p>تعداد الگوریتم‌های آزمایش شده: {len(results['comparison'])}</p>
    '''
    for alg_name, alg_results in results['comparison'].items():
        if 'scores' in alg_results:
            final_score = np.mean(alg_results['scores'][-10:]) if len(alg_results['scores']) >= 10 else np.mean(alg_results['scores'])
            best_score = np.max(alg_results['scores'])
            html_report += f'''
                <div class=\"metric\">
                    <strong>{alg_name.upper()}</strong><br>
                    امتیاز نهایی: {final_score:.2f}<br>
                    بهترین امتیاز: {best_score:.2f}
                </div>
            '''

html_report += '''
        </div>
    </div>
    
    <div class=\"section\">
        <h2>نمودارها</h2>
        <p>نمودارهای تولید شده:</p>
        <ul>
            <li>مقایسه الگوریتم‌ها</li>
            <li>منحنی‌های یادگیری</li>
            <li>تلفات آموزش</li>
            <li>مقایسه استراتژی‌های اکتشاف</li>
            <li>مقایسه هایپرپارامترها</li>
        </ul>
    </div>
    
    <div class=\"section\">
        <h2>فایل‌های تولید شده</h2>
        <ul>
            <li>visualizations/ - نمودارها و تصاویر</li>
            <li>evaluation/results/ - نتایج آزمایش‌ها</li>
            <li>models/saved_models/ - مدل‌های ذخیره شده</li>
            <li>logs/ - فایل‌های لاگ</li>
        </ul>
    </div>
</body>
</html>
'''

# ذخیره گزارش HTML
with open('evaluation/results/final_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

# ذخیره گزارش متنی
text_report = f'''
گزارش CA4: Policy Gradient Methods
=====================================
تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

خلاصه نتایج:
'''

if 'comparison' in results:
    text_report += '\nمقایسه الگوریتم‌ها:\n'
    for alg_name, alg_results in results['comparison'].items():
        if 'scores' in alg_results:
            final_score = np.mean(alg_results['scores'][-10:]) if len(alg_results['scores']) >= 10 else np.mean(alg_results['scores'])
            best_score = np.max(alg_results['scores'])
            text_report += f'  {alg_name}: نهایی={final_score:.2f}, بهترین={best_score:.2f}\n'

text_report += '''
فایل‌های تولید شده:
- visualizations/: نمودارها و تصاویر
- evaluation/results/: نتایج آزمایش‌ها  
- models/saved_models/: مدل‌های ذخیره شده
- logs/: فایل‌های لاگ

پروژه با موفقیت تکمیل شد!
'''

with open('evaluation/results/final_report.txt', 'w', encoding='utf-8') as f:
    f.write(text_report)

print('✅ گزارش نهایی ایجاد شد')
"

# نمایش خلاصه نتایج
echo "📊 خلاصه نتایج:"
echo "=================="
echo "📁 فولدرهای ایجاد شده:"
ls -la visualizations/ 2>/dev/null || echo "  visualizations/ - نمودارها"
ls -la evaluation/results/ 2>/dev/null || echo "  evaluation/results/ - نتایج"
ls -la models/saved_models/ 2>/dev/null || echo "  models/saved_models/ - مدل‌ها"
ls -la logs/ 2>/dev/null || echo "  logs/ - لاگ‌ها"

echo ""
echo "📈 فایل‌های تولید شده:"
find visualizations/ -name "*.png" 2>/dev/null | wc -l | xargs echo "  تعداد نمودارها:"
find evaluation/results/ -name "*.pkl" 2>/dev/null | wc -l | xargs echo "  تعداد فایل‌های نتایج:"

echo ""
echo "✅ پروژه CA4 با موفقیت تکمیل شد!"
echo "📋 برای مشاهده گزارش کامل، فایل evaluation/results/final_report.html را باز کنید"
echo "=========================================="