#!/bin/bash

# CA13: Advanced Model-Based RL and Hierarchical RL - Execution Script
# اسکریپت اجرای کامل پروژه CA13

echo "🚀 شروع اجرای پروژه CA13: Advanced Model-Based RL"
echo "================================================================"

# تنظیمات اولیه
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ایجاد پوشه‌های مورد نیاز
mkdir -p visualizations
mkdir -p results
mkdir -p logs

echo "📁 پوشه‌های مورد نیاز ایجاد شد"

# بررسی وجود فایل‌های مورد نیاز
echo "🔍 بررسی فایل‌های مورد نیاز..."

required_files=(
    "requirements.txt"
    "run_all_experiments.py"
    "CA13.ipynb"
    "agents/__init__.py"
    "agents/model_free.py"
    "agents/model_based.py"
    "agents/sample_efficient.py"
    "agents/hierarchical.py"
    "environments/grid_world.py"
    "models/world_model.py"
    "training_examples.py"
    "utils/visualization.py"
    "evaluation/advanced_evaluator.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ فایل مورد نیاز یافت نشد: $file"
        exit 1
    fi
done

echo "✅ تمام فایل‌های مورد نیاز موجود است"

# نصب وابستگی‌ها
echo "📦 نصب وابستگی‌ها..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "❌ pip یافت نشد. لطفاً Python و pip را نصب کنید."
    exit 1
fi

echo "✅ وابستگی‌ها نصب شد"

# اجرای تست‌های اولیه
echo "🧪 اجرای تست‌های اولیه..."

python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    # Test imports
    from agents.model_free import DQNAgent
    from agents.model_based import ModelBasedAgent
    from agents.sample_efficient import SampleEfficientAgent
    from agents.hierarchical import OptionsCriticAgent, FeudalAgent
    from environments.grid_world import SimpleGridWorld
    from models.world_model import VariationalWorldModel
    from training_examples import train_dqn_agent, evaluate_agent
    from utils.visualization import plot_training_curves
    from evaluation.advanced_evaluator import AdvancedRLEvaluator
    
    print('✅ تمام import ها موفقیت‌آمیز بود')
    
    # Test environment creation
    env = SimpleGridWorld(size=5)
    obs = env.reset()
    action = env.action_space.sample() if hasattr(env, 'action_space') else 0
    next_obs, reward, done, _ = env.step(action)
    print('✅ محیط با موفقیت ایجاد شد')
    
    # Test agent creation
    agent = DQNAgent(state_dim=2, action_dim=4)
    action = agent.act(obs)
    print('✅ عامل با موفقیت ایجاد شد')
    
except Exception as e:
    print(f'❌ خطا در تست‌های اولیه: {str(e)}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ تست‌های اولیه ناموفق بود"
    exit 1
fi

echo "✅ تست‌های اولیه موفقیت‌آمیز بود"

# اجرای آزمایشات اصلی
echo "🔬 شروع اجرای آزمایشات اصلی..."

# انتخاب نوع اجرا
if [ "$1" = "quick" ]; then
    echo "⚡ اجرای سریع (Quick Run)"
    python3 -c "
import sys
sys.path.insert(0, '.')
from run_all_experiments import run_single_agent_experiments
results = run_single_agent_experiments()
print('✅ آزمایشات سریع تکمیل شد')
"
elif [ "$1" = "notebook" ]; then
    echo "📓 اجرای Jupyter Notebook"
    if command -v jupyter &> /dev/null; then
        jupyter notebook CA13.ipynb
    else
        echo "❌ Jupyter یافت نشد. لطفاً Jupyter را نصب کنید: pip install jupyter"
        exit 1
    fi
elif [ "$1" = "demo" ]; then
    echo "🎯 اجرای نمایشی"
    python3 experiments/demo_comprehensive.py
else
    echo "🔬 اجرای کامل تمام آزمایشات"
    python3 run_all_experiments.py
fi

# بررسی نتایج
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 اجرای پروژه با موفقیت تکمیل شد!"
    echo "================================================================"
    
    # نمایش فایل‌های تولید شده
    echo "📁 فایل‌های تولید شده:"
    if [ -d "visualizations" ]; then
        echo "  📊 تصاویر و نمودارها:"
        ls -la visualizations/*.png 2>/dev/null | sed 's/^/    /'
    fi
    
    if [ -d "results" ]; then
        echo "  📋 نتایج:"
        ls -la results/*.json 2>/dev/null | sed 's/^/    /'
    fi
    
    echo ""
    echo "📖 برای مشاهده نتایج:"
    echo "  - نمودارها: پوشه visualizations/"
    echo "  - نتایج: پوشه results/"
    echo "  - لاگ‌ها: پوشه logs/"
    
    echo ""
    echo "🚀 دستورات مفید:"
    echo "  ./run.sh quick     - اجرای سریع"
    echo "  ./run.sh notebook  - اجرای Jupyter Notebook"
    echo "  ./run.sh demo      - اجرای نمایشی"
    echo "  ./run.sh           - اجرای کامل"
    
else
    echo ""
    echo "❌ خطا در اجرای پروژه"
    echo "================================================================"
    echo "📋 برای رفع مشکل:"
    echo "  1. بررسی لاگ‌های خطا در بالا"
    echo "  2. اطمینان از نصب صحیح وابستگی‌ها"
    echo "  3. بررسی وجود تمام فایل‌های مورد نیاز"
    echo "  4. اجرای مجدد: ./run.sh"
    exit 1
fi

echo ""
echo "✅ پروژه CA13 با موفقیت اجرا شد!"
