#!/usr/bin/env python3
"""
CA13: Individual Experiment Runner
اجرای جداگانه آزمایشات مختلف
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all_experiments import (
    run_single_agent_experiments,
    run_hierarchical_experiments,
    run_multi_agent_experiments,
    run_world_model_experiments,
    run_comprehensive_evaluation,
    create_visualizations_folder,
)


def main():
    """تابع اصلی"""
    parser = argparse.ArgumentParser(description="اجرای جداگانه آزمایشات CA13")
    parser.add_argument(
        "experiment",
        choices=["single", "hierarchical", "multi", "world", "comprehensive", "all"],
        help="نوع آزمایش برای اجرا",
    )
    parser.add_argument("--episodes", type=int, default=100, help="تعداد قسمت‌ها")
    parser.add_argument("--save", action="store_true", help="ذخیره نتایج")
    parser.add_argument("--plot", action="store_true", help="نمایش نمودارها")

    args = parser.parse_args()

    print(f"🔬 اجرای آزمایش: {args.experiment}")
    print(f"📊 تعداد قسمت‌ها: {args.episodes}")

    results = {}

    try:
        if args.experiment == "single" or args.experiment == "all":
            print("\n🎯 اجرای آزمایشات تک عامل...")
            results["single_agent"] = run_single_agent_experiments()

        if args.experiment == "hierarchical" or args.experiment == "all":
            print("\n🏗️ اجرای آزمایشات سلسله مراتبی...")
            results["hierarchical"] = run_hierarchical_experiments()

        if args.experiment == "multi" or args.experiment == "all":
            print("\n👥 اجرای آزمایشات چندعاملی...")
            results["multi_agent"] = run_multi_agent_experiments()

        if args.experiment == "world" or args.experiment == "all":
            print("\n🌍 اجرای آزمایشات مدل جهانی...")
            results["world_model"] = run_world_model_experiments()

        if args.experiment == "comprehensive" or args.experiment == "all":
            print("\n📊 اجرای ارزیابی جامع...")
            results["comprehensive"] = run_comprehensive_evaluation()

        # ذخیره نتایج
        if args.save:
            viz_folder = create_visualizations_folder()
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            results_serializable = convert_numpy(results)

            with open(
                os.path.join(viz_folder, f"{args.experiment}_results.json"), "w"
            ) as f:
                json.dump(results_serializable, f, indent=2)

            print(f"💾 نتایج ذخیره شد: {viz_folder}/{args.experiment}_results.json")

        # نمایش نمودارها
        if args.plot:
            plt.show()

        print("\n✅ آزمایشات با موفقیت تکمیل شد!")

    except Exception as e:
        print(f"❌ خطا در اجرای آزمایشات: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
