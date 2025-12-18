"""Evaluation helpers for AU-DMG (placeholder)."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="antmaze-medium-diverse-v2")
    args = parser.parse_args()
    print(f"Evaluation entry point (env={args.env}). Implement environment wrappers and metrics.")

if __name__ == "__main__":
    main()

