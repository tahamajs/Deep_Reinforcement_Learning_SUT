"""
Simple plotting utility for gamma/variance trajectories saved as CSV logs.
This script is non-executing by default and intended as a helper for reproduction.
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def plot(path: str, out: str):
    df = pd.read_csv(path)
    plt.figure(figsize=(8, 4))
    plt.plot(df["update"], df["gamma"], label="gamma")
    if "varA" in df.columns:
        plt.plot(df["update"], df["varA"], label="varA")
    plt.axhline(df["gamma"].min(), color="gray", linestyle="--", label="gamma_min")
    plt.axhline(df["gamma"].max(), color="gray", linestyle=":", label="gamma_max")
    plt.xlabel("update")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    p.add_argument("out", type=str)
    args = p.parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()










