"""
Helper to create a W&B sweep and run agents locally.
This script shells out to the `wandb` CLI. Ensure `wandb` is installed and you are logged in.
"""

import argparse
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sweep-file",
        type=str,
        default="paperAssignments/Assignments1-50/CA8/wandb_sweep.yaml",
    )
    p.add_argument(
        "--count", type=int, default=1, help="number of agents to run via wandb agent"
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Create sweep
    try:
        res = subprocess.run(
            ["wandb", "sweep", args.sweep_file],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            "Failed to create sweep. Make sure wandb is installed and logged in.",
            file=sys.stderr,
        )
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    # Parse sweep id from stdout (wandb prints 'Created sweep with ID: <id>')
    out = res.stdout.strip()
    sweep_id = None
    for line in out.splitlines():
        if "Created sweep with ID" in line:
            sweep_id = line.split()[-1]
            break
    if sweep_id is None:
        print("Could not parse sweep id; output:\n", out)
        sys.exit(1)

    print("Starting wandb agent(s) for sweep:", sweep_id)
    # Launch agents
    try:
        subprocess.run(
            ["wandb", "agent", sweep_id, "--count", str(args.count)], check=True
        )
    except subprocess.CalledProcessError as e:
        print("wandb agent failed:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


