#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="${ROOT}/../../.venv/bin/python"
export PYTHONPATH="$ROOT":"$PYTHONPATH"

echo "Running smoke run with PYTHONPATH=$PYTHONPATH using $PYTHON"
$PYTHON $ROOT/scripts/smoke_run.py
