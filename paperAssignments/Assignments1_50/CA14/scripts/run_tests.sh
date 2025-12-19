#!/usr/bin/env bash
set -euo pipefail

# Run tests using the venv python and ensure CA14 is on PYTHONPATH
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="${ROOT}/../../.venv/bin/python"
export PYTHONPATH="$ROOT":"$PYTHONPATH"

echo "Running tests with PYTHONPATH=$PYTHONPATH using $PYTHON"
$PYTHON -m pytest "$ROOT/tests" -q
