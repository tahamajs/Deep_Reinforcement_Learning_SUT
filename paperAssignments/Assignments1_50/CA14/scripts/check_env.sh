#!/usr/bin/env bash
set -euo pipefail

echo "Checking environment for CA14..."
PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
  echo "Python3 not found. Install Python 3.10+ and retry."
  exit 1
fi
echo "Using $PYTHON"

# Check pip
PIP=$(command -v pip3 || command -v pip || true)
if [ -z "$PIP" ]; then
  echo "pip not found. Try: python3 -m ensurepip --upgrade"
fi

# Check deps
python3 - <<'PY'
import importlib, sys
reqs = ['torch','gymnasium','numpy','pyyaml']
missing = [r for r in reqs if importlib.util.find_spec(r) is None]
if missing:
    print('Missing packages:', missing)
    print('Install: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('All required runtime packages are installed.')
PY

echo "All checks passed (or instructions printed)."