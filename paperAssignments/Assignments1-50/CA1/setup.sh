#!/usr/bin/env bash
set -euo pipefail

# Create a Python venv and install minimal dependencies for CA1
# Usage: ./setup.sh

PYTHON=${PYTHON:-python3}
VENV_DIR=.venv

echo "Creating virtualenv in ${VENV_DIR} using ${PYTHON}"
${PYTHON} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found in $(pwd)"
fi

echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"






