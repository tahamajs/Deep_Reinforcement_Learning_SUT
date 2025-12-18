#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run unit tests for CA10
python -m pytest tests/ -q
