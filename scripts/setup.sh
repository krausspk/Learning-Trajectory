#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --no-deps -e .
