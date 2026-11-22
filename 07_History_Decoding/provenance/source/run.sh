#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$experiment_dir")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$project_dir/.venv/bin/python" "$experiment_dir/code/training.py" --output "$experiment_dir" --mode all --workers "${1:-4}"
"$project_dir/.venv/bin/python" "$experiment_dir/code/audit_training.py"
"$project_dir/.venv/bin/python" "$experiment_dir/code/probes.py" --output "$experiment_dir"
"$project_dir/.venv/bin/python" "$experiment_dir/code/reporting.py" --output "$experiment_dir"
"$project_dir/.venv/bin/python" "$experiment_dir/code/finalize.py"
