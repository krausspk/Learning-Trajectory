#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$experiment_dir")"
python_bin="$project_dir/.venv/bin/python"
cd "$project_dir"
"$python_bin" -m pytest -q "$experiment_dir/tests"
"$python_bin" "$experiment_dir/code/analyze.py" --cohort exploratory
"$python_bin" "$experiment_dir/code/train_cohorts.py" --cohort one_bit --workers "${1:-4}"
"$python_bin" "$experiment_dir/code/analyze.py" --cohort one_bit
"$python_bin" "$experiment_dir/code/train_cohorts.py" --cohort multi_bit --workers "${1:-4}"
"$python_bin" "$experiment_dir/code/analyze.py" --cohort multi_bit
"$python_bin" "$experiment_dir/code/diagnose_feasibility.py" --pair-score "$experiment_dir/analysis/one_bit/event_both.npz" --pair-score "$experiment_dir/analysis/multi_bit/event_both.npz"
"$python_bin" "$experiment_dir/code/validate_results.py"
"$python_bin" "$experiment_dir/code/audit_outputs.py"
"$python_bin" "$experiment_dir/code/reporting.py"
