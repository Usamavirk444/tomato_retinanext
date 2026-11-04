#!/usr/bin/env bash
# One command for every check that runs without a GPU, dataset or checkpoint.
#   ./verify_all.sh
# Exit 0 only if everything passes. Documented manuscript/code divergences
# are reported as DISCREPANCY and do not fail the run.
set -u
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
fail=0
run () { printf '\n\033[1m== %s\033[0m\n' "$1"; shift; "$@" || fail=1; }

run "repository integrity"        python3 tests/test_repo_integrity.py
run "dataset preparation logic"   python3 tests/test_dataset_preparation.py
run "Phase D.3 fix regressions"   python3 tests/test_phase_d3_fixes.py
run "values against the paper"    python3 scripts/verify/validate_against_paper.py
run "reproducibility report"      python3 scripts/verify/reproducibility_report.py
run "tables 1-3"                  python3 scripts/eval/build_tables.py
run "figure sources"              python3 scripts/figures/make_manuscript_figures.py --check

printf '\n\033[1m== summary ==\033[0m\n'
if [ "$fail" -eq 0 ]; then
  echo "  all GPU-free checks passed"
else
  echo "  at least one check failed (figure sources fail without the prepared dataset)"
fi
echo "  reports written to reports/"
echo "  GPU-dependent work is listed in docs/REPRODUCIBILITY_STATUS.md"
exit $fail
