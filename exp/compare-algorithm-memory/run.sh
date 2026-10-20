#!/usr/bin/env bash
# Configurable SF/BP retained-storage matrix; see README.md.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
exec "${PYTHON:-python}" "${repo_root}/src/compare_algorithm_memory_matrix.py" "$@"
