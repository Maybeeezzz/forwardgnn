#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "${script_dir}/../src/test_bp_sparse_cuda.py" "$@"
