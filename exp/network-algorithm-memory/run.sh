#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
exec "${PYTHON:-python}" "${repo_root}/src/network_algorithm_memory.py" "$@"
