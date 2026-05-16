#!/usr/bin/env sh
# Purpose:
#   Create the project's local `.venv` with `uv` and install the dependencies
#   declared in `pyproject.toml`.
#
# Parameters (override through environment variables before `sh`):
#   UV_PYTHON      Python version or interpreter path used by `uv venv`.
#                  Default: 3.12
#   RECREATE_VENV  Remove and recreate `.venv` when set to 1/true.
#                  Default: 0
#   UV_LINK_MODE   uv link mode.
#                  Default: copy
#   UV_NATIVE_TLS  Use system TLS certificates when set to 1/true.
#                  Default: 0
#   UV_INDEX_URL   Optional primary package index URL.
#                  Default: empty
#   UV_EXTRA_INDEX_URL
#                  Optional extra package index URL.
#                  Default: empty
#   CHECK_ONLY     Only print resolved environment info when set to 1/true.
#                  Default: 0
#
# Notes:
#   - This script installs the project into `.venv`.
#   - `torch` is declared in `pyproject.toml`; on CUDA-capable Linux hosts,
#     verify the final runtime with:
#       uv run python -c "import torch; print(torch.cuda.is_available())"
#   - On the current macOS host, CUDA is not expected. MPS/CPU is the normal path.
#
# Examples:
#   sh run_install_deps.sh
#   UV_PYTHON=3.12 RECREATE_VENV=1 sh run_install_deps.sh
#   UV_NATIVE_TLS=1 sh run_install_deps.sh
#   UV_NATIVE_TLS=1 UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple sh run_install_deps.sh
#   CHECK_ONLY=1 sh run_install_deps.sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
cd "$PROJECT_DIR"

UV_PYTHON=${UV_PYTHON:-3.12}
RECREATE_VENV=${RECREATE_VENV:-0}
UV_LINK_MODE=${UV_LINK_MODE:-copy}
UV_NATIVE_TLS=${UV_NATIVE_TLS:-0}
UV_INDEX_URL=${UV_INDEX_URL:-}
UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL:-}
CHECK_ONLY=${CHECK_ONLY:-0}

if [ "$RECREATE_VENV" = "1" ] || [ "$RECREATE_VENV" = "true" ]; then
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  uv venv .venv --python "$UV_PYTHON"
fi

set -- uv sync --link-mode "$UV_LINK_MODE"
if [ "$UV_NATIVE_TLS" = "1" ] || [ "$UV_NATIVE_TLS" = "true" ]; then
  set -- "$@" --native-tls
fi
if [ -n "$UV_INDEX_URL" ]; then
  set -- "$@" --index-url "$UV_INDEX_URL"
fi
if [ -n "$UV_EXTRA_INDEX_URL" ]; then
  set -- "$@" --extra-index-url "$UV_EXTRA_INDEX_URL"
fi
"$@"

if [ "$CHECK_ONLY" = "1" ] || [ "$CHECK_ONLY" = "true" ]; then
  uv run python -c "import sys; print(sys.executable)"
  uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
  exit 0
fi

uv run python -c "import sys; print('python:', sys.executable)"
uv run python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'mps:', torch.backends.mps.is_available())"
