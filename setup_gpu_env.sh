#!/usr/bin/env bash
# Create .venv/ on the GPU box and install requirements_gpu.txt.
# Idempotent — safe to re-run.
#
# Override the interpreter with PYTHON_BIN=/path/to/python ./setup_gpu_env.sh.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            VER=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
            # Pick anything 3.10+
            if [ "$(printf '%s\n' "3.10" "$VER" | sort -V | head -n1)" = "3.10" ]; then
                PYTHON_BIN=$(command -v "$candidate")
                break
            fi
        fi
    done
fi
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: no python >= 3.10 found. Set PYTHON_BIN=/path/to/python and re-run." >&2
    exit 1
fi

echo "Using $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv/"
fi

# shellcheck source=/dev/null
. .venv/bin/activate

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements_gpu.txt

echo
echo "=== Versions ==="
python -c "import lerobot; print(f'lerobot         {lerobot.__version__}')"
python -c "import torch; print(f'torch           {torch.__version__}')"
python -c "import torch; print(f'cuda available  {torch.cuda.is_available()}')"
python -c "import torch; print(f'cuda devices    {torch.cuda.device_count()}')"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}  ({p.total_memory / 1e9:.1f} GB)')
" || true

echo
echo "Done. Activate with:  source .venv/bin/activate"
