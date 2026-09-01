#!/usr/bin/env bash
# Create the virtual environment this repository expects and verify it.
#
#   bash setup_env.sh                  create project_version and install requirements.txt
#   source project_version/bin/activate  then use it
#
# Override the name with VENV=some_other_name bash setup_env.sh
#
# Run this from the repository root. 
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"
VENV="${VENV:-project_version}"
echo "interpreter : $($PY -V 2>&1)"

if [ ! -d "$VENV" ]; then
    echo "creating    : $VENV"
    "$PY" -m venv "$VENV"
else
    echo "reusing     : $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

echo
echo "verifying the imports the pipeline actually uses"
python - <<'PYEOF'
import importlib, sys
ok = True
for mod in ("numpy", "pandas", "yaml", "torch", "matplotlib"):
    try:
        m = importlib.import_module(mod)
        print("  %-11s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as exc:
        ok = False
        print("  %-11s MISSING (%s)" % (mod, exc))
import torch
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda (%s)" % torch.cuda.get_device_name(0)
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    dev = "mps (Apple Silicon)"
print("  compute     %s" % dev)
sys.exit(0 if ok else 1)
PYEOF

echo
echo "next:"
echo "  source $VENV/bin/activate"
echo "  python ml_modelling/scripts/selfcheck.py"
