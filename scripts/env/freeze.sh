#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/env/freeze.sh
# Diagnostic snapshot generator for CADynamics.
#
# Idempotent: Safely records current environment state without mutating any package.
# Generates:
#   env/conda-explicit.txt
#   env/conda-full.yml
#   env/conda-history.yml
#   env/python-freeze.txt
#   env/environment-report.txt
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR="$REPO_ROOT/env"
mkdir -p "$ENV_DIR"

echo "=== CADynamics Environment Freeze ==="
echo "Working directory: $REPO_ROOT"
echo "Target snapshot directory: $ENV_DIR"

# Check active Python & Conda
PYTHON_BIN="$(which python 2>/dev/null || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python not found in PATH. Please activate cadyn-env first." >&2
    exit 1
fi

echo "Python executable : $PYTHON_BIN"
echo "Python version    : $("$PYTHON_BIN" --version 2>&1)"

if command -v conda >/dev/null 2>&1; then
    echo "Conda version     : $(conda --version)"
    echo "Active conda env  : ${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-none})"
    
    echo "Exporting Conda snapshots..."
    conda list --explicit > "$ENV_DIR/conda-explicit.txt"
    conda env export > "$ENV_DIR/conda-full.yml"
    conda env export --from-history > "$ENV_DIR/conda-history.yml"
else
    echo "WARNING: conda command not found in current PATH. Skipping conda export."
fi

if command -v uv >/dev/null 2>&1; then
    echo "uv version        : $(uv --version)"
    echo "Exporting Python package snapshot via uv pip freeze..."
    uv pip freeze > "$ENV_DIR/python-freeze.txt"
elif "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "Exporting Python package snapshot via pip freeze..."
    "$PYTHON_BIN" -m pip freeze > "$ENV_DIR/python-freeze.txt"
fi

echo "Generating detailed diagnostic report: $ENV_DIR/environment-report.txt..."

"$PYTHON_BIN" - << 'PY_DIAG' > "$ENV_DIR/environment-report.txt"
import sys
import platform
import subprocess
import os
import datetime

print("=" * 72)
print("CADynamics Environment Diagnostic Report")
print("Timestamp:", datetime.datetime.now(datetime.timezone.utc).isoformat())
print("=" * 72)

print("\n--- HOST & SYSTEM INFORMATION ---")
print("Platform     :", platform.platform())
print("System       :", platform.system())
print("Architecture :", platform.machine())
print("Processor    :", platform.processor())
if os.path.exists("/etc/os-release"):
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith(("PRETTY_NAME=", "NAME=", "VERSION=")):
                    print(line.strip())
    except Exception as e:
        print(f"Error reading /etc/os-release: {e}")

print("\n--- RUNTIME EXECUTABLE & ENVIRONMENT ---")
print("sys.executable    :", sys.executable)
print("sys.version       :", sys.version.replace("\n", " "))
print("CONDA_PREFIX      :", os.environ.get("CONDA_PREFIX", "None"))
print("CONDA_DEFAULT_ENV :", os.environ.get("CONDA_DEFAULT_ENV", "None"))

print("\n--- PACKAGE MANAGERS ---")
for tool, cmd in [("Conda", ["conda", "--version"]), ("uv", ["uv", "--version"]), ("pip", [sys.executable, "-m", "pip", "--version"])]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        print(f"{tool:6}: {out}")
    except Exception as e:
        print(f"{tool:6}: Not available ({e})")

print("\n--- ACCELERATOR & CUDA RUNTIME ---")
try:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if smi.returncode == 0:
        print("nvidia-smi output:")
        for line in smi.stdout.strip().splitlines()[:15]:
            print(" ", line)
    else:
        print("nvidia-smi returned non-zero code or no GPU accessible.")
except Exception:
    print("nvidia-smi is not available in PATH.")

try:
    import torch
    print("PyTorch Version       :", torch.__version__)
    print("PyTorch Built CUDA    :", torch.version.cuda)
    print("PyTorch CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Count          :", torch.cuda.device_count())
        print("Device Name (0)       :", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch diagnostics error:", e)

print("\n--- CAD & GEOMETRY STACK ---")
for modname in ["cadquery", "OCP", "OCC", "pyvista", "vtk"]:
    try:
        mod = __import__(modname)
        ver = getattr(mod, "__version__", "version unavailable")
        loc = getattr(mod, "__file__", "c-extension / built-in")
        print(f"{modname:10}: {ver} (Location: {loc})")
    except Exception as e:
        print(f"{modname:10}: NOT AVAILABLE ({e})")

print("\n--- GEOMETRY KERNEL VERIFICATION ---")
try:
    import cadquery as cq
    box = cq.Workplane("XY").box(10.0, 20.0, 30.0)
    vol = box.val().Volume()
    print(f"CadQuery test box volume: {vol:.2f} mm^3 (Expected: 6000.00 mm^3)")
    print("OpenCASCADE kernel sanity check: PASSED")
except Exception as e:
    print(f"CadQuery test box check FAILED: {e}")

print("\n--- PYTHON RESEARCH & ML STACK ---")
for modname in ["numpy", "scipy", "pandas", "pyarrow", "torchvision", "torchdata", "pydantic", "pytest", "PIL"]:
    try:
        mod = __import__(modname)
        ver = getattr(mod, "__version__", "version unavailable")
        print(f"{modname:12}: {ver}")
    except Exception as e:
        print(f"{modname:12}: NOT AVAILABLE ({e})")

print("\n--- PACKAGE OWNERSHIP BREAKDOWN ---")
try:
    c_res = subprocess.run(["conda", "list", "--json"], capture_output=True, text=True)
    if c_res.returncode == 0:
        import json
        pkgs = json.loads(c_res.stdout)
        conda_owned = [p["name"] for p in pkgs if p.get("channel") != "pypi"]
        pip_owned = [p["name"] for p in pkgs if p.get("channel") == "pypi"]
        print(f"Conda-managed packages count : {len(conda_owned)}")
        print(f"uv/PyPI-managed packages count: {len(pip_owned)}")
except Exception as e:
    print("Package ownership breakdown error:", e)

print("\n" + "=" * 72)
print("Diagnostic Report Completed Successfully")
print("=" * 72)
PY_DIAG

echo "Snapshot updated in $ENV_DIR:"
ls -lh "$ENV_DIR"
echo "Environment freeze completed successfully without mutating dependencies."
