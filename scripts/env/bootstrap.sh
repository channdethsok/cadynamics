#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# scripts/env/bootstrap.sh
# Reproducible environment bootstrap for CADynamics on Linux.
#
# Hybrid Dependency Architecture:
#   1. Conda owns Python, CadQuery, OCP, OpenCASCADE, Mesa, and uv (environment.yml).
#   2. uv owns Python research/ML packages (PyTorch, PyArrow, CADynamics).
#
# Safe to rerun. Does not hardcode machine-specific absolute paths.
# ==============================================================================

log() {
    echo -e "[bootstrap_env] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="cadyn-env"
ENV_FILE="$REPO_ROOT/environment.yml"

# Increase uv network timeout to 120s for environments behind high-latency proxies
export UV_HTTP_TIMEOUT=120

log "Repository root: $REPO_ROOT"
log "Target Conda environment: $ENV_NAME"

# ------------------------------------------------------------------------------
# 1. Locate or install Conda
# ------------------------------------------------------------------------------
init_conda() {
    if command -v conda >/dev/null 2>&1; then
        log "Found conda in PATH: $(conda --version)"
        CONDA_BASE="$(conda info --base)"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        return
    fi

    for candidate in "$HOME/.local/miniconda" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [ -d "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            log "Found existing Conda installation at $candidate"
            source "$candidate/etc/profile.d/conda.sh"
            return
        fi
    done

    log "Conda not found. Installing Miniconda to $HOME/.local/miniconda..."
    MINICONDA_DIR="$HOME/.local/miniconda"
    TMP_SH="/tmp/miniconda_installer_$$.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMP_SH"
    bash "$TMP_SH" -b -p "$MINICONDA_DIR"
    rm -f "$TMP_SH"
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    log "Miniconda installed successfully"
}

init_conda

# ------------------------------------------------------------------------------
# 2. Create or update Conda cadyn-env from environment.yml
# ------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Environment '$ENV_NAME' already exists. Skipping Conda solve and reusing environment."
else
    log "Creating '$ENV_NAME' from $ENV_FILE..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
log "Activated Conda environment at: $CONDA_PREFIX"

# Clean problematic MKL activation scripts if present
rm -f \
    "$CONDA_PREFIX/etc/conda/activate.d/libblas_mkl_activate.sh" \
    "$CONDA_PREFIX/etc/conda/deactivate.d/libblas_mkl_deactivate.sh" 2>/dev/null || true

# ------------------------------------------------------------------------------
# 3. Ensure uv is available in cadyn-env
# ------------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found in active environment. Installing uv via Conda..."
    conda install -y -c conda-forge uv
fi
log "uv version: $(uv --version)"

# ------------------------------------------------------------------------------
# 4. Install locked Python dependencies into Conda interpreter using uv
# ------------------------------------------------------------------------------
TARGET_PYTHON="$CONDA_PREFIX/bin/python"

log "Installing PyTorch 2.4.1 and TorchVision 0.19.1..."
# Attempt PyTorch dedicated CUDA wheel index first; fall back to standard PyPI if network/proxy restricts download.pytorch.org
if ! uv pip install \
    --python "$TARGET_PYTHON" \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121; then
    log "Notice: uv connection limit reached on proxy. Installing PyTorch via pip..."
    "$TARGET_PYTHON" -m pip install torch==2.4.1 torchvision==0.19.1
fi

log "Installing CADynamics and locked dependencies from repository..."
if ! uv pip install \
    --python "$TARGET_PYTHON" \
    -e "$REPO_ROOT"; then
    log "Notice: uv hit proxy connection limit. Falling back to sequential pip install..."
    "$TARGET_PYTHON" -m pip install -e "$REPO_ROOT"
fi

# ------------------------------------------------------------------------------
# 5. Verify environment integrity
# ------------------------------------------------------------------------------
log "Running environment verification..."
"$TARGET_PYTHON" "$SCRIPT_DIR/verify.py"

log "============================================================"
log "Bootstrap completed successfully!"
log "To activate this environment:"
log "    conda activate $ENV_NAME"
log "============================================================"
