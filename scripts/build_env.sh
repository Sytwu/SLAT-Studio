#!/usr/bin/env bash
# Phase 0: build the TRELLIS conda env (CUDA extensions) non-interactively.
#
# Decisions:
#   - We do NOT pass --new-env to setup.sh (its `conda create` would prompt and hang).
#   - We SKIP --flash-attn for Phase 0 (long, fragile source build). TRELLIS runs with
#     xformers via ATTN_BACKEND=xformers. Add flash-attn later if needed.
#
# Safety: there are TWO conda installs on this host (/home/.../miniconda3 and
# /project2/.../miniconda3) and shells may start with another env (e.g. `coz`) active.
# So we activate the target env by EXPLICIT PREFIX and HARD-GUARD that python really
# resolves inside it before running any pip/conda install — never pollute another env.
# NOTE: do NOT use `set -u` -- conda's own activate/deactivate hook scripts reference
# unbound vars (e.g. CONDA_MKL_INTERFACE_LAYER_BACKUP) and would fatally abort under it.
set -o pipefail

CONDA_BASE=/home/cookies/miniconda3
ENV_PREFIX="${CONDA_BASE}/envs/trellis"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, case-insensitive to dir name
TRELLIS_DIR="${REPO}/third_party/TRELLIS"

export CUDA_HOME=/usr/local/cuda-11.8          # must match pytorch-cuda=11.8
export PATH="$CUDA_HOME/bin:$PATH"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
  echo "[build] creating conda env at ${ENV_PREFIX} (python=3.10)"
  conda create -y -p "${ENV_PREFIX}" python=3.10 || { echo "[build] FAILED creating env"; exit 1; }
fi

# NOTE: `conda activate` does NOT win here -- the shell snapshot hardcodes another env
# (coz) at the front of PATH, outside conda's control. So we set PATH ourselves with the
# trellis env bin FIRST; that makes bare `python`/`pip` (used inside setup.sh) resolve here.
conda activate "${ENV_PREFIX}" 2>/dev/null || true   # sets CONDA_PREFIX for tools that read it
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV=trellis
export PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"

# HARD GUARD: abort if python is not inside the trellis env (prevents polluting coz/base).
ACTUAL_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
echo "[build] python: $(which python)  sys.prefix=${ACTUAL_PREFIX}"
if [ "${ACTUAL_PREFIX}" != "${ENV_PREFIX}" ]; then
  echo "[build] FATAL: active env is '${ACTUAL_PREFIX}', expected '${ENV_PREFIX}'. Aborting before any install."
  exit 2
fi

if ! python -c "import torch" 2>/dev/null; then
  echo "[build] installing pytorch 2.4.0 + cu118"
  conda install -y -p "${ENV_PREFIX}" pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 \
      -c pytorch -c nvidia || { echo "[build] FAILED installing torch"; exit 1; }
fi
python -c "import torch; print('[build] torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "[build] nvcc:"; nvcc --version 2>/dev/null | tail -2 || echo "[build] WARN nvcc not found"

cd "$TRELLIS_DIR"
echo "[build] sourcing TRELLIS setup.sh (basic, xformers, spconv, mipgaussian, kaolin, nvdiffrast, diffoctreerast)"
. ./setup.sh --basic --xformers --spconv --mipgaussian --kaolin --nvdiffrast --diffoctreerast

echo "[build] DONE"
