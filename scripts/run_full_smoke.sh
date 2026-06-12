#!/usr/bin/env bash
# Run the full image->3D smoke inside the trellis env with correct PATH/PYTHONPATH.
set -o pipefail
ENV_PREFIX=/home/cookies/miniconda3/envs/trellis
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${ENV_PREFIX}/bin:/usr/local/cuda-11.8/bin:${PATH}"
export PYTHONPATH="${REPO}/third_party/TRELLIS"
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
# pin to one GPU for the test
export CUDA_VISIBLE_DEVICES=0
python "${REPO}/scripts/full_smoke.py"
