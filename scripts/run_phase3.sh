#!/usr/bin/env bash
# Run the Phase 3 region-editing demo (RePaint masked SLAT sampling) in the trellis env.
# Requires outputs/phase1_base.npz (run scripts/run_phase1.sh first).
set -o pipefail
ENV_PREFIX=/home/cookies/miniconda3/envs/trellis
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${ENV_PREFIX}/bin:/usr/local/cuda-11.8/bin:${PATH}"
export PYTHONPATH="${REPO}"                                      # so `import slat_studio` works
export PYTHONPATH="${REPO}/third_party/TRELLIS:${PYTHONPATH}"    # and `import trellis`
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True          # reduce fragmentation on 24GB cards

# Single process: edit = one stage-2 pass + the gaussian decoder + text encoder (no stage-1,
# no mesh/rf decoders) ≈ the restyle footprint, fits one 24GB card.
python "${REPO}/examples/phase3_edit.py"
