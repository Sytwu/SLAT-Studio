#!/usr/bin/env bash
# Run the Phase 5 inpainting/completion demo (two-stage RePaint) in the trellis env.
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

# Single process: inpaint = stage-1 (SS encoder + SS flow + SS decoder) + stage-2 (slat flow) +
# text encoder ≈ a generation footprint, fits one 24GB card; the three GS decodes run afterward
# one at a time with the flow models parked on CPU.
python "${REPO}/examples/phase5_inpaint.py"
