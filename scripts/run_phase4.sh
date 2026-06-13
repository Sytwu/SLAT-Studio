#!/usr/bin/env bash
# Run the Phase 4 morphing demo (structure union + dissolve schedule) in the trellis env.
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

# Two separate processes: generating the target asset (stage-1 + stage-2 + text encoder) and
# the morph's gaussian decodes/renders each fit one 24GB card on their own, but not together.
python "${REPO}/examples/phase4_gen_target.py" && python "${REPO}/examples/phase4_morph.py"
