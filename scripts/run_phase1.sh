#!/usr/bin/env bash
# Run the Phase 1 capstone (generate -> cache SLAT -> restyle) in the trellis env.
set -o pipefail
ENV_PREFIX=/home/cookies/miniconda3/envs/trellis
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${ENV_PREFIX}/bin:/usr/local/cuda-11.8/bin:${PATH}"
export PYTHONPATH="${REPO}"                       # so `import slat_studio` works
export PYTHONPATH="${REPO}/third_party/TRELLIS:${PYTHONPATH}"   # and `import trellis`
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on 24GB cards

# Two separate processes so each stays within a single-generation memory footprint (24GB).
python "${REPO}/examples/phase1_generate.py" && python "${REPO}/examples/phase1_restyle.py"
