#!/usr/bin/env bash
# Rebuild the 3 source-compiled TRELLIS extensions that failed under pip build isolation
# (their setup.py imports torch, which the isolated build env lacks). --no-build-isolation
# lets the build see torch/ninja already installed in the trellis env.
set -o pipefail

CONDA_BASE=/home/cookies/miniconda3
ENV_PREFIX="${CONDA_BASE}/envs/trellis"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090 (Ada). Avoids autodetect issues in headless build.

# CUDA 11.8's nvcc rejects gcc > 11; system default gcc is 13. Force gcc-11 for host code
# and as nvcc's host compiler (-ccbin via CUDAHOSTCXX / NVCC_PREPEND_FLAGS).
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

# guard
ACTUAL_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
echo "[ext] python=$(which python) sys.prefix=${ACTUAL_PREFIX}"
[ "${ACTUAL_PREFIX}" = "${ENV_PREFIX}" ] || { echo "[ext] FATAL wrong env"; exit 2; }

EXT_DIR=/tmp/extensions
declare -A SRC=(
  [nvdiffrast]="${EXT_DIR}/nvdiffrast"
  [diffoctreerast]="${EXT_DIR}/diffoctreerast"
  [diff-gaussian-rasterization]="${EXT_DIR}/mip-splatting/submodules/diff-gaussian-rasterization"
)
# re-clone if /tmp got cleaned
[ -d "${EXT_DIR}/nvdiffrast" ]    || git clone https://github.com/NVlabs/nvdiffrast.git "${EXT_DIR}/nvdiffrast"
[ -d "${EXT_DIR}/diffoctreerast" ] || git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git "${EXT_DIR}/diffoctreerast"
[ -d "${EXT_DIR}/mip-splatting" ] || git clone https://github.com/autonomousvision/mip-splatting.git "${EXT_DIR}/mip-splatting"

echo "[ext] host compiler: $(${CXX} --version | head -1)"
for name in nvdiffrast diffoctreerast diff-gaussian-rasterization; do
  path="${SRC[$name]}"
  echo "============================================================"
  echo "[ext] building ${name} from ${path}"
  rm -rf "${path}/build" "${path}"/*.egg-info   # drop stale objects from the failed gcc-13 attempt
  python -m pip install --no-build-isolation "${path}" \
    && echo "[ext] OK ${name}" || echo "[ext] FAIL ${name}"
done

echo "[ext] verifying imports:"
python - <<'PY'
import importlib
for m in ["nvdiffrast.torch", "diffoctreerast", "diff_gaussian_rasterization"]:
    try:
        importlib.import_module(m); print("  [ok]", m)
    except Exception as e:
        print("  [FAIL]", m, "->", repr(e)[:140])
PY
echo "[ext] DONE"
