# Environment setup

SLAT-Studio reuses TRELLIS's environment (PyTorch + custom CUDA extensions). We do **not**
pin these in `pyproject.toml`; instead we build the TRELLIS env once and run inside it.

## Verified host (this machine)

- 4× NVIDIA RTX 4090 (24 GB each), driver 580.105.08 (CUDA 13.0 capable)
- CUDA toolkits available under `/usr/local/cuda-*` (11.7 → 13.0)
- conda 25.7, gcc 13.3, Python 3.10
- base env already has `torch 2.8.0+cu128`

## Building the TRELLIS env

TRELLIS is **clone-only** (no pip package) and compiles CUDA extensions, so it needs a
toolkit `nvcc` matching the PyTorch CUDA build.

`setup.sh --new-env` pins **PyTorch 2.4.0 + torchvision 0.19.0 + pytorch-cuda=11.8** in a
conda env named `trellis`. So the source-built extensions (`diffoctreerast`, `mip-gaussian`)
must compile against **CUDA 11.8** — set `CUDA_HOME=/usr/local/cuda-11.8` (present on this
host):

```bash
export CUDA_HOME=/usr/local/cuda-11.8     # MUST match the env's pytorch-cuda=11.8
export PATH="$CUDA_HOME/bin:$PATH"
cd third_party/TRELLIS
. ./setup.sh --new-env --basic --xformers --flash-attn \
    --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

`--new-env` is preferred over reusing base (which has torch 2.8/cu128) so extension wheels
(flash-attn, spconv, kaolin) match the torch build.

### Component notes / known gotchas
- `flash-attn`, `spconv`, `kaolin` need their wheel to match the env's torch+CUDA; mismatches
  are the most common failure. Prefer the versions `setup.sh` selects.
- `diffoctreerast` and `mip-gaussian` build from source (need `nvcc` + matching gcc).
- `nvdiffrast` needs a GL/EGL context; for headless render use EGL.

## Running SLAT-Studio in that env

```bash
conda activate trellis
# make both packages importable:
export PYTHONPATH="$PWD/third_party/TRELLIS:$PWD:$PYTHONPATH"
python scripts/smoke_test.py
```

## Smoke test

`scripts/smoke_test.py` checks the GPU + that `trellis` imports. The full pipeline smoke
(download `TRELLIS-image-large`, run image→3D) is gated behind `--full` since it pulls
multi-GB weights from Hugging Face.
