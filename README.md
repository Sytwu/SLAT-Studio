# SLAT-Studio

Downstream 3D tasks built on top of [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS)
and its **Structured LATent (SLAT)** representation.

Where TRELLIS does text/image → 3D, SLAT-Studio takes an **existing 3D asset (3DGS / mesh)
+ a text prompt** and performs:

- **3D Editing** — region / local edits (RePaint-style masked SLAT sampling)
- **Style Transfer / Material Alteration** — restyle appearance while preserving geometry
- **Interpolation & Morphing** — blend two assets in SLAT space

TRELLIS is vendored **unmodified** as a git submodule under `third_party/TRELLIS`; all new
logic lives in the `slat_studio` package and only *imports* `trellis.*`. We never edit the
submodule — custom sampling is done by subclassing TRELLIS classes.

## Status

Phase 0 — scaffolding & environment bring-up. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
and the design plan for the roadmap.

## Layout

```
third_party/TRELLIS/   # git submodule, UNMODIFIED (pinned commit)
slat_studio/
  bridge/    # external 3DGS/mesh -> SLAT (render -> DINOv2 -> voxelize -> VAE encode)
  editing/   # region edit (RePaint sampler subclass), detail variation
  style/     # text-conditioned restyle / appearance & material
  morph/     # SLAT structure alignment + interpolation
  io/        # load/save SLAT (.npz), 3DGS (.ply), mesh; GLB/OBJ export
  pipelines/ # high-level user-facing entry points
  samplers/  # custom flow samplers (subclass trellis samplers; no core edits)
examples/    # runnable demos per task
configs/
```

## Setup

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). In short:

```bash
git clone --recurse-submodules <this-repo>
cd slat-studio
# build the TRELLIS env (CUDA extensions):
cd third_party/TRELLIS && . ./setup.sh --new-env --basic --xformers --flash-attn \
    --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

Requires Linux + NVIDIA GPU (16GB+ VRAM), CUDA toolkit, and conda.
