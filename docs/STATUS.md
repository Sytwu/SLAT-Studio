# SLAT-Studio — Project Status / Handoff

_Last updated: 2026-06-12. Read this first when resuming._

## What this project is
A repo of **downstream 3D tasks on top of Microsoft TRELLIS** (SLAT = structured latents).
Goal: take an existing 3DGS/mesh **+ text prompt** and do **3D editing, style transfer /
material alteration, interpolation/morphing** — beyond TRELLIS's text/image→3D.

- Approved design plan: `/home/cookies/.claude/plans/trellis-paper-immutable-turtle.md`
- Architecture rule: TRELLIS is vendored **unmodified** as a git submodule at
  `third_party/TRELLIS`; the `slat_studio` package only `import trellis.*`. Custom sampling
  is done by **subclassing** TRELLIS classes — never edit the submodule.

## Status: Phase 0 COMPLETE ✅ (env + smoke + full pipeline verified)

| Item | State |
|---|---|
| Repo + package skeleton (`slat_studio/{bridge,editing,style,morph,io,pipelines,samplers}`) | ✅ stubs only, no algorithms yet |
| TRELLIS git submodule `third_party/TRELLIS` @ `442aa1e` (+ nested flexicubes) | ✅ unmodified |
| conda env `trellis` (torch 2.4.0/cu118, xformers, spconv, kaolin, nvdiffrast, diffoctreerast, diff-gaussian) | ✅ built |
| Import smoke test (`spconv` + `xformers`) | ✅ PASS |
| Full image→3D capstone (478k splats, render, GLB, visually verified) | ✅ PASS |
| git commit | ❌ NOT committed yet (waiting on user) |
| flash-attn | ❌ intentionally skipped; use `ATTN_BACKEND=xformers` |

Artifacts: `outputs/smoke.glb`, `outputs/smoke_gs.mp4`, `outputs/smoke_frame.png`.

## Environment — how to run (CRITICAL)
conda env lives at `/home/cookies/miniconda3/envs/trellis` (torch 2.4.0 + cu118).
```bash
export PATH=/home/cookies/miniconda3/envs/trellis/bin:/usr/local/cuda-11.8/bin:$PATH
export PYTHONPATH=/project2/cookies/SLAT-Studio/third_party/TRELLIS
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
```
Then e.g. `python scripts/smoke_test.py` or `bash scripts/run_full_smoke.sh`.

## Environment gotchas already solved (don't re-hit these)
1. **Wrong-env activation.** Two conda installs (`/home/.../miniconda3` = base,
   `/project2/.../miniconda3`), and the Claude shell snapshot hardcodes `coz/bin` at the
   FRONT of PATH. `conda activate` does NOT make bare `python`/`pip` use the target env →
   it silently installs into `coz`. Fix used everywhere: set `PATH=<env>/bin:...` explicitly
   and a hard guard asserting `sys.prefix == env` before any install. (coz was NOT polluted.)
2. **`set -u` vs conda hooks.** conda deactivate hooks use unbound vars; `set -u` makes them
   fatal. Build scripts use `set -o pipefail` only (no `-u`).
3. **pip build isolation.** The 3 source extensions import torch at build time → need
   `pip install --no-build-isolation`.
4. **gcc too new.** CUDA 11.8 nvcc rejects gcc > 11 (host is gcc 13). Build source extensions
   with `CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11` and
   `NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"`, `TORCH_CUDA_ARCH_LIST=8.9` (RTX 4090 Ada).

## Scripts (all in `scripts/`)
- `build_env.sh` — create env + wheel deps via TRELLIS setup.sh (no --new-env, no flash-attn).
- `build_ext_noiso.sh` — rebuild nvdiffrast/diffoctreerast/diff-gaussian (gcc-11, no-build-isolation).
- `smoke_test.py` — cheap import check (`--full` runs image→3D).
- `full_smoke.py` + `run_full_smoke.sh` — full image→3D capstone.

## TRELLIS import surface (confirmed, for future phases)
- `from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline`
- samplers: `trellis.pipelines.samplers.{FlowEulerSampler, FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler}` → subclass these for RePaint region-edit.
- encoding bridge will reuse `third_party/TRELLIS/dataset_toolkits/{render,voxelize,extract_feature,encode_latent}.py`.
- decode/render/export utils: `trellis.utils.{render_utils, postprocessing_utils}`.

## Next: Phase 1 (not started)
SLAT I/O + native restyle:
1. `slat_studio/io/` — save/load SLAT as `.npz` (coords {p_i} + feats {z_i}); cache the SLAT
   produced at generation time (high-fidelity native path).
2. `slat_studio/style/` — native restyle: freeze structure, re-run stage-2 with a new text prompt.
   - **RISK:** restyle needs the TEXT pipeline / CLIP text encoder, but env has
     **transformers 5.11.0** (very new). Verify `TrellisTextTo3DPipeline.from_pretrained(...)`
     loads; if it breaks, pin transformers to a TRELLIS-era version (~4.46) in the trellis env.
Then Phase 2 = encoding bridge (external 3DGS → SLAT round-trip fidelity report).

## Open items to decide with user
- Do the initial git commit? (Phase 0 scaffold + submodule + scripts.)
- Proceed to Phase 1 now?
