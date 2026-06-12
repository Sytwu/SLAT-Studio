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

## Status: Phase 0 ✅ + Phase 1 ✅ (native restyle + SLAT I/O verified)

| Item | State |
|---|---|
| TRELLIS git submodule `third_party/TRELLIS` @ `442aa1e` (+ nested flexicubes) | ✅ unmodified |
| conda env `trellis` (torch 2.4.0/cu118, xformers, spconv, kaolin, nvdiffrast, diffoctreerast, diff-gaussian) | ✅ built |
| Import smoke test (`spconv` + `xformers`) | ✅ PASS |
| Full image→3D capstone (478k splats, render, GLB, visually verified) | ✅ PASS |
| **Phase 1**: `slat_studio.io` SLAT `.npz` save/load | ✅ bit-exact round-trip PASS |
| **Phase 1**: `slat_studio.pipelines.text_to_slat` (text→3D returning the SLAT) | ✅ |
| **Phase 1**: `slat_studio.style.restyle` (freeze structure, re-prompt stage-2) | ✅ structure identical, appearance changed, visually verified |
| transformers 5.11.0 vs text pipeline (CLIP encoder) | ✅ works — NO pin needed |
| flash-attn | ❌ intentionally skipped; use `ATTN_BACKEND=xformers` |
| git | ✅ Phase 0 pushed (`origin` = github Sytwu/SLAT-Studio, private) |

Phase 0 artifacts: `outputs/smoke.glb`, `outputs/smoke_gs.mp4`, `outputs/smoke_frame.png`.
Phase 1 artifacts: `outputs/phase1_base.{npz,mp4}`, `outputs/phase1_restyled.mp4`,
`outputs/phase1_compare.png` (wooden chest → gold/emerald chest, same geometry).

### Phase 1 — how to run / key facts
- `bash scripts/run_phase1.sh` runs TWO processes: `examples/phase1_generate.py` (generate +
  cache SLAT + render base) then `examples/phase1_restyle.py` (load cached SLAT + restyle).
- **Why two processes:** text-xlarge (both flow models + 3 decoders + CLIP) + a full
  generation + the diff-gaussian renderer's cached buffers accumulate past 24GB if generation
  and restyle share one process. One generation per process fits; `.cpu()`+`empty_cache`
  within a process does NOT reliably reclaim it.
- **Restyle = `sample_slat(new_cond, cached_coords)` + `decode_slat`** — pure composition of
  TRELLIS public methods; structure (`coords`) is reused exactly, no voxelization round-trip.
- **GPU-offload gotcha:** `pipeline.device` returns the *first* model's device and
  `sample_slat` puts its noise there. Never `.cpu()` the first model (`sparse_structure_decoder`)
  or the restyle noise lands on CPU while `slat_flow_model` is on CUDA → device mismatch.
- Cheap I/O-only check (no weights): `python scripts/test_slat_io.py`.
- `scripts/run_full_smoke.py` (image→3D) is the GLB/mesh path; needs the heavy mesh decoder.

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
- `test_slat_io.py` — cheap SLAT `.npz` round-trip (synthetic SparseTensor, no weights).
- `run_phase1.sh` — Phase 1 capstone (generate → cache → restyle), two processes.

## TRELLIS import surface (confirmed, for future phases)
- `from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline`
- samplers: `trellis.pipelines.samplers.{FlowEulerSampler, FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler}` → subclass these for RePaint region-edit.
- encoding bridge will reuse `third_party/TRELLIS/dataset_toolkits/{render,voxelize,extract_feature,encode_latent}.py`.
- decode/render/export utils: `trellis.utils.{render_utils, postprocessing_utils}`.

## Next: Phase 2 — encoding bridge (not started)
External 3DGS → SLAT, the one genuinely new core component:
1. `slat_studio/bridge/` — render input 3DGS multiview → DINOv2 features → voxel occupancy
   (64³) → SLAT VAE encode. Reuse `third_party/TRELLIS/dataset_toolkits/{render,voxelize,
   extract_feature,encode_latent}.py`.
2. Deliver a **round-trip fidelity report**: external 3DGS → SLAT → decode → 3DGS, PSNR/LPIPS.
   If round-trip is poor, downstream edits inherit the error — this is the first experiment.

Then Phase 3 = region editing (RePaint sampler subclass of `FlowEulerSampler` + bbox mask),
Phase 4 = morphing, Phase 5 (stretch) = true PBR fields / inpainting.

## Open items to decide with user
- Phase 2 (encoding bridge) needs sample external 3DGS assets (.ply) to test on — which?
