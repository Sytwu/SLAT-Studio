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

## Status: Phase 0 ✅ + Phase 1 ✅ + Phase 2 ✅ + Phase 3 ✅ (region editing verified)

| Item | State |
|---|---|
| TRELLIS git submodule `third_party/TRELLIS` @ `442aa1e` (+ nested flexicubes) | ✅ unmodified |
| conda env `trellis` (torch 2.4.0/cu118, xformers, spconv, kaolin, nvdiffrast, diffoctreerast, diff-gaussian) | ✅ built |
| Import smoke test (`spconv` + `xformers`) | ✅ PASS |
| Full image→3D capstone (478k splats, render, GLB, visually verified) | ✅ PASS |
| **Phase 1**: `slat_studio.io` SLAT `.npz` save/load | ✅ bit-exact round-trip PASS |
| **Phase 1**: `slat_studio.pipelines.text_to_slat` (text→3D returning the SLAT) | ✅ |
| **Phase 1**: `slat_studio.style.restyle` (freeze structure, re-prompt stage-2) | ✅ structure identical, appearance changed, visually verified |
| **Phase 2**: `slat_studio.bridge` (external 3DGS → SLAT: render→voxelize→DINOv2→VAE encode) | ✅ round-trip PSNR 31.3 dB / SSIM 0.94, structure IoU 0.75 |
| **Phase 3**: `slat_studio.samplers.RepaintFlowSampler` + `slat_studio.editing.edit_region` (masked region edit) | ✅ out-box latents bit-exact, decode-space change 10.4× concentrated in box |
| transformers 5.11.0 vs text pipeline (CLIP encoder) | ✅ works — NO pin needed |
| flash-attn | ❌ intentionally skipped; use `ATTN_BACKEND=xformers` |
| git | ✅ Phase 0 pushed (`origin` = github Sytwu/SLAT-Studio, private) |

Phase 0 artifacts: `outputs/smoke.glb`, `outputs/smoke_gs.mp4`, `outputs/smoke_frame.png`.
Phase 1 artifacts: `outputs/phase1_base.{npz,mp4}`, `outputs/phase1_restyled.mp4`,
`outputs/phase1_compare.png` (wooden chest → gold/emerald chest, same geometry).
Phase 2 artifacts: `outputs/phase2_bridge.{npz,mp4}`, `outputs/phase2_compare.png`
(input asset | bridge round-trip, brightest view), `outputs/phase2_report.md` (the fidelity report).
Phase 3 artifacts: `outputs/phase3_edited.npz`, `outputs/phase3_edit.mp4`, `outputs/phase3_compare.png`
(source | edited | |diff|, brightest view: wooden chest with a lava-edited top half), `outputs/phase3_report.md`.

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

### Phase 2 — encoding bridge: how to run / key facts
- `bash scripts/run_phase2.sh` (ONE process) → `examples/phase2_bridge.py`. Needs
  `outputs/phase1_base.npz` (run Phase 1 first). Latest run: **PSNR 31.3 dB, SSIM 0.94**
  (150 views), **structure IoU 0.75**, latent cos 0.95 on shared voxels.
- **Bridge = `slat_studio.bridge.gaussian_to_slat`** = render multiview (TRELLIS gaussian
  renderer) → voxelize Gaussian **means** at 64³ → DINOv2 `dinov2_vitl14_reg` patch tokens →
  per-voxel mean over views → frozen SLAT VAE encoder. Mirrors `dataset_toolkits/{extract_feature,
  encode_latent}.py` but fully in-memory; reuses only public TRELLIS/utils3d APIs.
- **Why no Blender:** `dataset_toolkits/render.py` (Blender) + `voxelize.py` (open3d on a
  **mesh**) can't take a 3DGS. We render with `render_utils.render_multiview` (returns renders
  **and** matching CV cameras) and derive occupancy from Gaussian means — `floor((mean+0.5)*64)`
  recovers voxel indices because TRELLIS Gaussians use `aabb=[-0.5,-0.5,-0.5,1,1,1]` and decode
  clustered at voxel centers. Rendering + `utils3d.project_cv` share one camera set (no
  Blender→CV flip).
- **Encoder:** `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16` — the SAME SLAT
  VAE the GS decoder (`slat_dec_gs_swin8_B_64l8gs32`, shared by image- & text- models) consumes,
  so encode(image-large enc) → decode(text-xlarge dec) is consistent. Not in the cached image
  pipeline by default; pre-fetched via `huggingface_hub.hf_hub_download`.
- **Memory:** bridge needs only GS decoder + DINOv2 + encoder (NO flow models), loaded/freed in
  sequence → fits one process on 24GB (unlike Phase 1).
- **Stand-in input:** uses a TRELLIS-native asset as fake "external" 3DGS so we have ground-truth
  SLAT + coords to score against. Swap in a real external `.ply` later (decode-to-Gaussian step
  becomes a `.ply` load; everything downstream is unchanged).
- **Bridge over-counts voxels** (42k vs 32k native): Gaussian per-voxel offsets push some means
  into neighboring cells, so IoU caps ~0.75. A density/coverage threshold could tighten it later.

### Phase 3 — region editing: how to run / key facts
- `bash scripts/run_phase3.sh` (ONE process) → `examples/phase3_edit.py`. Needs
  `outputs/phase1_base.npz`. Latest run: structure preserved (coords identical), **out-of-box
  latents bit-exact** (max |Δ|=0.0), in-box latents changed (mean |Δ|=4.17), and per-Gaussian
  |Δcolor| **10.4× higher in-box than out-box** (wooden chest → lava-textured top half).
- **Sampler = `slat_studio.samplers.RepaintFlowSampler`** — subclass of TRELLIS's
  `FlowEulerGuidanceIntervalSampler` (so CFG + guidance-interval are reused verbatim). Overrides
  only `sample`: at each Euler step the unknown (in-box) voxels follow the model while the known
  (out-box) voxels are re-noised from the source x_0 — RePaint adapted to flow matching
  (`x_t = (1-t)·x_0 + (σ_min+(1-σ_min)t)·ε`). Optional `resample>1` does RePaint jump-back to
  harmonize the boundary (default 1 = off). Nothing in the submodule is edited.
- **Entry point = `slat_studio.editing.edit_region(pipe, source_slat, bbox, prompt)`** —
  normalizes source latents with `pipe.slat_normalization`, builds noise on the SAME coords via
  `.replace()` (so the per-voxel mask stays row-aligned), runs the sampler, denormalizes, then
  **hard-composites the source latents back outside the mask** → out-of-box preservation is
  bit-exact (not just approximate). `bbox` is voxel indices `(x0,y0,z0,x1,y1,z1)`; use
  `editing.normalized_to_voxel_bbox` for a box in `[-0.5,0.5]`.
- **Per-voxel masking trick:** SparseTensor `*` with an `[N,1]` tensor — the `[1,C]` batch-
  broadcast path raises and falls back to a plain elementwise mul → per-voxel gating over the 8
  channels. `.replace(feats)` keeps coords/order, so mask/known/noise/x_t all stay aligned.
- **Localization is not perfect outside the box** (out-box |Δcolor|=0.032, not 0): the GS decoder
  is a *global* sparse transformer, so freezing the out-box latents exactly still lets a little
  appearance change bleed across voxel attention. The latent freeze is exact; the *decoded* edit
  is concentrated (10.4×) but not perfectly contained — report both honestly.
- **Memory:** edit = one stage-2 pass + GS decoder + CLIP ≈ restyle footprint. The demo parks
  `slat_flow_model`+CLIP after editing and decodes source/edited **one at a time** (two live GS
  decodes OOM a 24GB card).

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
- `run_phase2.sh` — Phase 2 encoding-bridge fidelity report (one process; needs phase1_base.npz).
- `run_phase3.sh` — Phase 3 region-editing demo (one process; needs phase1_base.npz).

## TRELLIS import surface (confirmed, for future phases)
- `from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline`
- samplers: `trellis.pipelines.samplers.{FlowEulerSampler, FlowEulerCfgSampler, FlowEulerGuidanceIntervalSampler}` → subclass these for RePaint region-edit.
- encoding bridge will reuse `third_party/TRELLIS/dataset_toolkits/{render,voxelize,extract_feature,encode_latent}.py`.
- decode/render/export utils: `trellis.utils.{render_utils, postprocessing_utils}`.
- `render_utils.render_multiview(sample, resolution, nviews)` → `(colors, extrinsics,
  intrinsics)` with CV cameras that `utils3d.torch.project_cv` consumes directly (used by the bridge).
- SLAT VAE encoder: `trellis.models.from_pretrained("microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16")`.

## Next: Phase 4 — morphing (not started)
Interpolate between two SLAT assets (ref MorphAny3D). The hard part is sparse-voxel
correspondence: two SLATs have different `L`, different `{p_i}`. Sketch:
1. `slat_studio/morph/` — align two structures (union/nearest-voxel correspondence), then
   interpolate `{p_i}` (occupancy) + `{z_i}` (latents) across `t∈{0,.25,.5,.75,1}`.
2. Optionally re-run a few stage-2 steps at each `t` to clean up blended latents (like a light
   RePaint pass — can reuse `RepaintFlowSampler`).
3. Verify: intermediate frames render as a smooth source→target sequence.

Then Phase 5 (stretch) = true PBR fields (SLAT-Phys-style decoder) / inpainting.

## Open items to decide with user
- Phase 2 used a TRELLIS-native asset as a stand-in external input. Swapping in a **real
  external 3DGS `.ply`** is a small change (replace the decode-to-Gaussian step with a `.ply`
  loader); provide sample `.ply` assets when ready to validate the true external path.
- Bridge IoU caps ~0.75 from voxel over-counting — revisit with a Gaussian density/coverage
  threshold if structure fidelity matters for downstream edits.
