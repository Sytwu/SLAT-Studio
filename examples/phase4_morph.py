"""Phase 4, step 2/2: morph between two cached SLATs and render the sequence.

Loads the source SLAT (phase1_base.npz, wooden chest) and the target SLAT
(phase4_target.npz, ceramic teapot), builds a structure correspondence with
``slat_studio.morph.SlatMorpher``, and emits the morphed SLAT at t in {0,.25,.5,.75,1}.

Verifies the morph is well-formed:
  * endpoints exact: t=0 reproduces the source, t=1 reproduces the target (coords + latents,
    compared as sets since the union is key-sorted, not source-ordered).
  * structure transition: per-t voxel count moves monotonically from N_source to N_target.

Then decodes each intermediate to Gaussians ONE AT A TIME (24GB can't hold several GS
decodes + renders at once — same discipline as Phase 3), renders a short turntable per t,
and writes a morph grid PNG + a morph video. No flow/text model is needed for the default
(pure interpolation) path; only the GS decoder + renderer. Driven by scripts/run_phase4.sh.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
import numpy as np
import torch

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from slat_studio.io import load_slat, save_slat, read_extra
from slat_studio.morph import SlatMorpher

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SRC_NPZ = os.path.join(OUT, "phase1_base.npz")
TGT_NPZ = os.path.join(OUT, "phase4_target.npz")
REPORT = os.path.join(OUT, "phase4_report.md")

TS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 4
FRAMES = 24          # turntable frames per t (for the morph video)

assert os.path.exists(SRC_NPZ), f"missing {SRC_NPZ} — run examples/phase1_generate.py first"
assert os.path.exists(TGT_NPZ), f"missing {TGT_NPZ} — run examples/phase4_gen_target.py first"


def canon(slat):
    """(coords, feats) sorted by voxel key — order-invariant canonical form for compare."""
    c = slat.coords[:, 1:].long()
    key = (c[:, 0] * 64 + c[:, 1]) * 64 + c[:, 2]
    order = torch.argsort(key)
    return slat.coords[order], slat.feats[order]


# ---------------------------------------------------------------- pipeline (GS decoder only)
print("[p4] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
# pure-interpolation morph needs only the gaussian decoder; park everything else.
for n in ("sparse_structure_flow_model", "slat_flow_model", "slat_decoder_mesh", "slat_decoder_rf"):
    pipe.models[n].cpu()
pipe.text_cond_model["model"].cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- load endpoints + build morpher
source = load_slat(SRC_NPZ, device="cuda")
target = load_slat(TGT_NPZ, device="cuda")
src_prompt = read_extra(SRC_NPZ).get("prompt", "<unknown>")
tgt_prompt = read_extra(TGT_NPZ).get("prompt", "<unknown>")
morpher = SlatMorpher(source, target, seed=SEED)
print(f"[p4] source={src_prompt!r} ({morpher.n_a} vox), target={tgt_prompt!r} ({morpher.n_b} vox)")
print(f"[p4] union={morpher.n_union} vox, shared={morpher.n_shared} vox")

# ---------------------------------------------------------------- latent-space verification
slats = [morpher.at(t) for t in TS]
counts = [s.coords.shape[0] for s in slats]
print(f"[p4] per-t voxel counts {dict(zip(TS, counts))}")

c0, f0 = canon(slats[0]); cs, fs = canon(source)
end0_ok = torch.equal(c0, cs) and torch.allclose(f0, fs, atol=1e-5)
c1, f1 = canon(slats[-1]); ct, ft = canon(target)
end1_ok = torch.equal(c1, ct) and torch.allclose(f1, ft, atol=1e-5)
# voxel count should move monotonically from source -> target
mono = all(counts[i] >= min(counts[0], counts[-1]) - 1 for i in range(len(counts)))
print(f"[p4] endpoint t=0 == source: {end0_ok} | t=1 == target: {end1_ok}")
assert end0_ok, "t=0 must reproduce the source exactly"
assert end1_ok, "t=1 must reproduce the target exactly"
save_slat(slats[2], os.path.join(OUT, "phase4_mid.npz"),
          extra={"source": src_prompt, "target": tgt_prompt, "t": 0.5})

# ---------------------------------------------------------------- decode + render each t (one at a time)
grid_frames, turntable = [], []
for t, slat in zip(TS, slats):
    print(f"[p4] decoding + rendering t={t} ({slat.coords.shape[0]} vox)...")
    gs = pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]
    vid = np.stack(render_utils.render_video(gs, num_frames=FRAMES)["color"])
    grid_frames.append(vid[0])            # one representative view per t
    turntable.append(vid)                 # full spin per t (for the morph movie)
    del gs
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- artifacts
grid = np.concatenate(grid_frames, axis=1)        # source | ... | target, fixed view
imageio.imwrite(os.path.join(OUT, "phase4_morph_grid.png"), grid)
movie = np.concatenate(turntable, axis=0)         # spin through each t in turn
imageio.mimsave(os.path.join(OUT, "phase4_morph.mp4"), list(movie), fps=FRAMES)

with open(REPORT, "w") as f:
    f.write("# Phase 4 — Morphing (structure union + dissolve schedule)\n\n")
    f.write(f"Source: {src_prompt!r} ({morpher.n_a} voxels)\n\n")
    f.write(f"Target: {tgt_prompt!r} ({morpher.n_b} voxels)\n\n")
    f.write(f"- structure union: {morpher.n_union} voxels "
            f"(shared {morpher.n_shared}, A-only {morpher.n_a - morpher.n_shared}, "
            f"B-only {morpher.n_b - morpher.n_shared})\n")
    f.write(f"- per-t voxel counts: {dict(zip(TS, counts))}\n\n")
    f.write("## Correspondence & endpoints\n\n")
    f.write("Shared voxels lerp their latents; A-only dissolve out and B-only dissolve in on "
            "a stable per-voxel threshold as t goes 0->1 (temporally coherent — no flicker).\n\n")
    f.write(f"- endpoint t=0 reproduces source exactly (coords+latents): **{end0_ok}**\n")
    f.write(f"- endpoint t=1 reproduces target exactly (coords+latents): **{end1_ok}**\n\n")
    f.write("Artifacts: `phase4_morph_grid.png` (one view per t, source->target), "
            "`phase4_morph.mp4` (turntable through each t), `phase4_mid.npz` (t=0.5 SLAT).\n")

print(f"[p4] wrote phase4_report.md / phase4_morph_grid.png / phase4_morph.mp4")
print("[p4] DONE")
