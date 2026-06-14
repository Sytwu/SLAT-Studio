"""SLAT-Studio Gradio app — interactive front-end for the downstream 3D tasks.

One tab per capability (Generate / Restyle / Edit / Morph / Inpaint / Bridge). Every task
runs on a single, lazily-loaded ``TrellisTextTo3DPipeline`` and exchanges SLATs through a
**file library**: results are saved as ``outputs/*.npz`` and picked up by other tabs from a
dropdown. This mirrors the existing ``phaseN_*.npz`` workflow and survives restarts.

Results are shown as **interactive 3D gaussian splats** (``gr.Model3D``): each result is
decoded to gaussians and saved as ``outputs/<stem>.ply`` which the viewer renders as a
draggable/rotatable splat — no turntable video. The Edit and Inpaint tabs add a middle
column that previews the target region **on the asset itself**: the real splat with the
selected region tinted (🟩 green = will change, for Edit; 🟥 red = carve+regrow, for Inpaint).

This file only *uses* the public ``slat_studio`` API (which in turn only imports ``trellis``).
The vendored ``third_party/TRELLIS`` submodule is never touched.

Run it with ``scripts/run_app.sh`` (sets the env vars + PYTHONPATH the pipeline needs).

Memory note (24GB cards): the pipeline is kept resident with the unused mesh/radiance-field
decoders parked on CPU; only the gaussian path is decoded, one SLAT at a time, then freed.
"""
import os

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import glob
import traceback

import numpy as np
import torch
import gradio as gr

from trellis.pipelines import TrellisTextTo3DPipeline

from slat_studio.io import load_slat, save_slat
from slat_studio.pipelines import text_to_slat
from slat_studio.style import restyle
from slat_studio.editing import edit_region
from slat_studio.morph import SlatMorpher
from slat_studio.inpainting import inpaint_slat, carve_hole, load_ss_encoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

PRETRAINED = "microsoft/TRELLIS-text-xlarge"
GRID = 64
VIEWER_BG = [0.10, 0.10, 0.12, 1.0]   # gr.Model3D clear colour (dark, so splats pop)


# ----------------------------------------------------------------------------- lazy singletons
# Loaded on first use so the UI comes up instantly and the heavy models are only paid for once.
_PIPE = None
_SS_ENCODER = None
_BRIDGE = None  # (dinov2, slat_encoder)


def get_pipe():
    """Load the text pipeline once; park the mesh/RF decoders (we only render gaussians)."""
    global _PIPE
    if _PIPE is None:
        _PIPE = TrellisTextTo3DPipeline.from_pretrained(PRETRAINED)
        _PIPE.cuda()
        # Keep sparse_structure_decoder on GPU (it defines pipe.device for the noise). The two
        # slat decoders we never use are parked on CPU to leave room for flows + a GS decode.
        for n in ("slat_decoder_mesh", "slat_decoder_rf"):
            if n in _PIPE.models:
                _PIPE.models[n].cpu()
        torch.cuda.empty_cache()
    return _PIPE


def get_ss_encoder():
    """Sparse-structure VAE encoder used only by the Inpaint tab (loaded separately)."""
    global _SS_ENCODER
    if _SS_ENCODER is None:
        _SS_ENCODER = load_ss_encoder(device="cuda")
    return _SS_ENCODER


def get_bridge():
    """DINOv2 + frozen SLAT VAE encoder for the external-3DGS bridge (Bridge tab)."""
    global _BRIDGE
    if _BRIDGE is None:
        from slat_studio.bridge import load_dinov2, load_slat_encoder
        _BRIDGE = (load_dinov2(device="cuda"), load_slat_encoder(device="cuda"))
    return _BRIDGE


# ----------------------------------------------------------------------------- SLAT file library
def list_slats():
    """All ``outputs/*.npz`` basenames (newest first), the choices for every source dropdown."""
    paths = sorted(glob.glob(os.path.join(OUT, "*.npz")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(p) for p in paths]


def _npz_path(name: str) -> str:
    if not name:
        raise gr.Error("No SLAT selected — generate one first or pick one from the dropdown.")
    return os.path.join(OUT, name if name.endswith(".npz") else name + ".npz")


def _safe_stem(name: str) -> str:
    """Sanitize a user-typed 'save as' name into a bare .npz stem under outputs/."""
    stem = os.path.splitext(os.path.basename((name or "").strip()))[0]
    if not stem:
        stem = "untitled"
    return "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "untitled"


# ----------------------------------------------------------------------------- decode / render
def decode_gs(slat):
    return get_pipe().decode_slat(slat, ["gaussian"])["gaussian"][0]


def render_result(slat, stem: str):
    """Decode a SLAT to gaussians and save the full splat to ``outputs/<stem>.ply``; free GPU.
    Returns the ``.ply`` path — what ``gr.Model3D`` shows as a rotatable gaussian splat."""
    gs = decode_gs(slat)
    path = os.path.join(OUT, f"{stem}.ply")
    gs.save_ply(path)
    del gs
    torch.cuda.empty_cache()
    return path


def voxel_extent(slat):
    """(min[xyz], max[xyz]) of the occupied voxels, as plain int lists."""
    c = slat.coords[:, 1:].long()
    return c.min(0).values.tolist(), c.max(0).values.tolist()


def _err(prefix: str, e: Exception) -> str:
    traceback.print_exc()
    return f"❌ {prefix}: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- region preview
# The Edit/Inpaint middle column previews the region ON THE ASSET: decode the source to its
# gaussian splat and tint the gaussians inside the bbox toward green (Edit) / red (Inpaint), so
# you see exactly where the edit will land on the real object — same renderer/frame as the result
# viewer, so there is no glb-vs-splat axis mismatch. A gaussian at world p (in [-0.5,0.5], the GS
# decoder's aabb) sits in voxel floor((p+0.5)*64), the same convention as the bbox mask.
SH_C0 = 0.28209479177387814              # SH band-0 factor: rgb = 0.5 + SH_C0 * f_dc
EDIT_TINT = (0.15, 0.85, 0.25)           # green — region that will change (Edit)
HOLE_TINT = (0.95, 0.20, 0.20)           # red   — region carved + regrown (Inpaint)
TINT_ALPHA = 0.55                        # blend strength: how hard to pull region gaussians to tint
_PREVIEW_GS = {"key": None}              # one-entry decode cache: source -> CPU gaussians + base color


def _preview_gaussians(source):
    """Decode the source SLAT to gaussians once and stash them on CPU (keyed by the .npz mtime),
    so flipping between region presets on the same asset re-tints instead of re-decoding/holding VRAM."""
    path = _npz_path(source)
    key = (source, os.path.getmtime(path))
    if _PREVIEW_GS.get("key") != key:
        src = load_slat(path, device="cuda")
        gs = decode_gs(src)
        for a in ("_xyz", "_features_dc", "_opacity", "_scaling", "_rotation"):
            setattr(gs, a, getattr(gs, a).detach().cpu())
        # save_ply reads aabb + the *_bias buffers too; move them to CPU so it runs off-GPU
        gs.aabb = gs.aabb.cpu()
        gs.scale_bias = gs.scale_bias.cpu()
        gs.rots_bias = gs.rots_bias.cpu()
        gs.opacity_bias = gs.opacity_bias.cpu()
        gs.device = "cpu"
        del src
        torch.cuda.empty_cache()
        _PREVIEW_GS.clear()
        _PREVIEW_GS.update(key=key, gs=gs, base_fdc=gs._features_dc.clone())
    return _PREVIEW_GS["gs"], _PREVIEW_GS["base_fdc"]


def region_tinted_ply(source, bbox, tint, stem: str):
    """Save ``outputs/<stem>.ply``: the source splat with in-bbox gaussians blended toward ``tint``.
    Returns ``(path, n_in)``. Reuses the cached CPU gaussians, so only the color is recomputed."""
    gs, base_fdc = _preview_gaussians(source)
    vox = torch.floor((gs.get_xyz + 0.5) * GRID).long()      # gaussian world pos -> voxel (x,y,z)
    x0, y0, z0, x1, y1, z1 = bbox
    inb = ((vox[:, 0] >= x0) & (vox[:, 0] < x1) &
           (vox[:, 1] >= y0) & (vox[:, 1] < y1) &
           (vox[:, 2] >= z0) & (vox[:, 2] < z1))
    fdc = base_fdc.clone()
    rgb = 0.5 + SH_C0 * fdc[inb, 0, :]                       # current colour of in-region gaussians
    tint_t = torch.tensor(tint, dtype=rgb.dtype)
    rgb = (1 - TINT_ALPHA) * rgb + TINT_ALPHA * tint_t       # blend toward the tint (keep some original)
    fdc[inb, 0, :] = (rgb - 0.5) / SH_C0
    gs._features_dc = fdc
    path = os.path.join(OUT, f"{stem}.ply")
    gs.save_ply(path)
    gs._features_dc = base_fdc                               # restore the cache's untinted base
    return path, int(inb.sum().item())


# ============================================================================= TAB CALLBACKS
def do_generate(prompt, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a prompt."
    try:
        slat, _ = text_to_slat(get_pipe(), prompt.strip(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or prompt)
        save_slat(slat, os.path.join(OUT, stem + ".npz"),
                  extra={"prompt": prompt.strip(), "seed": int(seed), "task": "generate"})
        ply = render_result(slat, stem)
        n = slat.coords.shape[0]
        msg = f"✅ generated {n} voxels → saved **{stem}.npz**"
        return ply, msg
    except Exception as e:
        return None, _err("generate failed", e)


def do_restyle(source, prompt, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a restyle prompt.", gr.update()
    try:
        src = load_slat(_npz_path(source), device="cuda")
        slat, _ = restyle(get_pipe(), src, prompt.strip(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_restyled"))
        save_slat(slat, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "seed": int(seed),
                         "task": "restyle"})
        ply = render_result(slat, stem)
        msg = f"✅ restyled → saved **{stem}.npz** (structure kept)"
        return ply, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("restyle failed", e), gr.update()


def apply_preset(source, preset):
    """Fill the 6 bbox fields from the asset extent + a named region preset."""
    try:
        src = load_slat(_npz_path(source), device="cuda")
        mn, mx = voxel_extent(src)
        del src
        torch.cuda.empty_cache()
    except Exception:
        # fall back to full grid if the source can't be read
        mn, mx = [0, 0, 0], [GRID - 1, GRID - 1, GRID - 1]
    x0, y0, z0 = mn
    x1, y1, z1 = mx[0] + 1, mx[1] + 1, mx[2] + 1            # exclusive upper
    midx = (mn[0] + mx[0] + 1) // 2
    midy = (mn[1] + mx[1] + 1) // 2
    midz = (mn[2] + mx[2] + 1) // 2
    # TRELLIS authors assets +Z up (its orbit cameras use up=[0,0,1]), so the asset's vertical
    # axis is Z, not Y. "top/bottom" therefore slice Z; the horizontal axes are X (left/right)
    # and Y (front/back). This is what makes the picked region match the upright splat you see.
    if preset == TOP_HALF:
        z0 = midz
    elif preset == BOTTOM_HALF:
        z1 = midz
    elif preset == LEFT_HALF:
        x1 = midx
    elif preset == RIGHT_HALF:
        x0 = midx
    elif preset == FRONT_HALF:
        y0 = midy
    elif preset == BACK_HALF:
        y1 = midy
    elif preset == TOP_CORNER:
        x0, y0, z0 = midx, midy, midz
    # "full extent" leaves the full box
    return x0, y0, z0, x1, y1, z1


def apply_hole_preset(source, preset):
    """Inpaint hole = a sub-box of the asset (per-axis fractions of its extent), sized to ~1/8..1/4
    of the bounding box so the regrowth is clearly visible rather than a near-copy of the source.
    It stays anchored to one local region (not a full half like the Edit presets) so the survivors
    around it still line up and the fill blends in. ``HOLE_PRESETS`` maps a name -> per-axis
    ``(lo, hi)`` fractions in the same +Z-up / +Y-front frame as :func:`apply_preset`."""
    try:
        src = load_slat(_npz_path(source), device="cuda")
        mn, mx = voxel_extent(src)
        del src
        torch.cuda.empty_cache()
    except Exception:
        mn, mx = [0, 0, 0], [GRID - 1, GRID - 1, GRID - 1]
    fx, fy, fz = HOLE_PRESETS[preset]

    def axis(lo_i, hi_i, frac):
        span = hi_i + 1 - lo_i
        return lo_i + int(round(frac[0] * span)), lo_i + int(round(frac[1] * span))

    x0, x1 = axis(mn[0], mx[0], fx)
    y0, y1 = axis(mn[1], mx[1], fy)
    z0, z1 = axis(mn[2], mx[2], fz)
    return x0, y0, z0, x1, y1, z1


def _select_region(source, preset, tint, suffix):
    """Dropdown-driven region picker (Edit/Inpaint middle column): turn a named preset into a bbox
    AND tint that region on the source splat in one step, so picking from the dropdown is all the
    user does. The '— Select … —' sentinel (anything not in BBOX_PRESETS) clears the preview."""
    if preset not in BBOX_PRESETS:
        return (0, 0, 0, GRID, GRID, GRID, None, "Pick a region to preview it on the asset.")
    bbox = apply_preset(source, preset)
    try:
        path, n_in = region_tinted_ply(source, bbox, tint, _safe_stem(source) + suffix)
    except Exception as e:
        return (*bbox, None, _err("preview failed", e))
    return (*bbox, path, f"**{n_in}** voxels in region · bbox={bbox}")


def select_edit_region(source, preset):
    return _select_region(source, preset, EDIT_TINT, "_region")     # green = will change


def select_inpaint_region(source, preset):
    """Hole picker (Inpaint): a small local-hole preset -> bbox + red tint on the source splat."""
    if preset not in HOLE_PRESETS:
        return (0, 0, 0, GRID, GRID, GRID, None, "Pick a hole to preview it on the asset.")
    bbox = apply_hole_preset(source, preset)
    try:
        path, n_in = region_tinted_ply(source, bbox, HOLE_TINT, _safe_stem(source) + "_hole")
    except Exception as e:
        return (*bbox, None, _err("preview failed", e))
    return (*bbox, path, f"**{n_in}** voxels in hole · bbox={bbox} · ~1/8–1/4 of the asset")


def do_edit(source, prompt, x0, y0, z0, x1, y1, z1, seed, resample, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter an edit prompt.", gr.update()
    try:
        src = load_slat(_npz_path(source), device="cuda")
        bbox = tuple(int(v) for v in (x0, y0, z0, x1, y1, z1))
        edited, mask, _ = edit_region(get_pipe(), src, bbox, prompt.strip(),
                                      seed=int(seed), resample=int(resample), formats=())
        n_in = int(mask.sum().item())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_edited"))
        save_slat(edited, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "bbox": list(bbox),
                         "seed": int(seed), "task": "edit"})
        ply = render_result(edited, stem)
        msg = f"✅ edited {n_in} in-box voxels (of {src.coords.shape[0]}) → saved **{stem}.npz**"
        return ply, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("edit failed", e), gr.update()


_MORPH_CACHE = {"key": None}             # (source, prompt, seed) -> SlatMorpher (avoid re-restyling)


def _get_appearance_morpher(source, prompt, seed):
    """Build (and cache) a morpher between the source and its restyle to ``prompt``. Because restyle
    re-runs stage-2 on the SAME structure, the two endpoints share every voxel, so the morph is a
    pure latent lerp on a fixed geometry — no structure union / dissolve. The restyle (~30-60s) is
    cached so the grid + render-at-t don't pay for it twice."""
    key = (source, prompt.strip(), int(seed))
    if _MORPH_CACHE.get("key") != key:
        src = load_slat(_npz_path(source), device="cuda")
        target, _ = restyle(get_pipe(), src, prompt.strip(), seed=int(seed), formats=())
        morpher = SlatMorpher(src, target, seed=int(seed))   # identical coords -> all voxels shared
        _MORPH_CACHE.clear()
        _MORPH_CACHE.update(key=key, morpher=morpher)
    return _MORPH_CACHE["morpher"]


def do_morph_build(source, prompt, steps, seed):
    """Render the morph as N discrete frames (decode each ``.at(t)`` to its own ``.ply``) so the
    Frame slider can scrub between them with no further compute. The restyle endpoint is cached."""
    if not prompt or not prompt.strip():
        return None, "❌ Enter a target appearance prompt.", [], gr.update()
    try:
        morpher = _get_appearance_morpher(source, prompt, seed)
        n = int(steps)
        ts = np.linspace(0.0, 1.0, n).tolist()
        stem = _safe_stem(source) + "_morph"
        paths = [render_result(morpher.at(t), f"{stem}_f{i}") for i, t in enumerate(ts)]
        msg = (f"✅ built {n} frames (t=0…1, {morpher.n_union} voxels, structure shared) "
               "— drag the **Frame** slider under the viewer")
        return paths[0], msg, paths, gr.update(maximum=n - 1, value=0, label="Frame (t = 0.00)")
    except Exception as e:
        return None, _err("morph build failed", e), [], gr.update()


def show_morph_frame(idx, frames):
    """Scrub: show a precomputed morph frame and report its ``t`` in the slider label (no compute)."""
    if not frames:
        return None, gr.update()
    i = max(0, min(int(idx), len(frames) - 1))
    t = i / (len(frames) - 1) if len(frames) > 1 else 0.0
    return frames[i], gr.update(label=f"Frame (t = {t:.2f})")


def do_inpaint(source, prompt, x0, y0, z0, x1, y1, z1, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a prompt describing the asset.", gr.update(), None, None, gr.update()
    try:
        src = load_slat(_npz_path(source), device="cuda")
        bbox = tuple(int(v) for v in (x0, y0, z0, x1, y1, z1))
        holed, n_removed = carve_hole(src, bbox)                 # actually remove the hole voxels
        completed, _, info = inpaint_slat(get_pipe(), holed, bbox, prompt.strip(),
                                          get_ss_encoder(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_completed"))
        save_slat(completed, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "hole_bbox": list(bbox),
                         "seed": int(seed), "task": "inpaint"})
        done_ply = render_result(completed, stem)                # after: hole filled
        carved_ply = render_result(holed, stem + "_carved")      # before: hole still open (carved)
        msg = (f"✅ carved {n_removed} voxels, regrew {info['n_grown']} "
               f"(survivors bit-exact) → saved **{stem}.npz**")
        return (done_ply, msg, gr.update(choices=list_slats()),
                carved_ply, done_ply, gr.update(value=INPAINT_AFTER))
    except Exception as e:
        return None, _err("inpaint failed", e), gr.update(), None, None, gr.update()


def swap_inpaint_view(choice, carved_path, done_path):
    """Toggle the Inpaint result viewer between the carved (pre-fill) and inpainted splats."""
    return carved_path if choice == INPAINT_BEFORE else done_path


def do_bridge(ply_file, nviews, save_as):
    if not ply_file:
        return None, "❌ Upload a 3DGS .ply first."
    try:
        from trellis.representations import Gaussian
        from slat_studio import bridge
        dinov2, slat_enc = get_bridge()
        # aabb matches the TRELLIS GS decoder so Gaussian means map onto the 64^3 voxel grid.
        gs = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device="cuda")
        gs.load_ply(ply_file)
        slat, coords, _ = bridge.gaussian_to_slat(
            gs, slat_enc, dinov2, nviews=int(nviews), grid=GRID, device="cuda")
        del gs
        torch.cuda.empty_cache()
        stem = _safe_stem(save_as or "bridged")
        save_slat(slat, os.path.join(OUT, stem + ".npz"),
                  extra={"source_ply": os.path.basename(ply_file), "nviews": int(nviews),
                         "task": "bridge"})
        ply = render_result(slat, stem)
        msg = f"✅ bridged → {coords.shape[0]} voxels → saved **{stem}.npz**"
        return ply, msg
    except Exception as e:
        return None, _err("bridge failed", e)


# ============================================================================= UI
# TRELLIS is +Z up, so top/bottom slice Z and the horizontal halves slice X / Y. Labels carry
# the axis so the picked region matches the on-asset preview (see apply_preset).
TOP_HALF = "top half (up / +Z)"
BOTTOM_HALF = "bottom half (down / -Z)"
LEFT_HALF = "left half (-X)"
RIGHT_HALF = "right half (+X)"
FRONT_HALF = "front half (+Y)"
BACK_HALF = "back half (-Y)"
TOP_CORNER = "top-front-right corner (+X +Y +Z)"
BBOX_PRESETS = ["full extent", TOP_HALF, BOTTOM_HALF, LEFT_HALF, RIGHT_HALF,
                FRONT_HALF, BACK_HALF, TOP_CORNER]

# Inpaint carves a TRUE hole and regrows geometry, so its presets are SMALL local holes (a sub-box
# ~1/3 of the asset per axis). Carving a half makes the model hallucinate a whole new half that does
# not line up with the survivors — the cause of "weird" inpaints (a half-carve regrows e.g. a barrel
# where the lid was). Values are (lo, hi) fractions of the asset extent per axis, +Z up / +Y front.
# Per-axis (lo, hi) fractions of the asset extent. Sized so each hole is ~1/4 of the asset's
# bounding box (each span ~0.63 -> 0.63^3 ≈ 0.25): a big, clearly-visible chunk to regrow, while
# still anchored to one local region so the surviving voxels around it line up and the fill blends.
HOLE_PRESETS = {
    "top dent (lid center)":        ((0.18, 0.81), (0.18, 0.81), (0.37, 1.00)),
    "front gap (face center)":      ((0.18, 0.81), (0.37, 1.00), (0.18, 0.81)),
    "right-side gap":               ((0.37, 1.00), (0.18, 0.81), (0.18, 0.81)),
    "top-front-right corner chunk": ((0.37, 1.00), (0.37, 1.00), (0.37, 1.00)),
}

# Region/Hole dropdowns open on a non-preset sentinel so nothing previews until the user actively
# picks a region; selecting a real preset auto-tints it on the asset (see select_edit/inpaint_region).
SELECT_REGION = "— Select a region —"
SELECT_HOLE = "— Select a hole —"

# Inpaint result viewer flips between the carved (hole open, pre-fill) and the inpainted splat.
INPAINT_BEFORE = "Carved (hole open)"
INPAINT_AFTER = "Inpainted (filled)"

_src_dropdowns = []  # every source dropdown, refreshed together by the global button


def source_dropdown(label="Source SLAT"):
    dd = gr.Dropdown(choices=list_slats(), label=label, interactive=True)
    _src_dropdowns.append(dd)
    return dd


def bbox_row(visible=True):
    with gr.Row(visible=visible):
        x0 = gr.Number(label="x0", precision=0, value=0)
        y0 = gr.Number(label="y0", precision=0, value=0)
        z0 = gr.Number(label="z0", precision=0, value=0)
        x1 = gr.Number(label="x1", precision=0, value=GRID)
        y1 = gr.Number(label="y1", precision=0, value=GRID)
        z1 = gr.Number(label="z1", precision=0, value=GRID)
    return x0, y0, z0, x1, y1, z1


def result_viewer(label="Result (drag to rotate)", height=420):
    """Interactive gaussian-splat viewer fed an ``outputs/<stem>.ply``."""
    return gr.Model3D(label=label, display_mode="solid", clear_color=VIEWER_BG, height=height)


with gr.Blocks(title="SLAT-Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SLAT-Studio\n"
        "Training-free downstream 3D tasks on TRELLIS structured latents. Each tab saves its "
        "result as an `outputs/*.npz` SLAT that the other tabs can load — use **🔄 Refresh** "
        "after producing a new asset. Results show as **draggable gaussian splats**. First run "
        "of any tab loads the pipeline (~1 min)."
    )
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh SLAT library (all tabs)", scale=1)

    # --------------------------------------------------------------------- Generate
    with gr.Tabs():
        with gr.Tab("Generate"):
            gr.Markdown("Text → 3D. Produces the source SLAT for every other tab.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    g_prompt = gr.Textbox(label="Prompt", lines=2,
                                          value="A rustic wooden treasure chest with iron bands.")
                    g_save = gr.Textbox(label="Save as (name)", value="my_asset")
                    g_seed = gr.Number(label="Seed", precision=0, value=42)
                    g_run = gr.Button("Generate", variant="primary")
                    g_msg = gr.Markdown()
                with gr.Column(scale=5):
                    g_view = result_viewer()
            gen_evt = g_run.click(do_generate, [g_prompt, g_seed, g_save], [g_view, g_msg],
                                  show_progress="full")

        # --------------------------------------------------------------- Restyle
        with gr.Tab("Restyle"):
            gr.Markdown("Re-texture the **whole** asset from a new prompt; geometry preserved "
                        "(stage-2 re-run on the frozen structure).")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    r_src = source_dropdown()
                    r_prompt = gr.Textbox(label="New appearance / material prompt", lines=2,
                                          value="A treasure chest made of solid gold.")
                    r_save = gr.Textbox(label="Save as (name)", value="")
                    r_seed = gr.Number(label="Seed", precision=0, value=42)
                    r_run = gr.Button("Restyle", variant="primary")
                    r_msg = gr.Markdown()
                with gr.Column(scale=5):
                    r_view = result_viewer("Restyled (drag to rotate)")
            r_run.click(do_restyle, [r_src, r_prompt, r_seed, r_save], [r_view, r_msg, r_src],
                        show_progress="full")

        # --------------------------------------------------------------- Edit
        with gr.Tab("Edit (region)"):
            gr.Markdown("RePaint the latents **inside a voxel bbox** from a prompt; the rest stays "
                        "bit-exact. Pick a region from the dropdown — it shows on the asset tinted "
                        "🟩 green (= what will change) — then run.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):                          # inputs
                    e_src = source_dropdown()
                    e_preset = gr.Dropdown(choices=[SELECT_REGION] + BBOX_PRESETS,
                                           value=SELECT_REGION, label="Region — pick to preview")
                    e_prompt = gr.Textbox(label="In-box prompt", lines=2,
                                          value="molten glowing lava and burning embers")
                    e_save = gr.Textbox(label="Save as (name)", value="")
                    with gr.Row():
                        e_seed = gr.Number(label="Seed", precision=0, value=7)
                        e_resample = gr.Number(label="RePaint resample (1=off)", precision=0,
                                               value=1)
                    e_run = gr.Button("Edit region", variant="primary")
                    e_msg = gr.Markdown()
                    e_x0, e_y0, e_z0, e_x1, e_y1, e_z1 = bbox_row(visible=False)
                with gr.Column(scale=4):                          # region preview (on the asset)
                    e_view = result_viewer("Region preview — 🟩 will change", height=360)
                with gr.Column(scale=4):                          # result
                    e_result = result_viewer("Edited (drag to rotate)", height=360)
            e_bbox = [e_x0, e_y0, e_z0, e_x1, e_y1, e_z1]
            e_preset.change(select_edit_region, [e_src, e_preset], [*e_bbox, e_view, e_msg],
                            show_progress="full")
            e_run.click(do_edit, [e_src, e_prompt, *e_bbox, e_seed, e_resample, e_save],
                        [e_result, e_msg, e_src], show_progress="full")

        # --------------------------------------------------------------- Morph
        with gr.Tab("Morph"):
            gr.Markdown("Appearance morph on a **fixed structure**: restyle the source to a target "
                        "prompt (same voxels), then interpolate the latents. **t=0** is the source, "
                        "**t=1** the restyle, and the geometry never moves — only appearance / "
                        "material blends. Click **Build morph** to render the discrete frames, then "
                        "drag the **Frame** slider under the viewer to scrub between them.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    m_src = source_dropdown("Source (t=0)")
                    m_prompt = gr.Textbox(label="Target appearance prompt (t=1)", lines=2,
                                          value="A treasure chest made of solid gold.")
                    m_seed = gr.Number(label="Seed", precision=0, value=0)
                    m_steps = gr.Slider(2, 9, value=5, step=1,
                                        label="Frames (discrete t steps)")
                    m_build = gr.Button("Build morph", variant="primary")
                    m_msg = gr.Markdown()
                with gr.Column(scale=5):
                    m_result = result_viewer("Morph (drag to rotate)")
                    m_frame = gr.Slider(0, 4, value=0, step=1, label="Frame (t = 0.00)")
                    m_frames = gr.State([])
            m_build.click(do_morph_build, [m_src, m_prompt, m_steps, m_seed],
                          [m_result, m_msg, m_frames, m_frame], show_progress="full")
            m_frame.change(show_morph_frame, [m_frame, m_frames], [m_result, m_frame],
                           show_progress="hidden")

        # --------------------------------------------------------------- Inpaint
        with gr.Tab("Inpaint"):
            gr.Markdown("Carve a **small local hole**, then regrow **geometry + appearance** "
                        "(two-stage RePaint). Survivor voxels stay bit-exact; the fill is "
                        "prompt-conditioned. Pick a hole from the dropdown — it shows on the asset "
                        "tinted 🟥 red (= what gets carved + regrown) — then run. Flip the result "
                        "viewer between the carved hole and the filled result to compare. (Holes are "
                        "kept small on purpose: carving a whole half makes the model hallucinate a "
                        "mismatched new chunk.)")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):                          # inputs
                    i_src = source_dropdown()
                    i_preset = gr.Dropdown(choices=[SELECT_HOLE] + list(HOLE_PRESETS),
                                           value=SELECT_HOLE, label="Hole — pick to preview")
                    i_prompt = gr.Textbox(label="Asset prompt (steers the fill)", lines=2,
                                          value="A rustic wooden treasure chest with iron bands.")
                    i_save = gr.Textbox(label="Save as (name)", value="")
                    i_seed = gr.Number(label="Seed", precision=0, value=11)
                    i_run = gr.Button("Carve + inpaint", variant="primary")
                    i_msg = gr.Markdown()
                    i_x0, i_y0, i_z0, i_x1, i_y1, i_z1 = bbox_row(visible=False)
                with gr.Column(scale=4):                          # hole preview (on the asset)
                    i_view = result_viewer("Hole preview — 🟥 carve + regrow", height=360)
                with gr.Column(scale=4):                          # result (carved <-> inpainted)
                    i_result = result_viewer("Result (drag to rotate)", height=360)
                    i_toggle = gr.Radio([INPAINT_BEFORE, INPAINT_AFTER], value=INPAINT_AFTER,
                                        label="Show")
                    i_carved = gr.State()
                    i_done = gr.State()
            i_bbox = [i_x0, i_y0, i_z0, i_x1, i_y1, i_z1]
            i_preset.change(select_inpaint_region, [i_src, i_preset], [*i_bbox, i_view, i_msg],
                            show_progress="full")
            i_run.click(do_inpaint, [i_src, i_prompt, *i_bbox, i_seed, i_save],
                        [i_result, i_msg, i_src, i_carved, i_done, i_toggle], show_progress="full")
            i_toggle.change(swap_inpaint_view, [i_toggle, i_carved, i_done], i_result,
                            show_progress="hidden")

        # --------------------------------------------------------------- Bridge
        with gr.Tab("Bridge (.ply → SLAT)"):
            gr.Markdown("Encode an external 3DGS `.ply` into SLAT (render → DINOv2 → voxelize → "
                        "VAE encode). Expects a TRELLIS-style `.ply`. Slower: loads DINOv2.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    b_ply = gr.File(label="3DGS .ply", file_types=[".ply"], type="filepath")
                    b_save = gr.Textbox(label="Save as (name)", value="bridged")
                    b_nviews = gr.Slider(50, 200, value=150, step=10, label="Render views")
                    b_run = gr.Button("Bridge → SLAT", variant="primary")
                    b_msg = gr.Markdown()
                with gr.Column(scale=5):
                    b_view = result_viewer("Re-decoded from bridged SLAT")
            bridge_evt = b_run.click(do_bridge, [b_ply, b_nviews, b_save], [b_view, b_msg],
                                     show_progress="full")

    # global refresh: repopulate every source dropdown from the current outputs/ library
    def _refresh_all():
        choices = list_slats()
        return [gr.update(choices=choices) for _ in _src_dropdowns]

    refresh_btn.click(_refresh_all, None, _src_dropdowns)
    # Generate/Bridge have no source dropdown of their own; after they save a new asset,
    # auto-refresh every tab's dropdown (the list is fully populated only out here).
    gen_evt.then(_refresh_all, None, _src_dropdowns)
    bridge_evt.then(_refresh_all, None, _src_dropdowns)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
