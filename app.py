"""SLAT-Studio Gradio app — interactive front-end for the downstream 3D tasks.

One tab per capability (Generate / Restyle / Edit / Morph / Inpaint / Bridge). Every task
runs on a single, lazily-loaded ``TrellisTextTo3DPipeline`` and exchanges SLATs through a
**file library**: results are saved as ``outputs/*.npz`` and picked up by other tabs from a
dropdown. This mirrors the existing ``phaseN_*.npz`` workflow and survives restarts.

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
import time
import traceback

import imageio
import numpy as np
import torch
import gradio as gr

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from slat_studio.io import load_slat, save_slat, read_extra
from slat_studio.pipelines import text_to_slat
from slat_studio.style import restyle
from slat_studio.editing import edit_region
from slat_studio.morph import SlatMorpher
from slat_studio.inpainting import inpaint_slat, carve_hole, load_ss_encoder

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

PRETRAINED = "microsoft/TRELLIS-text-xlarge"
GRID = 64
RENDER_FRAMES = 30

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


def render_video(slat, stem: str, frames: int = RENDER_FRAMES) -> str:
    """Decode a SLAT to gaussians, render a turntable, write ``outputs/<stem>.mp4``; free GPU."""
    gs = decode_gs(slat)
    vid = render_utils.render_video(gs, num_frames=frames)["color"]
    del gs
    torch.cuda.empty_cache()
    path = os.path.join(OUT, f"{stem}.mp4")
    imageio.mimsave(path, list(vid), fps=30)
    return path


def render_front(slat, frames: int = RENDER_FRAMES) -> np.ndarray:
    """Single brightest-view frame (HWC uint8) — cheap preview for grids."""
    gs = decode_gs(slat)
    vid = np.stack(render_utils.render_video(gs, num_frames=frames)["color"])
    del gs
    torch.cuda.empty_cache()
    v = int(vid.reshape(vid.shape[0], -1).mean(axis=1).argmax())
    return vid[v]


def voxel_extent(slat):
    """(min[xyz], max[xyz]) of the occupied voxels, as plain int lists."""
    c = slat.coords[:, 1:].long()
    return c.min(0).values.tolist(), c.max(0).values.tolist()


def _err(prefix: str, e: Exception) -> str:
    traceback.print_exc()
    return f"❌ {prefix}: {type(e).__name__}: {e}"


# ============================================================================= TAB CALLBACKS
def do_generate(prompt, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a prompt.", gr.update()
    try:
        t0 = time.time()
        slat, _ = text_to_slat(get_pipe(), prompt.strip(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or prompt)
        save_slat(slat, os.path.join(OUT, stem + ".npz"),
                  extra={"prompt": prompt.strip(), "seed": int(seed), "task": "generate"})
        vid = render_video(slat, stem)
        n = slat.coords.shape[0]
        msg = f"✅ generated {n} voxels in {time.time()-t0:.0f}s → saved **{stem}.npz**"
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("generate failed", e), gr.update()


def do_restyle(source, prompt, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a restyle prompt.", gr.update()
    try:
        t0 = time.time()
        src = load_slat(_npz_path(source), device="cuda")
        slat, _ = restyle(get_pipe(), src, prompt.strip(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_restyled"))
        save_slat(slat, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "seed": int(seed),
                         "task": "restyle"})
        vid = render_video(slat, stem)
        msg = f"✅ restyled in {time.time()-t0:.0f}s → saved **{stem}.npz** (structure kept)"
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("restyle failed", e), gr.update()


def do_inspect(source):
    try:
        src = load_slat(_npz_path(source), device="cuda")
        mn, mx = voxel_extent(src)
        n = src.coords.shape[0]
        del src
        torch.cuda.empty_cache()
        return f"**{source}**: {n} voxels · voxel range min={mn} max={mx} (grid 0..{GRID-1})"
    except Exception as e:
        return _err("inspect failed", e)


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
    if preset == "top half (Y)":
        y0 = midy
    elif preset == "bottom half (Y)":
        y1 = midy
    elif preset == "left half (X)":
        x1 = midx
    elif preset == "right half (X)":
        x0 = midx
    elif preset == "front half (Z)":
        z1 = midz
    elif preset == "back half (Z)":
        z0 = midz
    elif preset == "corner wedge (+X +Z upper)":
        x0, z0 = midx, midz
    # "full extent" leaves the full box
    return x0, y0, z0, x1, y1, z1


def do_edit(source, prompt, x0, y0, z0, x1, y1, z1, seed, resample, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter an edit prompt.", gr.update()
    try:
        t0 = time.time()
        src = load_slat(_npz_path(source), device="cuda")
        bbox = tuple(int(v) for v in (x0, y0, z0, x1, y1, z1))
        edited, mask, _ = edit_region(get_pipe(), src, bbox, prompt.strip(),
                                      seed=int(seed), resample=int(resample), formats=())
        n_in = int(mask.sum().item())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_edited"))
        save_slat(edited, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "bbox": list(bbox),
                         "seed": int(seed), "task": "edit"})
        vid = render_video(edited, stem)
        msg = (f"✅ edited {n_in} in-box voxels (of {src.coords.shape[0]}) in {time.time()-t0:.0f}s "
               f"→ saved **{stem}.npz**")
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("edit failed", e), gr.update()


def do_morph_grid(source_a, source_b, steps, seed):
    try:
        a = load_slat(_npz_path(source_a), device="cuda")
        b = load_slat(_npz_path(source_b), device="cuda")
        morpher = SlatMorpher(a, b, seed=int(seed))
        ts = np.linspace(0.0, 1.0, int(steps)).tolist()
        frames, labels = [], []
        for t in ts:
            frames.append(render_front(morpher.at(t)))           # one decode at a time
            labels.append(f"t={t:.2f}")
        grid = np.concatenate(frames, axis=1)                    # side-by-side
        msg = ("✅ morph grid " + " | ".join(labels) +
               f"  (union {morpher.n_union} voxels, A={a.coords.shape[0]} B={b.coords.shape[0]})")
        return grid, msg
    except Exception as e:
        return None, _err("morph grid failed", e)


def do_morph_at(source_a, source_b, t, seed, save_as):
    try:
        t0 = time.time()
        a = load_slat(_npz_path(source_a), device="cuda")
        b = load_slat(_npz_path(source_b), device="cuda")
        morpher = SlatMorpher(a, b, seed=int(seed))
        mid = morpher.at(float(t))
        stem = _safe_stem(save_as or f"morph_t{float(t):.2f}".replace(".", "p"))
        save_slat(mid, os.path.join(OUT, stem + ".npz"),
                  extra={"source_a": source_a, "source_b": source_b, "t": float(t),
                         "seed": int(seed), "task": "morph"})
        vid = render_video(mid, stem)
        msg = f"✅ morph at t={float(t):.2f} ({mid.coords.shape[0]} voxels) in {time.time()-t0:.0f}s → **{stem}.npz**"
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("morph render failed", e), gr.update()


def do_inpaint(source, prompt, x0, y0, z0, x1, y1, z1, seed, save_as):
    if not prompt or not prompt.strip():
        return None, "❌ Enter a prompt describing the asset.", gr.update()
    try:
        t0 = time.time()
        src = load_slat(_npz_path(source), device="cuda")
        bbox = tuple(int(v) for v in (x0, y0, z0, x1, y1, z1))
        holed, n_removed = carve_hole(src, bbox)                 # actually remove the hole voxels
        completed, _, info = inpaint_slat(get_pipe(), holed, bbox, prompt.strip(),
                                          get_ss_encoder(), seed=int(seed), formats=())
        stem = _safe_stem(save_as or (os.path.splitext(source)[0] + "_completed"))
        save_slat(completed, os.path.join(OUT, stem + ".npz"),
                  extra={"source": source, "prompt": prompt.strip(), "hole_bbox": list(bbox),
                         "seed": int(seed), "task": "inpaint"})
        vid = render_video(completed, stem)
        msg = (f"✅ carved {n_removed} voxels, regrew {info['n_grown']} "
               f"(survivors bit-exact) in {time.time()-t0:.0f}s → saved **{stem}.npz**")
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("inpaint failed", e), gr.update()


def do_bridge(ply_file, nviews, save_as):
    if not ply_file:
        return None, "❌ Upload a 3DGS .ply first.", gr.update()
    try:
        from trellis.representations import Gaussian
        from slat_studio import bridge
        t0 = time.time()
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
        vid = render_video(slat, stem)
        msg = f"✅ bridged → {coords.shape[0]} voxels in {time.time()-t0:.0f}s → saved **{stem}.npz**"
        return vid, msg, gr.update(choices=list_slats())
    except Exception as e:
        return None, _err("bridge failed", e), gr.update()


# ============================================================================= UI
BBOX_PRESETS = ["full extent", "top half (Y)", "bottom half (Y)", "left half (X)",
                "right half (X)", "front half (Z)", "back half (Z)",
                "corner wedge (+X +Z upper)"]

_src_dropdowns = []  # every source dropdown, refreshed together by the global button


def source_dropdown(label="Source SLAT"):
    dd = gr.Dropdown(choices=list_slats(), label=label, interactive=True)
    _src_dropdowns.append(dd)
    return dd


def bbox_row():
    with gr.Row():
        x0 = gr.Number(label="x0", precision=0, value=0)
        y0 = gr.Number(label="y0", precision=0, value=0)
        z0 = gr.Number(label="z0", precision=0, value=0)
        x1 = gr.Number(label="x1", precision=0, value=GRID)
        y1 = gr.Number(label="y1", precision=0, value=GRID)
        z1 = gr.Number(label="z1", precision=0, value=GRID)
    return x0, y0, z0, x1, y1, z1


with gr.Blocks(title="SLAT-Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# SLAT-Studio\n"
        "Training-free downstream 3D tasks on TRELLIS structured latents. Each tab saves its "
        "result as an `outputs/*.npz` SLAT that the other tabs can load — use **🔄 Refresh** "
        "after producing a new asset. First run of any tab loads the pipeline (~1 min)."
    )
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh SLAT library (all tabs)", scale=1)

    # --------------------------------------------------------------------- Generate
    with gr.Tabs():
        with gr.Tab("Generate"):
            gr.Markdown("Text → 3D. Produces the source SLAT for every other tab.")
            with gr.Row():
                with gr.Column():
                    g_prompt = gr.Textbox(label="Prompt", lines=2,
                                          value="A rustic wooden treasure chest with iron bands.")
                    g_seed = gr.Number(label="Seed", precision=0, value=42)
                    g_save = gr.Textbox(label="Save as (name)", value="my_asset")
                    g_run = gr.Button("Generate", variant="primary")
                    g_msg = gr.Markdown()
                g_vid = gr.Video(label="Turntable", autoplay=True, loop=True)
            g_run.click(do_generate, [g_prompt, g_seed, g_save], [g_vid, g_msg, g_save],
                        show_progress="full")

        # --------------------------------------------------------------- Restyle
        with gr.Tab("Restyle"):
            gr.Markdown("Re-texture the **whole** asset from a new prompt; geometry preserved "
                        "(stage-2 re-run on the frozen structure).")
            with gr.Row():
                with gr.Column():
                    r_src = source_dropdown()
                    r_prompt = gr.Textbox(label="New appearance / material prompt", lines=2,
                                          value="A treasure chest made of solid gold.")
                    r_seed = gr.Number(label="Seed", precision=0, value=42)
                    r_save = gr.Textbox(label="Save as (name)", value="")
                    r_run = gr.Button("Restyle", variant="primary")
                    r_msg = gr.Markdown()
                r_vid = gr.Video(label="Restyled", autoplay=True, loop=True)
            r_run.click(do_restyle, [r_src, r_prompt, r_seed, r_save], [r_vid, r_msg, r_src],
                        show_progress="full")

        # --------------------------------------------------------------- Edit
        with gr.Tab("Edit (region)"):
            gr.Markdown("RePaint the latents **inside a voxel bbox** from a prompt; the rest stays "
                        "bit-exact. Use *Inspect* to see the extent, or a preset to fill the box.")
            with gr.Row():
                with gr.Column():
                    e_src = source_dropdown()
                    with gr.Row():
                        e_inspect = gr.Button("Inspect extent")
                        e_preset = gr.Dropdown(choices=BBOX_PRESETS, value="top half (Y)",
                                               label="Preset")
                        e_apply = gr.Button("Apply preset → bbox")
                    e_x0, e_y0, e_z0, e_x1, e_y1, e_z1 = bbox_row()
                    e_prompt = gr.Textbox(label="In-box prompt", lines=2,
                                          value="molten glowing lava and burning embers")
                    with gr.Row():
                        e_seed = gr.Number(label="Seed", precision=0, value=7)
                        e_resample = gr.Number(label="RePaint resample (1=off)", precision=0,
                                               value=1)
                    e_save = gr.Textbox(label="Save as (name)", value="")
                    e_run = gr.Button("Edit region", variant="primary")
                    e_msg = gr.Markdown()
                e_vid = gr.Video(label="Edited", autoplay=True, loop=True)
            e_inspect.click(do_inspect, [e_src], [e_msg])
            e_apply.click(apply_preset, [e_src, e_preset],
                          [e_x0, e_y0, e_z0, e_x1, e_y1, e_z1])
            e_run.click(do_edit,
                        [e_src, e_prompt, e_x0, e_y0, e_z0, e_x1, e_y1, e_z1,
                         e_seed, e_resample, e_save],
                        [e_vid, e_msg, e_src], show_progress="full")

        # --------------------------------------------------------------- Morph
        with gr.Tab("Morph"):
            gr.Markdown("Interpolate between two SLATs (structure union + per-voxel dissolve). "
                        "Endpoints are exact. Grid = quick stills; Render at t = turntable.")
            with gr.Row():
                m_a = source_dropdown("Source A (t=0)")
                m_b = source_dropdown("Source B (t=1)")
                m_seed = gr.Number(label="Seed", precision=0, value=0)
            with gr.Row():
                m_steps = gr.Slider(2, 9, value=5, step=1, label="Grid steps")
                m_grid_btn = gr.Button("Morph grid (stills)", variant="primary")
            m_grid = gr.Image(label="Morph grid (t=0 … t=1)")
            with gr.Row():
                m_t = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="t")
                m_save = gr.Textbox(label="Save as (name)", value="")
                m_at_btn = gr.Button("Render at t", variant="primary")
            m_vid = gr.Video(label="Morph at t", autoplay=True, loop=True)
            m_msg = gr.Markdown()
            m_grid_btn.click(do_morph_grid, [m_a, m_b, m_steps, m_seed], [m_grid, m_msg],
                             show_progress="full")
            m_at_btn.click(do_morph_at, [m_a, m_b, m_t, m_seed, m_save],
                           [m_vid, m_msg, m_a], show_progress="full")

        # --------------------------------------------------------------- Inpaint
        with gr.Tab("Inpaint"):
            gr.Markdown("Carve a bbox hole, then regrow **geometry + appearance** (two-stage "
                        "RePaint). Survivor voxels stay bit-exact; the fill is prompt-conditioned.")
            with gr.Row():
                with gr.Column():
                    i_src = source_dropdown()
                    with gr.Row():
                        i_inspect = gr.Button("Inspect extent")
                        i_preset = gr.Dropdown(choices=BBOX_PRESETS,
                                               value="corner wedge (+X +Z upper)", label="Preset")
                        i_apply = gr.Button("Apply preset → hole bbox")
                    i_x0, i_y0, i_z0, i_x1, i_y1, i_z1 = bbox_row()
                    i_prompt = gr.Textbox(label="Asset prompt (steers the fill)", lines=2,
                                          value="A rustic wooden treasure chest with iron bands.")
                    i_seed = gr.Number(label="Seed", precision=0, value=11)
                    i_save = gr.Textbox(label="Save as (name)", value="")
                    i_run = gr.Button("Carve + inpaint", variant="primary")
                    i_msg = gr.Markdown()
                i_vid = gr.Video(label="Completed", autoplay=True, loop=True)
            i_inspect.click(do_inspect, [i_src], [i_msg])
            i_apply.click(apply_preset, [i_src, i_preset],
                          [i_x0, i_y0, i_z0, i_x1, i_y1, i_z1])
            i_run.click(do_inpaint,
                        [i_src, i_prompt, i_x0, i_y0, i_z0, i_x1, i_y1, i_z1, i_seed, i_save],
                        [i_vid, i_msg, i_src], show_progress="full")

        # --------------------------------------------------------------- Bridge
        with gr.Tab("Bridge (.ply → SLAT)"):
            gr.Markdown("Encode an external 3DGS `.ply` into SLAT (render → DINOv2 → voxelize → "
                        "VAE encode). Expects a TRELLIS-style `.ply`. Slower: loads DINOv2.")
            with gr.Row():
                with gr.Column():
                    b_ply = gr.File(label="3DGS .ply", file_types=[".ply"], type="filepath")
                    b_nviews = gr.Slider(50, 200, value=150, step=10, label="Render views")
                    b_save = gr.Textbox(label="Save as (name)", value="bridged")
                    b_run = gr.Button("Bridge → SLAT", variant="primary")
                    b_msg = gr.Markdown()
                b_vid = gr.Video(label="Re-decoded from bridged SLAT", autoplay=True, loop=True)
            b_run.click(do_bridge, [b_ply, b_nviews, b_save], [b_vid, b_msg, b_save],
                        show_progress="full")

    # global refresh: repopulate every source dropdown from the current outputs/ library
    def _refresh_all():
        choices = list_slats()
        return [gr.update(choices=choices) for _ in _src_dropdowns]

    refresh_btn.click(_refresh_all, None, _src_dropdowns)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
