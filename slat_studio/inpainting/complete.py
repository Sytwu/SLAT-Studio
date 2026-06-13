"""Two-stage RePaint SLAT inpainting / completion (training-free).

Given a source SLAT with a hole (a voxel bounding box whose voxels were removed, or simply a
region we declare "unknown"), regenerate plausible geometry + appearance inside the hole that
is consistent with the surrounding asset and a text prompt — while leaving everything outside
the hole bit-exact.

Why two stages (and why RePaint on each):
    TRELLIS generates in two steps — (1) a *dense* sparse-structure latent that decodes to a
    ``64**3`` occupancy ``{p_i}``, then (2) a *sparse* structured latent ``{z_i}`` on those
    active voxels. A hole is missing geometry, so we cannot just regenerate latents on a fixed
    structure (that is Phase-3 region editing). We must first *grow new structure* in the hole,
    then *paint latents* on it. RePaint pins the known part at every Euler step:

      Stage 1 — encode the holed occupancy with the SS VAE encoder -> a known dense latent;
                ``DenseRepaintFlowSampler`` regenerates the hole cells, keeps the rest ->
                decode -> a *filled* occupancy (new coords, a superset of the surviving ones).
      Stage 2 — on the new coords, the voxels that coincide with surviving source voxels keep
                their (bit-exact) source latent; the newly-grown voxels are "unknown" and are
                RePaint-sampled by the Phase-3 ``RepaintFlowSampler``.

Pure composition of TRELLIS public methods + our two sampler subclasses. Nothing under
``third_party/TRELLIS`` is modified. The SS stage has no latent normalization (the structure
flow trains directly on encoder outputs), so the stage-1 known latent is the encoder output
as-is; the slat stage uses the pipeline's ``slat_normalization`` exactly like Phase 3.

Reference (training-free SLAT inpainting): InpaintSLat (arXiv:2605.00664) — a different
mechanism (initial-noise optimization); ours is a RePaint masked-sampling adaptation.
"""
from typing import Optional, Sequence, Tuple

from ..samplers import RepaintFlowSampler, DenseRepaintFlowSampler
from ..morph.interpolate import _lin

__all__ = [
    "inpaint_slat",
    "carve_hole",
    "load_ss_encoder",
    "coords_to_occupancy",
    "occupancy_to_coords",
    "DEFAULT_SS_ENCODER",
]

# The sparse-structure VAE encoder ships in the image-large checkpoint (the text pipeline
# only loads the *decoder*); same family the structure flow was trained against.
DEFAULT_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


def load_ss_encoder(pretrained: str = DEFAULT_SS_ENCODER, device: str = "cuda"):
    """Load the frozen sparse-structure VAE encoder (occupancy ``[B,1,64^3]`` -> latent)."""
    from trellis import models
    return models.from_pretrained(pretrained).eval().to(device)


def _vox(coords):
    """Return the ``[N,3]`` integer x,y,z columns of a SparseTensor's coords."""
    return coords[:, 1:].long() if coords.shape[1] == 4 else coords.long()


def _inside_bbox(coords, bbox):
    """Boolean ``[N]`` mask of voxels inside ``bbox`` (inclusive lower, exclusive upper)."""
    c = _vox(coords)
    x0, y0, z0, x1, y1, z1 = bbox
    return (
        (c[:, 0] >= x0) & (c[:, 0] < x1) &
        (c[:, 1] >= y0) & (c[:, 1] < y1) &
        (c[:, 2] >= z0) & (c[:, 2] < z1)
    )


def coords_to_occupancy(coords, res: int = 64):
    """Scatter ``[N,4]``/``[N,3]`` voxel coords into a dense ``[1,1,res,res,res]`` float grid."""
    import torch
    c = _vox(coords)
    occ = torch.zeros(1, 1, res, res, res, device=coords.device)
    occ[0, 0, c[:, 0], c[:, 1], c[:, 2]] = 1.0
    return occ


def occupancy_to_coords(occ, thresh: float = 0.0):
    """Dense occupancy ``[1,1,r,r,r]`` -> ``[N,4]`` int coords (matches the TRELLIS pipeline)."""
    import torch
    return torch.argwhere(occ > thresh)[:, [0, 2, 3, 4]].int()


def carve_hole(slat, bbox):
    """Remove the voxels inside ``bbox`` from a SLAT (returns a new SparseTensor + #removed).

    Used to *make* a holed input for demos/tests; ``inpaint_slat`` does not need this (it takes
    the hole as a bbox and treats those voxels as unknown regardless of whether they are present).
    """
    keep = ~_inside_bbox(slat.coords, bbox)
    holed = slat.__class__(feats=slat.feats[keep], coords=slat.coords[keep])
    return holed, int((~keep).sum().item())


def _hole_ss_mask(bbox, res: int = 64, ss_res: int = 16, device="cuda"):
    """Dense ``[1,1,ss_res^3]`` mask: 1 for every stage-1 latent cell overlapping the hole bbox."""
    import torch
    f = res // ss_res  # voxels per latent cell along each axis (64/16 = 4)
    x0, y0, z0, x1, y1, z1 = bbox
    lo = (x0 // f, y0 // f, z0 // f)
    hi = (-(-x1 // f), -(-y1 // f), -(-z1 // f))  # ceil-divide so partially-covered cells count
    m = torch.zeros(1, 1, ss_res, ss_res, ss_res, device=device)
    m[0, 0, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1.0
    return m


def inpaint_slat(
    pipeline,
    source_slat,
    hole_bbox,
    prompt: str,
    ss_encoder,
    seed: int = 42,
    ss_resample: int = 1,
    slat_resample: int = 1,
    ss_sampler_params: Optional[dict] = None,
    slat_sampler_params: Optional[dict] = None,
    formats: Sequence[str] = ("gaussian",),
) -> Tuple[object, object, dict]:
    """Complete the hole in ``source_slat`` from ``prompt``; keep the rest bit-exact.

    Args:
        pipeline: a loaded ``TrellisTextTo3DPipeline`` (provides cond, both flow models, the SS
            decoder, slat normalization, and the slat decoders).
        source_slat: the SLAT to complete (a ``SparseTensor``, denormalized as stored). May still
            contain voxels inside the hole — they are ignored (treated as unknown).
        hole_bbox: ``(x0,y0,z0, x1,y1,z1)`` voxel indices (0..64, inclusive lower / exclusive upper).
        prompt: text describing the asset (steers what fills the hole).
        ss_encoder: the sparse-structure VAE encoder from :func:`load_ss_encoder`.
        seed: RNG seed (reproducible structure + latents).
        ss_resample / slat_resample: RePaint jump-back counts for stage 1 / stage 2 (1 = off).
        ss_sampler_params / slat_sampler_params: overrides merged over the pipeline defaults.
        formats: slat decoders to run on the completed SLAT.

    Returns:
        ``(completed_slat, outputs, info)`` — the completed SLAT (denormalized; latents on
        surviving voxels are bit-exact equal to the source), decoded assets, and an info dict.
    """
    import torch
    from trellis.modules import sparse as sp

    device = source_slat.feats.device
    res = 64
    ss_res = int(pipeline.models["sparse_structure_flow_model"].resolution)
    cond = pipeline.get_cond([prompt])

    # ---- known geometry = source minus the hole (so the hole is truly "unknown") ----------
    known_vox = ~_inside_bbox(source_slat.coords, hole_bbox)
    known_coords = source_slat.coords[known_vox]
    occ_known = coords_to_occupancy(known_coords, res)                  # [1,1,64,64,64] float

    # ---- stage 1: structure RePaint in the dense SS latent --------------------------------
    with torch.no_grad():
        # encoder casts to fp16 internally; its input_layer stays fp32, so feed fp32 occupancy
        z0_known = ss_encoder(occ_known.float())
    ss_mask = _hole_ss_mask(hole_bbox, res, ss_res, device=device)     # [1,1,16,16,16]
    torch.manual_seed(seed)
    ss_noise = torch.randn_like(z0_known)
    ss_sampler = DenseRepaintFlowSampler(sigma_min=pipeline.sparse_structure_sampler.sigma_min)
    ssp = {**pipeline.sparse_structure_sampler_params, **(ss_sampler_params or {})}
    z_s = ss_sampler.sample(
        pipeline.models["sparse_structure_flow_model"], ss_noise, **cond,
        x_0_known=z0_known, mask=ss_mask, resample=ss_resample, **ssp,
    ).samples
    new_coords = occupancy_to_coords(pipeline.models["sparse_structure_decoder"](z_s))

    # ---- align surviving latents onto the new (filled) structure --------------------------
    key_new = _lin(new_coords, res)
    key_src = _lin(source_slat.coords, res)
    in_src = _inside_bbox(source_slat.coords, hole_bbox)
    # a new voxel is "known" only if it coincided with a *surviving* (outside-hole) source voxel
    survivor_keys = key_src[~in_src]
    known_new = torch.isin(key_new, survivor_keys)                     # [M] bool
    order = torch.argsort(key_src)
    gathered = order[torch.searchsorted(key_src[order], key_new[known_new])]

    C = source_slat.feats.shape[1]
    src_raw = torch.zeros(new_coords.shape[0], C, device=device)
    src_raw[known_new] = source_slat.feats[gathered]                   # bit-exact source latents

    norm = pipeline.slat_normalization
    mean = torch.tensor(norm["mean"], device=device)[None]
    std = torch.tensor(norm["std"], device=device)[None]
    x0_feats = torch.zeros(new_coords.shape[0], C, device=device)
    x0_feats[known_new] = (src_raw[known_new] - mean) / std           # normalized known latents
    x_0_known = sp.SparseTensor(feats=x0_feats, coords=new_coords)
    mask = (~known_new).float().unsqueeze(1)                          # 1 = regenerate (filled)

    # ---- stage 2: latent RePaint on the filled voxels (Phase-3 sampler) -------------------
    torch.manual_seed(seed + 1)
    noise = x_0_known.replace(torch.randn_like(x_0_known.feats))
    sampler = RepaintFlowSampler(sigma_min=pipeline.slat_sampler.sigma_min)
    spp = {**pipeline.slat_sampler_params, **(slat_sampler_params or {})}
    gen_norm = sampler.sample(
        pipeline.models["slat_flow_model"], noise, **cond,
        x_0_known=x_0_known, mask=mask, resample=slat_resample, **spp,
    ).samples

    gen_feats = gen_norm.feats * std + mean
    keep = 1 - mask
    final = gen_feats * mask + src_raw * keep                         # survivors bit-exact
    completed = sp.SparseTensor(feats=final, coords=new_coords)

    outputs = pipeline.decode_slat(completed, list(formats)) if formats else {}
    info = {
        "n_source": int(source_slat.coords.shape[0]),
        "n_known": int(known_vox.sum().item()),
        "n_removed_in_hole": int(in_src.sum().item()),
        "n_completed": int(new_coords.shape[0]),
        "n_kept_in_completed": int(known_new.sum().item()),
        "n_grown": int((~known_new).sum().item()),
        "ss_res": ss_res,
        "hole_bbox": tuple(int(v) for v in hole_bbox),
    }
    return completed, outputs, info
