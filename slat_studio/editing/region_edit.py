"""Region editing: regenerate the latents inside a voxel bbox from a new prompt.

Phase 3. Given a source SLAT (native or bridged) and a voxel-space bounding box, re-sample
*only* the latents inside the box conditioned on a new text prompt, while preserving the
latents outside the box. The structure ``{p_i}`` (coords) is unchanged everywhere — only the
appearance latents ``{z_i}`` inside the box move.

This is pure composition of TRELLIS public methods + our ``RepaintFlowSampler`` (a subclass
of the stock slat sampler). Nothing under ``third_party/TRELLIS`` is modified.

Pipeline:
    source SLAT (denormalized)
      -> normalize with the pipeline's slat_normalization        => x_0_known
      -> noise = randn on the SAME coords/ordering
      -> RepaintFlowSampler.sample(known=x_0_known, mask=bbox)    => normalized edited latents
      -> denormalize
      -> hard-composite source latents back outside the mask      => bit-exact preservation
"""
from typing import Optional, Sequence, Tuple

from ..samplers import RepaintFlowSampler, bbox_mask

__all__ = ["edit_region", "normalized_to_voxel_bbox"]


def normalized_to_voxel_bbox(bbox_norm, resolution: int = 64):
    """Convert a bbox in TRELLIS normalized space ``[-0.5,0.5]`` to voxel indices.

    Args:
        bbox_norm: ``(x0,y0,z0, x1,y1,z1)`` floats in ``[-0.5, 0.5]``.
        resolution: SLAT grid (64).

    Returns:
        ``(x0,y0,z0, x1,y1,z1)`` ints in ``[0, resolution]`` (``index = floor((c+0.5)*res)``).
    """
    import math
    out = []
    for v in bbox_norm:
        out.append(int(math.floor((v + 0.5) * resolution)))
    return tuple(out)


def edit_region(
    pipeline,
    source_slat,
    bbox,
    prompt: str,
    seed: int = 42,
    resample: int = 1,
    slat_sampler_params: Optional[dict] = None,
    formats: Sequence[str] = ("gaussian",),
) -> Tuple[object, object, dict]:
    """RePaint-edit the latents inside ``bbox`` from ``prompt``; keep the rest fixed.

    Args:
        pipeline: a loaded ``TrellisTextTo3DPipeline`` (provides cond, normalization, flow model).
        source_slat: the SLAT to edit (a ``SparseTensor``, denormalized as stored).
        bbox: ``(x0,y0,z0, x1,y1,z1)`` voxel indices (inclusive lower, exclusive upper).
            Use :func:`normalized_to_voxel_bbox` for a box specified in ``[-0.5,0.5]``.
        prompt: text describing the desired content inside the box.
        seed: RNG seed for the in-box noise (reproducible).
        resample: RePaint jump-back count (1 = off; >1 harmonizes the boundary, slower).
        slat_sampler_params: overrides merged over the pipeline's defaults (steps/cfg/…).
        formats: decoders to run on the edited SLAT.

    Returns:
        ``(edited_slat, mask, outputs)`` — the edited SLAT (denormalized, latents outside the
        box bit-exact equal to the source), the ``[N,1]`` in-box mask, and the decoded assets.
    """
    import torch

    device = source_slat.feats.device
    norm = pipeline.slat_normalization
    mean = torch.tensor(norm["mean"], device=device)[None]   # [1,C]
    std = torch.tensor(norm["std"], device=device)[None]     # [1,C]

    # known clean latents in the sampler's normalized space (same coords/ordering as source)
    x_0_known = source_slat.replace((source_slat.feats - mean) / std)
    mask = bbox_mask(source_slat.coords, bbox, device=device)  # [N,1]
    n_in = int(mask.sum().item())
    if n_in == 0:
        raise ValueError(f"bbox {bbox} selects 0 voxels — nothing to edit")

    torch.manual_seed(seed)
    noise = x_0_known.replace(torch.randn_like(x_0_known.feats))

    cond = pipeline.get_cond([prompt])
    flow_model = pipeline.models["slat_flow_model"]
    sampler = RepaintFlowSampler(sigma_min=pipeline.slat_sampler.sigma_min)
    params = {**pipeline.slat_sampler_params, **(slat_sampler_params or {})}

    edited_norm = sampler.sample(
        flow_model, noise, **cond,
        x_0_known=x_0_known, mask=mask, resample=resample, **params,
    ).samples

    # denormalize, then hard-composite the original latents back outside the box so that
    # everything outside the edit is preserved bit-exact (no drift from the tiny t=0 jitter).
    edited_feats = edited_norm.feats * std + mean
    keep = 1 - mask
    composited = edited_feats * mask + source_slat.feats * keep
    edited_slat = source_slat.replace(composited)

    outputs = pipeline.decode_slat(edited_slat, list(formats)) if formats else {}
    return edited_slat, mask, outputs
