"""Native-asset restyle: freeze a SLAT's structure, regenerate appearance from a new prompt.

The coarse geometry of a TRELLIS asset lives in the *structure* ``{p_i}`` (the SparseTensor
``coords``); the appearance / material lives in the per-voxel latents ``{z_i}`` (``feats``).
To restyle, we keep the structure fixed and re-run only stage-2 (the SLAT flow) conditioned
on a new text prompt — exactly what ``TrellisTextTo3DPipeline.run_variant`` does, except we
take the structure from a cached SLAT instead of voxelizing a mesh, so geometry is preserved
exactly (no voxelization round-trip).

This is pure composition of TRELLIS's *public* pipeline methods (``get_cond`` / ``sample_slat``
/ ``decode_slat``) — no edits to, and no subclassing of, the vendored submodule.
"""
from typing import Optional, Sequence, Tuple

__all__ = ["restyle"]

_DEFAULT_FORMATS = ("mesh", "gaussian", "radiance_field")


def restyle(
    pipeline,
    structure,
    prompt: str,
    seed: int = 42,
    slat_sampler_params: Optional[dict] = None,
    formats: Sequence[str] = _DEFAULT_FORMATS,
) -> Tuple[object, dict]:
    """Re-texture an asset by re-running stage-2 on a fixed structure with a new prompt.

    Args:
        pipeline: a loaded ``TrellisTextTo3DPipeline``.
        structure: the structure to keep. Either a SLAT ``SparseTensor`` (its ``coords`` are
            used) or a raw ``[N, 4]`` int coords tensor.
        prompt: the new text prompt describing the desired appearance/material.
        seed: RNG seed for the stage-2 noise (same seed + structure => reproducible).
        slat_sampler_params: optional overrides for the SLAT sampler (e.g. ``{"steps": 12,
            "cfg_strength": 7.5}``).
        formats: which decoders to run.

    Returns:
        ``(slat, outputs)`` — the newly sampled SLAT SparseTensor and the decoded assets dict.
    """
    import torch

    coords = structure.coords if hasattr(structure, "coords") else structure
    cond = pipeline.get_cond([prompt])
    torch.manual_seed(seed)
    slat = pipeline.sample_slat(cond, coords, slat_sampler_params or {})
    outputs = pipeline.decode_slat(slat, list(formats))
    return slat, outputs
