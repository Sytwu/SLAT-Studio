"""High-level generation helpers that expose the intermediate SLAT.

TRELLIS's ``TrellisTextTo3DPipeline.run`` decodes and throws the SLAT away. For downstream
tasks (restyle / edit / morph) we need that SLAT, so :func:`text_to_slat` reproduces ``run``
step-by-step from the pipeline's *public* methods and returns the SLAT alongside the decoded
assets. No submodule edits.
"""
from typing import Optional, Sequence, Tuple

__all__ = ["text_to_slat"]

_DEFAULT_FORMATS = ("mesh", "gaussian", "radiance_field")


def text_to_slat(
    pipeline,
    prompt: str,
    seed: int = 42,
    sparse_structure_sampler_params: Optional[dict] = None,
    slat_sampler_params: Optional[dict] = None,
    formats: Sequence[str] = _DEFAULT_FORMATS,
) -> Tuple[object, dict]:
    """Run text-to-3D and return ``(slat, outputs)``.

    Mirrors ``TrellisTextTo3DPipeline.run`` but keeps the structured latent so it can be
    cached (:func:`slat_studio.io.save_slat`) or restyled (:func:`slat_studio.style.restyle`).

    Args:
        pipeline: a loaded ``TrellisTextTo3DPipeline``.
        prompt: the text prompt.
        seed: RNG seed for stage-1 (structure) + stage-2 (latent).
        sparse_structure_sampler_params: optional overrides for the stage-1 sampler.
        slat_sampler_params: optional overrides for the stage-2 sampler.
        formats: which decoders to run (pass an empty sequence to skip decoding).

    Returns:
        ``(slat, outputs)`` — the SLAT SparseTensor and the decoded assets dict
        (``{}`` if ``formats`` is empty).
    """
    import torch

    cond = pipeline.get_cond([prompt])
    torch.manual_seed(seed)
    coords = pipeline.sample_sparse_structure(cond, 1, sparse_structure_sampler_params or {})
    slat = pipeline.sample_slat(cond, coords, slat_sampler_params or {})
    outputs = pipeline.decode_slat(slat, list(formats)) if formats else {}
    return slat, outputs
