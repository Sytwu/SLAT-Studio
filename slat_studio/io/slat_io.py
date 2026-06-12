"""Save / load a SLAT (structured latent) as a single ``.npz`` file.

A SLAT is a ``trellis.modules.sparse.SparseTensor`` holding, for each active voxel,
a coordinate ``p_i`` (``coords``, an ``[N, 4]`` int tensor of ``[batch, x, y, z]``) and a
latent feature ``z_i`` (``feats``, an ``[N, C]`` float tensor). This is the high-fidelity,
zero-loss native representation: caching it at generation time lets us decode or restyle an
asset later without re-running stage-1/stage-2.

We persist exactly ``coords`` + ``feats`` (+ a little metadata) so a round-trip reproduces
the SparseTensor bit-for-bit. ``trellis`` is imported lazily inside the functions so that
importing this module does not require the CUDA env to be active.
"""
from typing import Optional
import numpy as np

__all__ = ["save_slat", "load_slat"]

_FORMAT = "slat-studio/slat-npz-v1"


def save_slat(slat, path: str, extra: Optional[dict] = None) -> str:
    """Serialize a SLAT SparseTensor to ``path`` (a ``.npz``).

    Args:
        slat: a ``trellis...SparseTensor`` (single- or multi-batch).
        path: output ``.npz`` path.
        extra: optional small JSON-serializable dict stored alongside (e.g. the prompt/seed).

    Returns:
        The path written.
    """
    coords = slat.coords.detach().cpu().numpy().astype(np.int32)   # [N, 4] = [batch, x, y, z]
    feats = slat.feats.detach().cpu().numpy().astype(np.float32)   # [N, C]
    payload = dict(
        format=np.array(_FORMAT),
        coords=coords,
        feats=feats,
        num_voxels=np.array(coords.shape[0], dtype=np.int64),
        channels=np.array(feats.shape[1], dtype=np.int64),
    )
    if extra:
        # stored as a 0-d object array; loaded back with allow_pickle
        payload["extra"] = np.array(extra, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_slat(path: str, device: str = "cuda"):
    """Load a ``.npz`` written by :func:`save_slat` back into a SparseTensor on ``device``.

    Returns:
        A ``trellis...SparseTensor`` with the original ``coords`` and ``feats``.
    """
    import torch
    from trellis.modules import sparse as sp

    data = np.load(path, allow_pickle=True)
    fmt = str(data["format"]) if "format" in data else "<unknown>"
    if fmt != _FORMAT:
        raise ValueError(f"{path}: unexpected SLAT format {fmt!r}, expected {_FORMAT!r}")
    coords = torch.from_numpy(data["coords"].astype(np.int32)).to(device)
    feats = torch.from_numpy(data["feats"].astype(np.float32)).to(device)
    return sp.SparseTensor(feats=feats, coords=coords)


def read_extra(path: str) -> dict:
    """Return the optional ``extra`` metadata dict saved with a SLAT, or ``{}``."""
    data = np.load(path, allow_pickle=True)
    if "extra" in data:
        return data["extra"].item()
    return {}
