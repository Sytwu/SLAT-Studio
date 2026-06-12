"""Phase 0 smoke test.

Checks (cheap, no weights):
  - PyTorch + CUDA visible
  - `trellis` importable from the submodule
  - key TRELLIS classes resolve (pipelines, samplers)

With --full it also downloads TRELLIS-image-large and runs a stock image->3D pass
(multi-GB download, needs network + GPU). Run inside the `trellis` conda env with
third_party/TRELLIS on PYTHONPATH.
"""
import argparse
import sys


def check_torch() -> bool:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import torch: {e}")
        return False
    print(f"[ok] torch {torch.__version__}, cuda={torch.version.cuda}, "
          f"available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    return torch.cuda.is_available()


def check_trellis() -> bool:
    try:
        import trellis  # noqa: F401
        from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: F401
        from trellis.pipelines import samplers  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] import trellis: {e}")
        print("       -> is third_party/TRELLIS on PYTHONPATH and the env built?")
        return False
    print("[ok] trellis imports (pipelines + samplers resolve)")
    return True


def run_full() -> bool:
    try:
        from PIL import Image
        from trellis.pipelines import TrellisImageTo3DPipeline
        pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipe.cuda()
        img = Image.open("assets/example.png").convert("RGBA")
        out = pipe.run(img, seed=1)
        print(f"[ok] full pipeline ran; outputs: {list(out.keys())}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] full pipeline: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also download weights and run image->3D")
    args = ap.parse_args()

    ok = check_torch()
    ok = check_trellis() and ok
    if args.full:
        ok = run_full() and ok
    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
