"""Phase 0 capstone: full image->3D run to exercise the whole stack end-to-end.

Downloads microsoft/TRELLIS-image-large, runs the pipeline on a bundled example image,
and exports a GLB + a rendered video, confirming the SLAT decoders (gaussian/mesh) and the
CUDA renderers work. Run via scripts/run_full_smoke.sh (sets env + PYTHONPATH).
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

TRELLIS_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "TRELLIS")
IMG = os.path.join(TRELLIS_DIR, "assets", "example_image", "T.png")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

print("[full] loading pipeline (downloads weights on first run)...")
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()

print(f"[full] running on {IMG}")
image = Image.open(IMG)
outputs = pipe.run(image, seed=1)
print("[full] output formats:", list(outputs.keys()))

gs = outputs["gaussian"][0]
print(f"[full] gaussian: {gs.get_xyz.shape[0]} splats")

# exercise the CUDA gaussian renderer
video = render_utils.render_video(gs, num_frames=30)["color"]
imageio.mimsave(os.path.join(OUT, "smoke_gs.mp4"), video, fps=30)
print("[full] wrote outputs/smoke_gs.mp4")

# exercise mesh decode + GLB export (gaussian appearance baked onto mesh)
glb = postprocessing_utils.to_glb(gs, outputs["mesh"][0], simplify=0.95, texture_size=1024)
glb.export(os.path.join(OUT, "smoke.glb"))
print("[full] wrote outputs/smoke.glb")
print("[full] DONE")
