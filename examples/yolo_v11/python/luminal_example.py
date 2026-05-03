"""Run YOLO v11n through the luminal_cuda_lite backend (Python).

Mirrors `python/reference.py` but, instead of running the eager PyTorch
forward, wraps the (BN-fused) model with ``torch.compile(..., backend=
luminal_backend)`` so the actual computation is offloaded to luminal's CUDA
backend. Compares the decoded predictions against the eager reference and
reports detections after NMS.

Requirements:
  * `luminal` Python module built with the cuda feature.
    (See `crates/luminal_python/run_tests_cuda.sh`.)
  * `ultralytics`, `torch` (CUDA), `opencv-python-headless`, `numpy`.
"""

from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

import sys

# Re-use the shared preprocessing helpers from reference.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import preprocess, fetch_sample_image  # noqa: E402

from luminal import luminal_backend  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the luminal_cuda_lite backend.")
    device = torch.device("cuda")

    img_path = fetch_sample_image()
    nchw, meta = preprocess(img_path)
    x = torch.from_numpy(nchw).to(device)

    yolo = YOLO("yolo11n.pt")
    pt_model = yolo.model.eval()
    pt_model.fuse()  # fold BN into Conv (forward_fuse)
    pt_model.model[-1].export = True  # decode-only output
    pt_model.to(device)

    # PyTorch eager reference
    with torch.no_grad():
        ref = pt_model(x)
    if isinstance(ref, (list, tuple)):
        ref = ref[0]

    # luminal compiled forward
    torch._dynamo.reset()
    compiled = torch.compile(pt_model, backend=luminal_backend)
    with torch.no_grad():
        out = compiled(x)
    if isinstance(out, (list, tuple)):
        out = out[0]

    max_diff = float(torch.max(torch.abs(out - ref)))
    mean_diff = float(torch.mean(torch.abs(out - ref)))
    print(f"output shape:  {tuple(out.shape)}")
    print(f"reference shape: {tuple(ref.shape)}")
    print(f"max_abs:  {max_diff:.4e}")
    print(f"mean_abs: {mean_diff:.4e}")

    # NMS on luminal output for a sanity-check display
    detections = non_max_suppression(
        out.detach().clone(), conf_thres=0.25, iou_thres=0.45, max_det=300
    )[0]
    coco = pt_model.names if hasattr(pt_model, "names") else {}
    print("\nLuminal detections (after NMS):")
    pad_x, pad_y = meta["pad"]
    r = meta["ratio"]
    for det in detections.tolist():
        x1, y1, x2, y2, conf, cls = det
        ox1 = (x1 - pad_x) / r
        oy1 = (y1 - pad_y) / r
        ox2 = (x2 - pad_x) / r
        oy2 = (y2 - pad_y) / r
        print(
            f"  {coco.get(int(cls), str(int(cls))):>14}  conf={conf:.3f}"
            f"  xyxy=[{ox1:.1f}, {oy1:.1f}, {ox2:.1f}, {oy2:.1f}]"
        )


if __name__ == "__main__":
    main()
