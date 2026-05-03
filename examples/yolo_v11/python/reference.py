"""YOLO v11n reference inference + weight export for the Rust example.

This script:
  1. Downloads yolo11n.pt (Ultralytics).
  2. Fuses Conv + BatchNorm + bias into a single bias-augmented Conv2d (forward_fuse).
  3. Runs inference on a sample image and saves:
       - reference_input.bin   : preprocessed (1, 3, 640, 640) f32 NCHW image bytes
       - reference_output.bin  : raw decoded prediction tensor (1, 4+nc, num_anchors) f32
       - reference_boxes.json  : human readable list of detections after NMS
       - weights.safetensors   : fused conv weights+biases for every layer in the model
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_sample_image() -> Path:
    """Use the bus.jpg sample bundled with Ultralytics, downloading it if needed."""
    img_path = ARTIFACT_DIR / "bus.jpg"
    if img_path.exists():
        return img_path
    # Ultralytics ships a sample bus.jpg with the package
    candidates = [
        Path("/home/ubuntu/.local/lib/python3.10/site-packages/ultralytics/assets/bus.jpg"),
    ]
    for c in candidates:
        if c.exists():
            import shutil
            shutil.copy(c, img_path)
            return img_path
    # Fallback to ultralytics download
    from ultralytics.utils.downloads import safe_download
    safe_download(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
        dir=str(ARTIFACT_DIR),
    )
    return img_path


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad image to new_shape preserving aspect ratio (Ultralytics letterbox)."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    import cv2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def preprocess(img_path: Path) -> tuple[np.ndarray, dict]:
    import cv2
    raw_bgr = cv2.imread(str(img_path))
    assert raw_bgr is not None, f"Failed to read {img_path}"
    # Letterbox to 640x640 (matches default Ultralytics imgsz)
    lb, r, (pad_x, pad_y) = letterbox(raw_bgr, (640, 640))
    # BGR -> RGB, HWC -> CHW
    rgb = lb[:, :, ::-1].copy()
    chw = rgb.transpose(2, 0, 1)
    f32 = chw.astype(np.float32) / 255.0
    nchw = f32[None, ...]  # (1, 3, 640, 640)
    return nchw, {
        "orig_shape": raw_bgr.shape[:2],
        "ratio": r,
        "pad": (pad_x, pad_y),
    }


def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
    """Folds BatchNorm into a preceding Conv2d, returning a bias-augmented Conv2d."""
    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.data.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    if conv.bias is None:
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    else:
        b_conv = conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.data.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


def export_weights(model: torch.nn.Module, output_path: Path) -> None:
    """After model.fuse(), every Conv block has its bn folded. Export only the
    parameters we actually need on the Rust side, plus pre-split halves of the
    cv1 weights for each block that does ``cv1.chunk(2, 1)`` in PyTorch
    (C3k2, C2PSA). Pre-splitting avoids a luminal cascade-cleanup pitfall where
    a slice (iota+gather) feeding a residual add can leave HLIR ops without
    kernel alternatives."""
    state = {}
    for name, p in model.state_dict().items():
        # Skip helper buffers used by the original Detect head
        if name.endswith("anchors") or name.endswith("strides"):
            continue
        if "num_batches_tracked" in name:
            continue
        if "running_mean" in name or "running_var" in name:
            continue
        state[name] = p.detach().cpu().contiguous().to(torch.float32)

    # ----- Pre-split cv1 weights/biases for chunk-style modules ----- #
    # C3k2 (layers 2, 4, 6, 8, 13, 16, 19, 22) and C2PSA (layer 10) all do
    # `cv1(x).chunk(2, 1)` inside forward. Splitting the weight along the output
    # channel dim into two halves means Rust can run two convs and never has to
    # slice a tensor on the channel dim.
    chunk_layers = [2, 4, 6, 8, 10, 13, 16, 19, 22]
    for layer_idx in chunk_layers:
        w_name = f"model.{layer_idx}.cv1.conv.weight"
        b_name = f"model.{layer_idx}.cv1.conv.bias"
        if w_name not in state:
            continue
        w = state[w_name]
        c2 = w.shape[0]
        assert c2 % 2 == 0, f"{w_name} out_channels {c2} not divisible by 2"
        c = c2 // 2
        state[f"model.{layer_idx}.cv1a.conv.weight"] = w[:c].clone().contiguous()
        state[f"model.{layer_idx}.cv1b.conv.weight"] = w[c:].clone().contiguous()
        if b_name in state:
            b = state[b_name]
            state[f"model.{layer_idx}.cv1a.conv.bias"] = b[:c].clone().contiguous()
            state[f"model.{layer_idx}.cv1b.conv.bias"] = b[c:].clone().contiguous()

    # ----- Pre-split Attention.qkv into q/k/v ----- #
    # Layer 10 has model.10.m.0.attn.qkv that gets split into (q_dim, k_dim, v_dim)
    # along dim 1 (output channels) inside forward. Pre-splitting the conv weights
    # lets us run three small convs and avoid the slicing pattern.
    qkv_w_name = "model.10.m.0.attn.qkv.conv.weight"
    qkv_b_name = "model.10.m.0.attn.qkv.conv.bias"
    if qkv_w_name in state:
        # YOLO v11n: c=128, num_heads=2, head_dim=64, key_dim=32
        # h_total = c + 2*nh*key_dim = 128 + 128 = 256
        # split sizes per-head: key_dim, key_dim, head_dim repeated num_heads times,
        # but PyTorch treats the layout as (num_heads, key_dim*2 + head_dim, ...)
        # which means the conv output (256 channels) is interpreted as
        # (num_heads=2, kd2_hd=128). Inside that slot, the first key_dim=32 is q,
        # next key_dim=32 is k, and last head_dim=64 is v.
        # So the linear order in the 256-channel dim is:
        #   [q_h0 (32), k_h0 (32), v_h0 (64), q_h1 (32), k_h1 (32), v_h1 (64)]
        num_heads, key_dim, head_dim = 2, 32, 64
        per_head = key_dim * 2 + head_dim  # 128
        w_full = state[qkv_w_name]
        # w_full shape: (256, 128, 1, 1) for layer 10
        w_full = w_full.reshape(num_heads, per_head, *w_full.shape[1:])
        q_w = w_full[:, :key_dim].reshape(-1, *w_full.shape[2:]).clone().contiguous()
        k_w = w_full[:, key_dim:2 * key_dim].reshape(-1, *w_full.shape[2:]).clone().contiguous()
        v_w = w_full[:, 2 * key_dim:].reshape(-1, *w_full.shape[2:]).clone().contiguous()
        state["model.10.m.0.attn.q_split.conv.weight"] = q_w
        state["model.10.m.0.attn.k_split.conv.weight"] = k_w
        state["model.10.m.0.attn.v_split.conv.weight"] = v_w
        if qkv_b_name in state:
            b_full = state[qkv_b_name].reshape(num_heads, per_head)
            state["model.10.m.0.attn.q_split.conv.bias"] = b_full[:, :key_dim].reshape(-1).clone().contiguous()
            state["model.10.m.0.attn.k_split.conv.bias"] = b_full[:, key_dim:2 * key_dim].reshape(-1).clone().contiguous()
            state["model.10.m.0.attn.v_split.conv.bias"] = b_full[:, 2 * key_dim:].reshape(-1).clone().contiguous()

    save_file(state, str(output_path))
    print(f"Wrote {output_path} with {len(state)} tensors.")


def dump_named_shapes(model: torch.nn.Module, path: Path) -> None:
    lines = []
    for name, p in model.state_dict().items():
        if "num_batches_tracked" in name:
            continue
        if "running_mean" in name or "running_var" in name:
            continue
        if name.endswith("anchors") or name.endswith("strides"):
            continue
        lines.append(f"{name}\t{tuple(p.shape)}")
    path.write_text("\n".join(lines))
    print(f"Wrote {path} ({len(lines)} entries).")


def main():
    img_path = fetch_sample_image()
    print(f"Sample image: {img_path}")

    yolo = YOLO("yolo11n.pt")
    yolo.model.eval()
    yolo.model.fuse()  # fold BN into conv (creates forward_fuse)

    # Make Detect.export so forward returns the (decoded preds, raw feats) tuple as one tensor
    # Save fused weights
    export_weights(yolo.model, ARTIFACT_DIR / "weights.safetensors")
    dump_named_shapes(yolo.model, ARTIFACT_DIR / "weights_index.txt")

    # Save full architecture summary too for debugging
    (ARTIFACT_DIR / "model_arch.txt").write_text(str(yolo.model))

    # Preprocess
    nchw, meta = preprocess(img_path)
    nchw_t = torch.from_numpy(nchw)

    # Save preprocessed input
    nchw.tofile(ARTIFACT_DIR / "reference_input.bin")
    print(f"Wrote reference_input.bin shape={nchw.shape} dtype={nchw.dtype}")

    # Run model in eval mode -- get raw predictions tensor.
    # In eval mode, Detect.forward returns either y or (y, preds) depending on export.
    yolo.model.eval()
    yolo.model.model[-1].export = True  # only the decoded tensor
    with torch.inference_mode():
        out = yolo.model(nchw_t)
    if isinstance(out, (list, tuple)):
        out = out[0]
    print(f"Model output shape: {tuple(out.shape)}")

    # Save raw decoded predictions: (1, 4+nc, num_anchors) for nc=80 -> (1, 84, 8400)
    arr = out.detach().cpu().numpy().astype(np.float32)
    arr.tofile(ARTIFACT_DIR / "reference_output.bin")
    with open(ARTIFACT_DIR / "reference_output.json", "w") as f:
        json.dump({"shape": list(arr.shape), "dtype": "float32"}, f)
    print(f"Wrote reference_output.bin shape={arr.shape}")

    # Run NMS for sanity-check detections
    nms_in = out
    if nms_in.dim() == 3 and nms_in.shape[1] < nms_in.shape[2]:
        # (B, 4+nc, A) -> NMS expects same layout
        pass
    detections = non_max_suppression(nms_in.clone(), conf_thres=0.25, iou_thres=0.45, max_det=300)[0]
    pretty = []
    coco = yolo.model.names if hasattr(yolo.model, "names") else {}
    for det in detections.tolist():
        x1, y1, x2, y2, conf, cls = det
        # Undo letterbox to original image coordinates
        pad_x, pad_y = meta["pad"]
        r = meta["ratio"]
        ox1 = (x1 - pad_x) / r
        oy1 = (y1 - pad_y) / r
        ox2 = (x2 - pad_x) / r
        oy2 = (y2 - pad_y) / r
        pretty.append({
            "class": int(cls),
            "label": coco.get(int(cls), str(int(cls))),
            "conf": conf,
            "letterbox_xyxy": [x1, y1, x2, y2],
            "orig_xyxy": [ox1, oy1, ox2, oy2],
        })
    with open(ARTIFACT_DIR / "reference_boxes.json", "w") as f:
        json.dump({
            "image": str(img_path),
            "orig_shape": meta["orig_shape"],
            "letterbox_pad": meta["pad"],
            "ratio": meta["ratio"],
            "detections": pretty,
        }, f, indent=2)
    print("Detections:")
    for d in pretty:
        print(f"  {d['label']:>14s}  conf={d['conf']:.3f}  xyxy={[round(v,1) for v in d['orig_xyxy']]}")

    print("Done.")


if __name__ == "__main__":
    main()
