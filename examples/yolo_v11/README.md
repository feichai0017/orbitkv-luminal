# YOLO v11n on Luminal

End-to-end Object detection demo running the Ultralytics yolo11n model on the
`luminal_cuda_lite` backend.

## Layout

```
examples/yolo_v11/
├── Cargo.toml              # Rust crate (binaries: yolo_v11, yolo_v11_tiny)
├── src/
│   ├── main.rs             # Full forward + reference comparison + greedy detections
│   ├── main_tiny.rs        # Smoke-test for the first three layers (Conv, Conv, C3k2)
│   └── model.rs            # YOLO v11n architecture in luminal IR
├── python/
│   ├── reference.py        # PyTorch eager reference + weight prep
│   └── luminal_example.py  # torch.compile(..., backend=luminal_backend) demo
└── artifacts/              # Produced by python/reference.py (gitignored)
    ├── bus.jpg
    ├── reference_input.bin
    ├── reference_output.bin
    ├── reference_boxes.json
    └── weights.safetensors
```

## Quick start

1. **Generate reference data + fused weights** (PyTorch + Ultralytics):

   ```bash
   pip install ultralytics torch opencv-python-headless
   python examples/yolo_v11/python/reference.py
   ```

   This downloads `yolo11n.pt`, fuses Conv + BN, runs the eager forward on a
   bundled bus image, and writes `examples/yolo_v11/artifacts/`.

2. **Run the Rust example** (CUDA, e.g. on a GH200 / H100):

   ```bash
   # Quick smoke test (first three layers, compiles in ~3s)
   cargo run --release -p yolo_v11 --bin yolo_v11_tiny

   # Full model on the default bus.jpg sample
   cargo run --release -p yolo_v11 --bin yolo_v11

   # Full model on any JPEG or PNG
   cargo run --release -p yolo_v11 --bin yolo_v11 -- --input /path/to/image.jpg --output /tmp/yolo_annotated.png

   # Positional shorthand: <input> <output>
   cargo run --release -p yolo_v11 --bin yolo_v11 -- /path/to/image.jpg /tmp/yolo_annotated.png
   ```

   `yolo_v11_tiny` loads `weights.safetensors`, runs only `model[0]`, `model[1]`,
   and `model[2]`, and reports the (1, 64, 160, 160) intermediate. This output
   matches PyTorch's eager forward exactly (within f32 rounding).

   `yolo_v11` builds the entire YOLO v11n graph and the Detect head, preprocesses
   a JPEG/PNG with a Rust implementation of the 640x640 Ultralytics-style
   letterbox transform, runs the forward, applies class-aware NMS, and prints
   detections in the original image coordinates. For image inputs, it also writes
   an annotated PNG to `examples/yolo_v11/artifacts/annotated.png` by default.
   The input and annotated output paths can be supplied as CLI arguments:
   `--input /path/to/image.png --output /path/to/out.png`.

   For exact regression against the saved PyTorch tensor, run:

   ```bash
   YOLO_INPUT_BIN=1 YOLO_COMPARE_REF=1 cargo run --release -p yolo_v11 --bin yolo_v11
   ```

   The exact regression path feeds the saved PyTorch-preprocessed tensor and is
   the strict check for Luminal-vs-PyTorch model output parity. The direct image
   path may differ slightly from Python/OpenCV preprocessing because it uses
   Rust image decoding and resizing.

3. **(Optional) Run the Python compiled-model example**:

   Requires `luminal_python` built with the cuda feature (see
   `crates/luminal_python/run_tests_cuda.sh`).

   ```bash
   python examples/yolo_v11/python/luminal_example.py
   ```

   The pytest version is `crates/luminal_python/tests/test_yolo_v11.py`.

## Implementation notes

* All Conv blocks are loaded with `bn` folded into a bias-augmented Conv2d
  (`forward_fuse`), so the saved tensors are just `<layer>.conv.weight` and
  `<layer>.conv.bias`.
* The `C3k2`, `C3k`, `C2PSA`, and `Attention` modules in PyTorch use
  `tensor.chunk(2, dim=1)` (or `qkv.split([...], dim=...)`) to produce two/three
  channel-slices that then take separate paths. Slicing followed by a residual
  add inside a bottleneck triggers a cascade in luminal_cuda_lite's e-graph
  cleanup that prunes the only kernel alternatives. To work around this, the
  Python script pre-splits those conv weights along the output-channel dim and
  the Rust model exposes them as separate convs (`cv1a`/`cv1b` for C3k2/C2PSA,
  `q_split`/`k_split`/`v_split` for Attention).
* Anchors, per-anchor strides, and the DFL projection weight are fed from Rust
  via `runtime.set_data`. The DFL projection is the constant `arange(reg_max)`.
* `make_contiguous` (a free function in `src/model.rs`) materializes a
  non-contiguous view via `gather + iota` (the same trick `GraphTensor::output`
  uses internally). It's applied wherever an op chain produces a strided view
  that the next op needs to read sequentially.
* 1x1 convolutions skip the unfold path and use a direct 2D matmul, so
  luminal_cuda_lite's `TileMatmulFullSplit` kernel can match.

## Known limitation: full-model compile time

The `yolo_v11` binary builds a graph of ~2,200 HLIR nodes (~100 convolutions
plus the Detect head). luminal_cuda_lite's e-graph rewrite phase runs many
rules to fixpoint over the whole graph, which on a conv-heavy vision model
becomes the dominant cost. On a Grace-Hopper class machine this phase can
take >10 minutes (using ~30+ GB of host RAM in the egglog tables) before the
search and execution finally proceed.

If you only want to verify the model is wired up correctly, prefer
`yolo_v11_tiny` — it compiles in seconds and exercises the same code paths
(Conv2d, depthwise, bias, SiLU, slicing, residuals, concat, C3k2). The full
model is mathematically identical and the Python torch.compile path
(`crates/luminal_python/tests/test_yolo_v11.py`) is a useful alternative
because the pt2 export decomposes the graph slightly differently.
