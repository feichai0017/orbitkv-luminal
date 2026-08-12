# yolo_v11

YOLO11n object detection as pure logical ops.

Points at ultralytics YOLO11n (the nano detection checkpoint). The anatomy
is 100% of the real model: the C3k2/C2PSA backbone-neck, SPPF, the
three-scale detect head with DFL (reg_max 16), and the anchor/stride
decode.

## The per-scale DFL respelling

The parked CUDA-era model concatenated the three scale outputs first and
ran DFL + dist2bbox on the concatenated tensor by slicing it back apart.
That is the exact `slice-downstream-of-concat` shape that detonates egglog
saturation (the rejoin-divergence family). Here the DFL
softmax-expectation, the lt/rb split, and dist2bbox run **per scale**; the
final `(1, 84, 8400)` assembly concat feeds outputs only — nothing slices
a concat.

Anchors and strides are host-fed inputs (`yolo.anchors.{i}` of shape
`(2, s*s)`, `yolo.strides.{i}` of shape `(1, s*s)`), not baked constants,
so the graph stays resolution-honest.

## Runtime status

`tests/smoke.rs::full_graph_records_cleanly` proves the full ~2,200-node
graph records into a native egglog program (sub-second; always on).

Saturation + search + execution of the full 640×640 net is a documented
heavy path: the bounded probe (`saturation_probe`, `#[ignore]`d) was
killed by a 3 GB RSS watchdog after ~4 minutes on 2026-08-12, consistent
with the parked-era report (>10 min / 30+ GB). Run it only by name, under
a watchdog, on a machine with headroom:

```sh
cargo test --release -p yolo_v11 --test smoke saturation_probe -- --ignored
```

`main.rs` carries the full pipeline (letterbox → search → execute → NMS →
box drawing) for such an attended run.
