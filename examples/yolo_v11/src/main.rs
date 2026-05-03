mod model;

use std::{fs::File, io::Read, path::PathBuf, time::Instant};

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const ARTIFACT_DIR: &str = "examples/yolo_v11/artifacts";

fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let mut f = File::open(path).expect("Failed to open binary file");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(buf.len() % 4, 0, "binary file size must be multiple of 4");
    let mut data = Vec::with_capacity(buf.len() / 4);
    let mut chunk = [0u8; 4];
    for i in 0..buf.len() / 4 {
        chunk.copy_from_slice(&buf[i * 4..(i + 1) * 4]);
        data.push(f32::from_le_bytes(chunk));
    }
    data
}

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let search_graphs: usize = std::env::var("YOLO_SEARCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    println!(
        "NOTE: the full YOLO v11n graph is large (~2200 HLIR nodes). The current\n\
         luminal_cuda_lite e-graph rewrite phase can take many minutes to converge\n\
         on this many nodes. If you want a fast smoke-test, run `yolo_v11_tiny`\n\
         instead, which compiles only the first three layers."
    );

    let cwd = std::env::current_dir().unwrap();
    let artifact_dir = cwd.join(ARTIFACT_DIR);
    println!("Using artifact directory: {}", artifact_dir.display());

    let weights_path = artifact_dir.join("weights.safetensors");
    let input_path = artifact_dir.join("reference_input.bin");
    let output_path = artifact_dir.join("reference_output.bin");

    assert!(weights_path.exists(), "Missing {:?}; run python/reference.py first", weights_path);
    assert!(input_path.exists(), "Missing {:?}; run python/reference.py first", input_path);
    assert!(output_path.exists(), "Missing {:?}; run python/reference.py first", output_path);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Build graph
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let yolo = YoloV11::init(&mut cx);
    let logits = yolo.forward(img).output();

    println!("Building E-Graph...");
    let t0 = Instant::now();
    cx.build_search_space::<CudaRuntime>();
    println!("  built E-Graph in {:?}", t0.elapsed());

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // Initialize anchors, strides, and DFL constant.
    let (anchors_flat, strides_flat) = make_anchors_and_strides(&[80, 40, 20], &STRIDES);
    runtime.set_data(yolo.detect.anchors, anchors_flat.clone());
    runtime.set_data(yolo.detect.strides, strides_flat.clone());
    runtime.set_data(yolo.detect.dfl_weight, dfl_weight());

    // Read input image
    let img_data = read_f32_bin(&input_path);
    let expected_input = 1 * 3 * IMG_SIZE * IMG_SIZE;
    assert_eq!(img_data.len(), expected_input, "input size mismatch");
    runtime.set_data(img, img_data);

    println!("Compiling (search_graphs={search_graphs})...");
    let t0 = Instant::now();
    runtime = cx.search(runtime, search_graphs);
    println!("  search took {:?}", t0.elapsed());

    // Re-set anchors/strides/dfl/img after search (search may consume the inputs)
    runtime.set_data(yolo.detect.anchors, anchors_flat);
    runtime.set_data(yolo.detect.strides, strides_flat);
    runtime.set_data(yolo.detect.dfl_weight, dfl_weight());
    let img_data = read_f32_bin(&input_path);
    runtime.set_data(img, img_data);

    println!("Executing...");
    let t0 = Instant::now();
    runtime.execute(&cx.dyn_map);
    let elapsed = t0.elapsed();
    println!("  forward took {:?}", elapsed);

    // Get output (1, 4 + NC, 8400) — Detect with export=True returns the
    // DECODED predictions (4 box coords + NC class scores), not the raw
    // (NC + REG_MAX*4) channels.
    let out = runtime.get_f32(logits);
    let total_anchors: usize = 80 * 80 + 40 * 40 + 20 * 20;
    let expected_out_len = 1 * (4 + NC) * total_anchors;
    println!(
        "  output buffer length: {} (expected {} for shape (1, {}, {}))",
        out.len(),
        expected_out_len,
        4 + NC,
        total_anchors
    );
    let out = &out[..expected_out_len];

    // Load reference output
    let ref_out = read_f32_bin(&output_path);
    assert_eq!(ref_out.len(), expected_out_len, "reference output size mismatch");

    // Compute element-wise difference statistics
    let (mut max_abs, mut sum_abs) = (0.0_f32, 0.0_f64);
    let mut argmax_idx = 0usize;
    for (i, (a, b)) in out.iter().zip(ref_out.iter()).enumerate() {
        let d = (a - b).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            argmax_idx = i;
        }
    }
    let mean_abs = sum_abs / expected_out_len as f64;
    println!(
        "Comparison vs Python reference: max_abs={:.6} mean_abs={:.6e}  (worst at idx {} our={} ref={})",
        max_abs, mean_abs, argmax_idx, out[argmax_idx], ref_out[argmax_idx]
    );

    // Show top detections (greedy max class per anchor)
    print_top_detections(out, total_anchors);
}

fn print_top_detections(out: &[f32], total_anchors: usize) {
    // Layout: (1, NO, A) flat. We iterate columns (anchors).
    let nc = NC;
    let mut detections = Vec::new();
    for a in 0..total_anchors {
        // box xywh
        let cx = out[0 * total_anchors + a];
        let cy = out[1 * total_anchors + a];
        let w = out[2 * total_anchors + a];
        let h = out[3 * total_anchors + a];
        let mut best_score = 0.0_f32;
        let mut best_class = 0usize;
        for c in 0..nc {
            let s = out[(4 + c) * total_anchors + a];
            if s > best_score {
                best_score = s;
                best_class = c;
            }
        }
        if best_score > 0.25 {
            detections.push((best_score, best_class, cx, cy, w, h));
        }
    }
    detections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("Top {} pre-NMS detections (conf > 0.25):", detections.len().min(10));
    let coco_names = coco_names();
    for det in detections.iter().take(10) {
        let (s, c, cx, cy, w, h) = *det;
        let name = coco_names.get(c).copied().unwrap_or("?");
        let x1 = cx - w / 2.0;
        let y1 = cy - h / 2.0;
        let x2 = cx + w / 2.0;
        let y2 = cy + h / 2.0;
        println!(
            "  conf={:.3} class={:>14}  xyxy=[{:.1}, {:.1}, {:.1}, {:.1}]",
            s, name, x1, y1, x2, y2
        );
    }
}

fn coco_names() -> [&'static str; NC] {
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
}
