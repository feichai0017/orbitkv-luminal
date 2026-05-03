//! Minimal correctness probe: run only the very first Conv layer (3->16, k=3, s=2)
//! and compare against PyTorch's eager output. Used to validate the conv2d
//! implementation before we attempt the full YOLO graph.
//!
//! Run with:  cargo run -p yolo_v11 --release --bin yolo_v11_tiny

use std::{fs::File, io::Read, path::PathBuf, time::Instant};

use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};

#[path = "model.rs"]
mod model;
use model::*;

fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let mut f = File::open(path).expect("Failed to open binary file");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut data = Vec::with_capacity(buf.len() / 4);
    let mut chunk = [0u8; 4];
    for i in 0..buf.len() / 4 {
        chunk.copy_from_slice(&buf[i * 4..(i + 1) * 4]);
        data.push(f32::from_le_bytes(chunk));
    }
    data
}

fn main() {
    let cwd = std::env::current_dir().unwrap();
    let artifact_dir = cwd.join("examples/yolo_v11/artifacts");

    let weights_path = artifact_dir.join("weights.safetensors");
    let input_path = artifact_dir.join("reference_input.bin");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let n_layers: usize = std::env::var("YOLO_TINY_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    println!("Building YOLO subset with {n_layers} backbone layers");
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let conv0 = Conv::new("model.0.conv", 3, C0, 3, 2, 1, &mut cx);
    let mut x = conv0.forward(img);
    if n_layers >= 2 {
        let conv1 = Conv::new("model.1.conv", C0, C1, 3, 2, 1, &mut cx);
        x = conv1.forward(x);
    }
    if n_layers >= 3 {
        let c3k2_2 = C3k2::new("model.2", C1, C2, 1, false, 0.25, true, &mut cx);
        x = c3k2_2.forward(x);
    }
    if n_layers >= 4 {
        let conv3 = Conv::new("model.3.conv", C2, C2, 3, 2, 1, &mut cx);
        x = conv3.forward(x);
    }
    if n_layers >= 5 {
        let c3k2_4 = C3k2::new("model.4", C2, C3, 1, false, 0.25, true, &mut cx);
        x = c3k2_4.forward(x);
    }
    if n_layers >= 6 {
        let conv5 = Conv::new("model.5.conv", C3, C3, 3, 2, 1, &mut cx);
        x = conv5.forward(x);
    }
    if n_layers >= 7 {
        let c3k2_6 = C3k2::new("model.6", C3, C3, 1, true, 0.5, true, &mut cx);
        x = c3k2_6.forward(x);
    }
    if n_layers >= 8 {
        let conv7 = Conv::new("model.7.conv", C3, C4, 3, 2, 1, &mut cx);
        x = conv7.forward(x);
    }
    if n_layers >= 9 {
        let c3k2_8 = C3k2::new("model.8", C4, C4, 1, true, 0.5, true, &mut cx);
        x = c3k2_8.forward(x);
    }
    if n_layers >= 10 {
        let sppf = Sppf::new("model.9", C4, C4, 5, &mut cx);
        x = sppf.forward(x);
    }
    if n_layers >= 11 {
        let c2psa = C2psa::new("model.10", C4, C4, 1, 0.5, &mut cx);
        x = c2psa.forward(x);
    }
    if n_layers >= 12 {
        x = upsample_2x(x);
    }
    if n_layers >= 13 {
        let conv11 = Conv::new("model.13.cv1.conv", C4, C4 / 2, 1, 1, 0, &mut cx);
        x = conv11.forward(x);
    }
    let logits = x.output();

    println!("Building E-Graph...");
    let t0 = Instant::now();
    cx.build_search_space::<CudaRuntime>();
    println!("  built E-Graph in {:?}", t0.elapsed());

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let img_data = read_f32_bin(&input_path);
    runtime.set_data(img, img_data);

    println!("Compiling (search_graphs=1)...");
    let t0 = Instant::now();
    runtime = cx.search(runtime, 1);
    println!("  search took {:?}", t0.elapsed());

    let img_data = read_f32_bin(&input_path);
    runtime.set_data(img, img_data);

    println!("Executing...");
    let t0 = Instant::now();
    runtime.execute(&cx.dyn_map);
    println!("  forward took {:?}", t0.elapsed());

    let out = runtime.get_f32(logits);
    println!("Output buffer len {}", out.len());
    println!("First 16 outputs: {:?}", &out[..16.min(out.len())]);
    println!(
        "Stats: min={:.4} max={:.4} mean={:.4}",
        out.iter().cloned().fold(f32::INFINITY, f32::min),
        out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        out.iter().sum::<f32>() / out.len() as f32
    );
}
