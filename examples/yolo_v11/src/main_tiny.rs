//! Correctness probe for the first three YOLO model layers.
//!
//! Run with:
//!   cargo run -p yolo_v11 --release --bin yolo_v11_tiny

use std::{fs::File, io::Read, path::PathBuf, time::Instant};

use luminal::hlir::{Input, NativeRuntime};
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use safetensors::SafeTensors;

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

fn load_safetensors_native(
    input_labels: &[(NodeIndex, String)],
    runtime: &mut NativeRuntime,
    file_path: &PathBuf,
) {
    let f = File::open(file_path).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&f).unwrap() };
    let st = SafeTensors::deserialize(&mmap).unwrap();
    for (node, label) in input_labels {
        if let Ok(tensor) = st.tensor(label) {
            assert_eq!(
                tensor.dtype(),
                safetensors::Dtype::F32,
                "{label} is not f32"
            );
            let data: &[f32] = bytemuck::cast_slice(tensor.data());
            runtime.set_data(*node, data);
        }
    }
}

fn main() {
    let cwd = std::env::current_dir().unwrap();
    let artifact_dir = cwd.join("examples/yolo_v11/artifacts");

    let weights_path = artifact_dir.join("weights.safetensors");
    let input_path = artifact_dir.join("reference_input.bin");

    let n_layers = 3usize;
    let probe: Option<String> = None;
    let use_native = false;
    println!("Building YOLO subset with first {n_layers} model layers");
    let mut cx = Graph::default();
    let img = cx.named_tensor("input.image", (1usize, 3usize, IMG_SIZE, IMG_SIZE));
    let conv0 = Conv::new("model.0.conv", 3, C0, 3, 2, 1, &mut cx);
    let mut x = conv0.forward(img);
    let mut l4 = None;
    let mut l6 = None;
    let mut l10 = None;
    let mut l13 = None;
    let mut l16 = None;
    let mut l19 = None;
    if n_layers >= 2 {
        let conv1 = Conv::new("model.1.conv", C0, C1, 3, 2, 1, &mut cx);
        x = conv1.forward(x);
    }
    if n_layers >= 3 {
        let c3k2_2 = C3k2::new("model.2", C1, C2, 1, false, 0.25, true, &mut cx);
        x = match probe.as_deref() {
            Some("model.2.cv1a") => c3k2_2.cv1a.forward(x),
            Some("model.2.cv1b") => c3k2_2.cv1b.forward(x),
            Some("model.2.m.0.cv1") => {
                let b = c3k2_2.cv1b.forward(x);
                let C3k2Inner::Bottleneck(bn) = &c3k2_2.m[0] else {
                    unreachable!()
                };
                bn.cv1.forward(b)
            }
            Some("model.2.m.0.cv2") => {
                let b = c3k2_2.cv1b.forward(x);
                let C3k2Inner::Bottleneck(bn) = &c3k2_2.m[0] else {
                    unreachable!()
                };
                bn.cv2.forward(bn.cv1.forward(b))
            }
            Some("model.2.m.0.add") => {
                let b = c3k2_2.cv1b.forward(x);
                let C3k2Inner::Bottleneck(bn) = &c3k2_2.m[0] else {
                    unreachable!()
                };
                b + bn.cv2.forward(bn.cv1.forward(b))
            }
            Some("model.2.cat") => {
                let a = c3k2_2.cv1a.forward(x);
                let b = c3k2_2.cv1b.forward(x);
                let C3k2Inner::Bottleneck(bn) = &c3k2_2.m[0] else {
                    unreachable!()
                };
                let m = b + bn.cv2.forward(bn.cv1.forward(b));
                make_contiguous(a.concat_along(b, 1).concat_along(m, 1))
            }
            _ => c3k2_2.forward(x),
        };
    }
    if n_layers >= 4 {
        let conv3 = Conv::new("model.3.conv", C2, C2, 3, 2, 1, &mut cx);
        x = conv3.forward(x);
    }
    if n_layers >= 5 {
        let c3k2_4 = C3k2::new("model.4", C2, C3, 1, false, 0.25, true, &mut cx);
        x = c3k2_4.forward(x);
        l4 = Some(x);
    }
    if n_layers >= 6 {
        let conv5 = Conv::new("model.5.conv", C3, C3, 3, 2, 1, &mut cx);
        x = conv5.forward(x);
    }
    if n_layers >= 7 {
        let c3k2_6 = C3k2::new("model.6", C3, C3, 1, true, 0.5, true, &mut cx);
        x = c3k2_6.forward(x);
        l6 = Some(x);
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
        l10 = Some(x);
    }
    if n_layers >= 12 {
        x = upsample_2x(x);
    }
    if n_layers >= 13 {
        x = make_contiguous(x.concat_along(l6.expect("layer 6 output required"), 1));
    }
    if n_layers >= 14 {
        let c3k2_13 = C3k2::new("model.13", C4 + C3, C3, 1, false, 0.5, true, &mut cx);
        x = c3k2_13.forward(x);
        l13 = Some(x);
    }
    if n_layers >= 15 {
        x = upsample_2x(x);
    }
    if n_layers >= 16 {
        x = make_contiguous(x.concat_along(l4.expect("layer 4 output required"), 1));
    }
    if n_layers >= 17 {
        let c3k2_16 = C3k2::new("model.16", C3 + C3, C2, 1, false, 0.5, true, &mut cx);
        x = c3k2_16.forward(x);
        l16 = Some(x);
    }
    if n_layers >= 18 {
        let conv17 = Conv::new("model.17.conv", C2, C2, 3, 2, 1, &mut cx);
        x = conv17.forward(x);
    }
    if n_layers >= 19 {
        x = make_contiguous(x.concat_along(l13.expect("layer 13 output required"), 1));
    }
    if n_layers >= 20 {
        let c3k2_19 = C3k2::new("model.19", C2 + C3, C3, 1, false, 0.5, true, &mut cx);
        x = c3k2_19.forward(x);
        l19 = Some(x);
    }
    if n_layers >= 21 {
        let conv20 = Conv::new("model.20.conv", C3, C3, 3, 2, 1, &mut cx);
        x = conv20.forward(x);
    }
    if n_layers >= 22 {
        x = make_contiguous(x.concat_along(l10.expect("layer 10 output required"), 1));
    }
    if n_layers >= 23 {
        let c3k2_22 = C3k2::new("model.22", C3 + C4, C4, 1, true, 0.5, true, &mut cx);
        x = c3k2_22.forward(x);
    }
    if n_layers >= 24 {
        let detect = Detect::new("model.23", &[C2, C3, C4], &[80, 40, 20], &mut cx);
        x = detect.forward(&[
            l16.expect("layer 16 output required"),
            l19.expect("layer 19 output required"),
            x,
        ]);
    }
    let logits = x.output();
    let input_labels = cx
        .graph
        .node_indices()
        .filter_map(|node| {
            (*cx.graph[node])
                .as_any()
                .downcast_ref::<Input>()
                .map(|input| (node, input.label.clone()))
        })
        .collect::<Vec<_>>();

    if use_native {
        println!("Building native E-Graph...");
        let t0 = Instant::now();
        cx.build_search_space::<NativeRuntime>();
        println!("  built E-Graph in {:?}", t0.elapsed());

        println!("Compiling native graph...");
        let t0 = Instant::now();
        let mut runtime = cx.search(NativeRuntime::default(), 1);
        println!("  search took {:?}", t0.elapsed());

        println!("Loading weights...");
        load_safetensors_native(&input_labels, &mut runtime, &weights_path);
        let img_data = read_f32_bin(&input_path);
        runtime.set_data(img, img_data);

        println!("Executing native...");
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
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    println!("Building E-Graph...");
    let t0 = Instant::now();
    cx.build_search_space::<CudaRuntime>();
    println!("  built E-Graph in {:?}", t0.elapsed());

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    if n_layers >= 24 {
        let (anchors_flat, strides_flat) = make_anchors_and_strides(&[80, 40, 20], &STRIDES);
        for node in cx.graph.node_indices() {
            let Some(input) = (*cx.graph[node])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
            else {
                continue;
            };
            match input.label.as_str() {
                "yolo.anchors" => runtime.set_data(node, anchors_flat.clone()),
                "yolo.strides" => runtime.set_data(node, strides_flat.clone()),
                "model.23.dfl.conv.weight" => runtime.set_data(node, dfl_weight()),
                _ => {}
            }
        }
    }

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
