//! Standalone correctness probe for one YOLO convolution.
//!
//! By default this runs `model.2.m.0.cv1` on the saved PyTorch `model.2.cv1b`
//! activation and compares against the saved PyTorch output for the same block.

use std::{fs::File, io::Read, path::PathBuf, time::Instant};

use luminal::hlir::{Input, NativeRuntime};
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use safetensors::SafeTensors;

#[path = "model.rs"]
mod model;
use model::Conv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Probe {
    Standalone,
}

impl Probe {
    fn input_shape(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Standalone => (1, 16, 160, 160),
        }
    }

    fn input_file(self, artifact_dir: &std::path::Path) -> PathBuf {
        match self {
            Self::Standalone => artifact_dir.join("model_2_cv1b.bin"),
        }
    }

    fn reference_file(self, artifact_dir: &std::path::Path) -> PathBuf {
        match self {
            Self::Standalone => artifact_dir.join("model_2_m_0_cv1.bin"),
        }
    }
}

fn read_f32_bin(path: &PathBuf) -> Vec<f32> {
    let mut f = File::open(path).expect("failed to open binary file");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(buf.len() % 4, 0, "{} is not f32-aligned", path.display());
    bytemuck::cast_slice::<u8, f32>(&buf).to_vec()
}

fn compare_reference(out: &[f32], path: &PathBuf) {
    let ref_out = read_f32_bin(path);
    assert_eq!(
        out.len(),
        ref_out.len(),
        "reference length mismatch for {}",
        path.display()
    );

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
    let c = argmax_idx / (160 * 160);
    let rem = argmax_idx % (160 * 160);
    let y = rem / 160;
    let x = rem % 160;
    println!(
        "Comparison vs {}: max_abs={:.6} mean_abs={:.6e} worst idx {} (c={}, y={}, x={}) our={} ref={}",
        path.display(),
        max_abs,
        sum_abs / out.len() as f64,
        argmax_idx,
        c,
        y,
        x,
        out[argmax_idx],
        ref_out[argmax_idx]
    );
}

fn input_labels(cx: &Graph) -> Vec<(NodeIndex, String)> {
    cx.graph
        .node_indices()
        .filter_map(|node| {
            (*cx.graph[node])
                .as_any()
                .downcast_ref::<Input>()
                .map(|input| (node, input.label.clone()))
        })
        .collect()
}

fn load_safetensors_native(
    labels: &[(NodeIndex, String)],
    runtime: &mut NativeRuntime,
    file_path: &PathBuf,
) {
    let f = File::open(file_path).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&f).unwrap() };
    let st = SafeTensors::deserialize(&mmap).unwrap();
    for (node, label) in labels {
        if let Ok(tensor) = st.tensor(label) {
            assert_eq!(
                tensor.dtype(),
                safetensors::Dtype::F32,
                "{label} is not f32"
            );
            runtime.set_data(*node, bytemuck::cast_slice::<u8, f32>(tensor.data()));
        }
    }
}

fn main() {
    let cwd = std::env::current_dir().unwrap();
    let artifact_dir = cwd.join("examples/yolo_v11/artifacts");
    let weights_path = artifact_dir.join("weights.safetensors");
    let probe = Probe::Standalone;
    let input_path = probe.input_file(&artifact_dir);
    let reference_path = probe.reference_file(&artifact_dir);
    let use_native = false;

    println!("Running {probe:?} probe");
    let mut cx = Graph::default();
    let input = cx.named_tensor("probe.input", probe.input_shape());
    let output_tensor = match probe {
        Probe::Standalone => {
            let conv = Conv::new("model.2.m.0.cv1.conv", 16, 8, 3, 1, 1, &mut cx);
            conv.forward(input)
        }
    };
    let output = output_tensor.output();
    let labels = input_labels(&cx);

    if use_native {
        println!("Building native E-Graph...");
        let t0 = Instant::now();
        cx.build_search_space::<NativeRuntime>();
        println!("  built E-Graph in {:?}", t0.elapsed());

        println!("Compiling native graph...");
        let t0 = Instant::now();
        let mut runtime = cx.search(NativeRuntime::default(), 1);
        println!("  search took {:?}", t0.elapsed());

        load_safetensors_native(&labels, &mut runtime, &weights_path);
        runtime.set_data(input, read_f32_bin(&input_path));

        println!("Executing native...");
        let t0 = Instant::now();
        runtime.execute(&cx.dyn_map);
        println!("  forward took {:?}", t0.elapsed());

        let out = runtime.get_f32(output);
        println!("Output buffer len {}", out.len());
        println!("First 16 outputs: {:?}", &out[..16.min(out.len())]);
        compare_reference(out, &reference_path);
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    println!("Building CUDA E-Graph...");
    let t0 = Instant::now();
    cx.build_search_space::<CudaRuntime>();
    println!("  built E-Graph in {:?}", t0.elapsed());

    println!("Compiling CUDA graph...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    runtime.set_data(input, read_f32_bin(&input_path));
    let t0 = Instant::now();
    runtime = cx.search(runtime, 1);
    println!("  search took {:?}", t0.elapsed());

    runtime.set_data(input, read_f32_bin(&input_path));

    println!("Executing CUDA...");
    let t0 = Instant::now();
    runtime.execute(&cx.dyn_map);
    println!("  forward took {:?}", t0.elapsed());

    let out = runtime.get_f32(output);
    println!("Output buffer len {}", out.len());
    println!("First 16 outputs: {:?}", &out[..16.min(out.len())]);
    compare_reference(&out, &reference_path);
}
