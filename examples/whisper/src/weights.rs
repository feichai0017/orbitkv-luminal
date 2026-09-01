//! Label-driven staging straight from the HF single-shard checkpoint
//! (f32/f16 — no combine step needed). Every checkpoint-backed input's
//! LABEL equals its HF key, so staging enumerates
//! `cx.logical.input_specs()` and stages each spec whose label is a
//! checkpoint key. Inputs whose labels are NOT checkpoint keys (the
//! anonymous runtime inputs `arg.*` — mel, token, q_pos, gather /
//! scatter indices — and the `cache.*` slot pools) are runtime inputs,
//! staged by handle in the transcribe loop.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

fn view_to_f32(view: &safetensors::tensor::TensorView) -> Vec<f32> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
        Dtype::F16 => bytemuck::cast_slice::<u8, half::f16>(view.data())
            .iter()
            .map(|x| x.to_f32())
            .collect(),
        other => panic!("unsupported checkpoint dtype {other:?}"),
    }
}

/// A runtime input: staged by handle in the transcribe loop, never
/// checkpoint-backed (and never junk-filled — several are Int).
fn is_runtime_input(label: &str) -> bool {
    label.starts_with("arg.") || label.starts_with("cache.")
}

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model.safetensors");
    let file = std::fs::File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        let label = &spec.label;
        let Ok(view) = tensors.tensor(label) else {
            // Not a checkpoint key: a runtime input (arg.*, cache.*),
            // staged by handle in the transcribe loop — unless it
            // claims otherwise.
            if is_runtime_input(label) {
                continue;
            }
            return Err(format!("checkpoint is missing '{label}'").into());
        };
        let expected: usize = spec
            .dims
            .iter()
            .map(|d| d.to_usize().expect("model dims are static"))
            .product();
        let numel: usize = view.shape().iter().product();
        if numel != expected {
            return Err(format!(
                "'{label}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                view.shape()
            )
            .into());
        }
        match spec.dtype {
            DType::F32 => pairs.push((spec.id, view_to_f32(&view).into())),
            other => {
                return Err(format!("'{label}': unsupported model input dtype {other:?}").into());
            }
        }
    }
    Ok(pairs)
}

pub fn random_weights(cx: &Graph) -> Vec<(NodeIndex, TypedBuffer)> {
    cx.logical
        .input_specs()
        .into_iter()
        .filter(|spec| !is_runtime_input(&spec.label))
        .enumerate()
        .map(|(seed, spec)| {
            assert_eq!(
                spec.dtype,
                DType::F32,
                "'{}': junk fill only covers F32 weights",
                spec.label
            );
            let n: usize = spec
                .dims
                .iter()
                .map(|d| d.to_usize().expect("model dims are static"))
                .product();
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                .collect();
            (spec.id, data.into())
        })
        .collect()
}
