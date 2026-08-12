//! Kind-dispatched staging: fp8 weights stage as NATIVE E4M3FN code
//! buffers (the codes ARE the weights — no conversion); scales and
//! numeric tensors stage f32. Loud bails; no silent skips.

use crate::hf::tensor_to_f32;
use crate::model::{Llama31Fp8, WeightKind};
use luminal::prelude::float8::F8E4M3;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::{Dtype, SafeTensors};
use std::error::Error;
use std::path::Path;

pub fn load_safetensors_weights(
    model: &Llama31Fp8,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_fp8_unfused_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for (key, handle, kind) in model.weight_bindings() {
        let view = tensors
            .tensor(&key)
            .map_err(|e| format!("checkpoint is missing '{key}': {e}"))?;
        let expected: usize = handle
            .dims()
            .iter()
            .map(|d| d.to_usize().expect("model dims are static"))
            .product();
        let numel: usize = view.shape().iter().product::<usize>().max(1);
        if numel != expected {
            return Err(format!(
                "'{key}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                view.shape()
            )
            .into());
        }
        let payload: TypedBuffer = match kind {
            WeightKind::Fp8 => {
                if view.dtype() != Dtype::F8_E4M3 {
                    return Err(format!(
                        "'{key}': expected F8_E4M3 in the checkpoint, found {:?}",
                        view.dtype()
                    )
                    .into());
                }
                view.data()
                    .iter()
                    .map(|byte| F8E4M3::from_bits(*byte))
                    .collect::<Vec<F8E4M3>>()
                    .into()
            }
            WeightKind::Numeric => tensor_to_f32(&view).into(),
        };
        pairs.push((handle.id, payload));
    }
    Ok(pairs)
}

/// Deterministic fake parameters: fp8 weights get in-range codes,
/// scales get 1.0, numerics get the mini-runner formula.
pub fn random_weights(model: &Llama31Fp8) -> Vec<(NodeIndex, TypedBuffer)> {
    model
        .weight_bindings()
        .into_iter()
        .enumerate()
        .map(|(seed, (key, handle, kind))| {
            let n: usize = handle
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("model dims are static"))
                .product::<usize>()
                .max(1);
            let payload: TypedBuffer = match kind {
                WeightKind::Fp8 => (0..n)
                    .map(|i| {
                        let v = (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6;
                        F8E4M3::from_f32(v)
                    })
                    .collect::<Vec<F8E4M3>>()
                    .into(),
                WeightKind::Numeric if key.ends_with("_scale") => vec![1.0f32].into(),
                WeightKind::Numeric => (0..n)
                    .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                    .collect::<Vec<f32>>()
                    .into(),
            };
            (handle.id, payload)
        })
        .collect()
}
