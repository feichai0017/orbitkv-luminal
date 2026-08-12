//! Host-side weight staging: the combined bf16 checkpoint converts to
//! f32 on the way in (the parameters ARE f32-typed in this model) —
//! per-tensor, keyed by [`crate::model::Qwen::weight_bindings`]. Loud
//! bails on a missing key or a shape mismatch; no silent skips.

use crate::hf::tensor_to_f32;
use crate::model::Qwen;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Read `model_combined_bf16_v1.safetensors` (the `hf.rs` artifact) and
/// produce the (tensor id, f32 data) pairs for every model parameter.
pub fn load_safetensors_weights(
    model: &Qwen,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_bf16_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for (key, handle) in model.weight_bindings() {
        let view = tensors
            .tensor(&key)
            .map_err(|e| format!("checkpoint is missing '{key}': {e}"))?;
        let expected: usize = handle
            .dims()
            .iter()
            .map(|d| d.to_usize().expect("model dims are static"))
            .product();
        let numel: usize = view.shape().iter().product();
        if numel != expected {
            return Err(format!(
                "'{key}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                view.shape()
            )
            .into());
        }
        pairs.push((handle.id, tensor_to_f32(&view).into()));
    }
    Ok(pairs)
}

/// Deterministic pseudo-random parameters (the mini-runner formula) for
/// offline runs and the smoke tests — anatomy-real, weights fake.
pub fn random_weights(model: &Qwen) -> Vec<(NodeIndex, TypedBuffer)> {
    model
        .weight_bindings()
        .into_iter()
        .enumerate()
        .map(|(seed, (_, handle))| {
            let n: usize = handle
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("model dims are static"))
                .product();
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                .collect();
            (handle.id, data.into())
        })
        .collect()
}
