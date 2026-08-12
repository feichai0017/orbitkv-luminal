//! Staging straight from the HF single-shard checkpoint (f32/f16 —
//! no combine step needed), keyed by the model's exact HF names.

use crate::model::Whisper;
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

pub fn load_safetensors_weights(
    model: &Whisper,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model.safetensors");
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
        pairs.push((handle.id, view_to_f32(&view).into()));
    }
    Ok(pairs)
}

pub fn random_weights(model: &Whisper) -> Vec<(NodeIndex, TypedBuffer)> {
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
