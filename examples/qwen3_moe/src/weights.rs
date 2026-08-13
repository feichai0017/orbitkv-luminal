//! Host-side weight staging, LABEL-DRIVEN (ruling 2026-08-13): the
//! graph's own input interface (`cx.logical.input_specs()`) is the
//! staging key. Every checkpoint-backed tensor's label IS its HF
//! checkpoint key, so any spec whose label appears in the combined
//! bf16 checkpoint (unfused, norms F32) stages by `spec.id`,
//! converting to f32 on the way in. Specs whose labels are NOT
//! checkpoint keys (token, q_pos, rope tables, gather/scatter,
//! kv_cache.*) are runtime inputs — the decode loop stages them by
//! handle. Loud bails on a shape mismatch or an unsupported dtype;
//! no hand-written key maps.

use crate::DecodeStep;
use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{Expression, NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

fn spec_len(label: &str, dims: &[Expression]) -> usize {
    dims.iter()
        .map(|d| {
            d.to_usize()
                .unwrap_or_else(|| panic!("'{label}': model dims are static"))
        })
        .product()
}

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_qwen3moe_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        let Ok(view) = tensors.tensor(&spec.label) else {
            // Not a checkpoint key — a runtime input; the decode loop
            // stages it by handle.
            continue;
        };
        let expected = spec_len(&spec.label, &spec.dims);
        let numel: usize = view.shape().iter().product();
        if numel != expected {
            return Err(format!(
                "'{}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                spec.label,
                view.shape()
            )
            .into());
        }
        let buffer: TypedBuffer = match spec.dtype {
            DType::F32 => tensor_to_f32(&view).into(),
            other => {
                return Err(format!(
                    "'{}': qwen3_moe stages F32 weights only, spec authored as {other:?}",
                    spec.label
                )
                .into());
            }
        };
        pairs.push((spec.id, buffer));
    }
    if pairs.is_empty() {
        return Err("no input-spec label matched a checkpoint key".into());
    }
    Ok(pairs)
}

/// Deterministic pseudo-random parameters for offline runs and smoke
/// tests — anatomy-real, weights fake. With no checkpoint to match
/// labels against, the checkpoint-backed specs are every input MINUS
/// the step's runtime handles (which the decode loop stages); shapes
/// come from `spec.dims`, dtype from `spec.dtype`.
pub fn random_weights(step: &DecodeStep) -> Vec<(NodeIndex, TypedBuffer)> {
    let runtime: std::collections::HashSet<NodeIndex> = [
        step.token.id,
        step.q_pos.id,
        step.rope_cos.id,
        step.rope_sin.id,
        step.rope_rot.id,
        step.gather_idx.id,
        step.scatter_idx.id,
    ]
    .into_iter()
    .chain(step.pool.layers.iter().flat_map(|(k, v)| [k.id, v.id]))
    .collect();

    step.cx
        .logical
        .input_specs()
        .into_iter()
        .filter(|spec| !runtime.contains(&spec.id))
        .enumerate()
        .map(|(seed, spec)| {
            assert_eq!(
                spec.dtype,
                DType::F32,
                "'{}': random fill supports F32 weights only",
                spec.label
            );
            let n = spec_len(&spec.label, &spec.dims);
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                .collect();
            (spec.id, data.into())
        })
        .collect()
}
