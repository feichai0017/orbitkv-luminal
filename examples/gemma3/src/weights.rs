//! Host-side weight staging: the combined bf16 checkpoint (unfused,
//! norms F32) converts to f32 on the way in, LABEL-DRIVEN — every
//! checkpoint-backed input's label IS its HF checkpoint key, so staging
//! walks `cx.logical.input_specs()` and matches labels against the
//! checkpoint. Runtime inputs (anonymous `arg.*` — token, q_pos, rope
//! tables, gather/scatter — and the `cache.*` pool) stay handle-staged
//! in the decoder. Loud bails on a missing weight key or a shape
//! mismatch; no silent skips.

use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Gemma3's runtime inputs by label: anonymous args (token, q_pos,
/// rope tables, gather/scatter indices) and the kv-cache pool. These
/// are never checkpoint keys — the decoder stages them by handle.
fn is_runtime_label(label: &str) -> bool {
    label.starts_with("arg.") || label.starts_with("cache.")
}

fn spec_numel(dims: &[luminal::prelude::IntExpr]) -> usize {
    dims.iter()
        .map(|d| d.to_usize().expect("model dims are static"))
        .product()
}

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_gemma3_text_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        let view = match tensors.tensor(&spec.label) {
            Ok(view) => view,
            Err(e) => {
                if is_runtime_label(&spec.label) {
                    continue; // runtime input — staged by handle elsewhere
                }
                return Err(format!("checkpoint is missing '{}': {e}", spec.label).into());
            }
        };
        let expected = spec_numel(&spec.dims);
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
                    "'{}': unsupported authored dtype {other:?} for checkpoint staging",
                    spec.label
                )
                .into());
            }
        };
        pairs.push((spec.id, buffer));
    }
    Ok(pairs)
}

/// Deterministic pseudo-random parameters for offline runs and smoke
/// tests — anatomy-real, weights fake. Fills exactly the checkpoint-
/// backed inputs (everything but the `arg.*`/`cache.*` runtime labels),
/// shaped from the spec's declared dims.
pub fn random_weights(cx: &Graph) -> Vec<(NodeIndex, TypedBuffer)> {
    cx.logical
        .input_specs()
        .into_iter()
        .filter(|spec| !is_runtime_label(&spec.label))
        .enumerate()
        .map(|(seed, spec)| {
            assert_eq!(
                spec.dtype,
                DType::F32,
                "'{}': random fill only supports F32 weights",
                spec.label
            );
            let n = spec_numel(&spec.dims);
            let data: Vec<f32> = (0..n)
                .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                .collect();
            (spec.id, data.into())
        })
        .collect()
}
