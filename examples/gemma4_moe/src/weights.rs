//! Host-side weight staging, LABEL-DRIVEN (ruling 2026-08-13): every
//! checkpoint-backed tensor's input LABEL equals its key in the
//! combined bf16 checkpoint (unfused, norms F32), so staging
//! enumerates `cx.logical.input_specs()` and stages every spec whose
//! label the checkpoint knows, converting to f32 on the way in. Labels
//! the checkpoint does NOT know (`arg.*` runtime feeds, `cache.*` pool
//! slabs) are runtime inputs and stay handle-staged in the decoder.
//! Loud bails on a shape or dtype mismatch; no silent skips of a
//! matched key.

use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Element count a spec's declared dims demand (all static here).
fn spec_numel(dims: &[luminal::prelude::Expression]) -> usize {
    dims.iter()
        .map(|d| d.to_usize().expect("model dims are static"))
        .product()
}

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_gemma4moe_text_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        // Not a checkpoint key -> runtime input (arg.*, cache.*):
        // staged by handle in the decoder, never here.
        let Ok(view) = tensors.tensor(&spec.label) else {
            continue;
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
                    "'{}': checkpoint-backed input declares dtype {other:?}; \
                     gemma4_moe stages F32 weights only",
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
/// tests — anatomy-real, weights fake. Label-driven: every input spec
/// that is not a runtime feed (`arg.*`) or a cache slab (`cache.*`)
/// is checkpoint-backed and gets a junk fill sized by its declared
/// dims.
pub fn random_weights(cx: &Graph) -> Vec<(NodeIndex, TypedBuffer)> {
    cx.logical
        .input_specs()
        .into_iter()
        .filter(|spec| !spec.label.starts_with("arg.") && !spec.label.starts_with("cache."))
        .enumerate()
        .map(|(seed, spec)| {
            assert_eq!(
                spec.dtype,
                DType::F32,
                "'{}': junk fill supports F32 weights only",
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
