//! Host-side weight staging, LABEL-DRIVEN (ruling 2026-08-13): every
//! checkpoint-backed input's LABEL is its HF checkpoint key, so staging
//! walks `cx.logical.input_specs()` and matches labels against the
//! checkpoint file — no hand-written key → handle table. Inputs whose
//! labels are NOT checkpoint keys (token, q_pos, rope tables, cache
//! pairs — the anonymous `arg.{k}` inputs here) are runtime inputs and
//! stay handle-staged in `lib.rs`. The combined bf16 checkpoint
//! converts to f32 on the way in (the parameters ARE f32-typed in this
//! model); loud bails on a shape mismatch or an unexpected dtype — no
//! silent skips of a matched key.

use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Read `model_combined_bf16_v1.safetensors` (the `hf.rs` artifact) and
/// produce the (tensor id, f32 data) pairs for every input spec whose
/// label is a checkpoint key.
pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_bf16_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        let Ok(view) = tensors.tensor(&spec.label) else {
            // Runtime inputs (token, q_pos, rope tables, gather/scatter
            // indices, caches) are the anonymous `arg.{k}` ports here —
            // staged by handle in `lib.rs`. Any OTHER label missing from
            // the checkpoint is a real absent weight: bail loudly, never
            // silently reclassify (house doctrine; matches gemma3/llama3).
            if spec.label.starts_with("arg.") || spec.label.starts_with("cache.") {
                continue;
            }
            return Err(format!(
                "'{}': named model input missing from the checkpoint",
                spec.label
            )
            .into());
        };
        let expected: usize = spec
            .dims
            .iter()
            .map(|d| d.to_usize().expect("model dims are static"))
            .product();
        let numel: usize = view.shape().iter().product();
        if numel != expected {
            return Err(format!(
                "'{}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                spec.label,
                view.shape()
            )
            .into());
        }
        match spec.dtype {
            DType::F32 => pairs.push((spec.id, tensor_to_f32(&view).into())),
            other => {
                return Err(format!(
                    "'{}': this model's parameters stage as F32, but the spec says {other:?}",
                    spec.label
                )
                .into());
            }
        }
    }
    Ok(pairs)
}

/// Deterministic pseudo-random parameters (the mini-runner formula) for
/// offline runs and the smoke tests — anatomy-real, weights fake. The
/// checkpoint-backed set is every NAMED input; the anonymous `arg.{k}`
/// labels are the runtime inputs and are skipped.
pub fn random_weights(cx: &Graph) -> Vec<(NodeIndex, TypedBuffer)> {
    cx.logical
        .input_specs()
        .into_iter()
        .filter(|spec| !spec.label.starts_with("arg."))
        .enumerate()
        .map(|(seed, spec)| {
            assert_eq!(
                spec.dtype,
                DType::F32,
                "'{}': this model's parameters are F32",
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
