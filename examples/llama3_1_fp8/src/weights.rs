//! Label-driven staging (ruling 2026-08-13): `cx.logical.input_specs()`
//! enumerates every bound input; specs whose LABELS are checkpoint keys
//! stage by `spec.id`, dtype-dispatched — F8E4M3 weights stage as
//! NATIVE E4M3FN code buffers (the codes ARE the weights — no
//! conversion), F32 numerics convert via `tensor_to_f32`. Inputs whose
//! labels are NOT checkpoint keys (token/q_pos/rope/index args as
//! `arg.*`, caches as `kv_cache.*`) are runtime inputs — the Decoder
//! stages those by handle. Loud bails on shape/dtype mismatches.

use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::float8::F8E4M3;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::{Dtype, SafeTensors};
use std::error::Error;
use std::path::Path;

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_fp8_unfused_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        // Not a checkpoint key => runtime input, staged by handle.
        let Ok(view) = tensors.tensor(&spec.label) else {
            continue;
        };
        let label = &spec.label;
        let expected: usize = spec
            .dims
            .iter()
            .map(|d| d.to_usize().expect("model dims are static"))
            .product();
        let numel: usize = view.shape().iter().product::<usize>().max(1);
        if numel != expected {
            return Err(format!(
                "'{label}': checkpoint shape {:?} has {numel} elements, model expects {expected}",
                view.shape()
            )
            .into());
        }
        let payload: TypedBuffer = match spec.dtype {
            DType::F8E4M3 => {
                if view.dtype() != Dtype::F8_E4M3 {
                    return Err(format!(
                        "'{label}': expected F8_E4M3 in the checkpoint, found {:?}",
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
            DType::F32 => tensor_to_f32(&view).into(),
            other => {
                return Err(format!(
                    "'{label}': no staging rule for input dtype {other:?}"
                )
                .into());
            }
        };
        pairs.push((spec.id, payload));
    }
    Ok(pairs)
}

/// Deterministic fake parameters: fp8 weights get in-range codes,
/// scales get 1.0, numerics get the mini-runner formula. Runtime
/// inputs (`arg.*` args, `kv_cache.*` caches — the labels that are
/// never checkpoint keys) stay Decoder-staged, exactly as in the real
/// path.
pub fn random_weights(cx: &Graph) -> Vec<(NodeIndex, TypedBuffer)> {
    cx.logical
        .input_specs()
        .into_iter()
        .filter(|spec| {
            !spec.label.starts_with("arg.") && !spec.label.starts_with("kv_cache.")
        })
        .enumerate()
        .map(|(seed, spec)| {
            let n: usize = spec
                .dims
                .iter()
                .map(|d| d.to_usize().expect("model dims are static"))
                .product::<usize>()
                .max(1);
            let payload: TypedBuffer = match spec.dtype {
                DType::F8E4M3 => (0..n)
                    .map(|i| {
                        let v = (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6;
                        F8E4M3::from_f32(v)
                    })
                    .collect::<Vec<F8E4M3>>()
                    .into(),
                DType::F32 if spec.label.ends_with("_scale") => vec![1.0f32].into(),
                DType::F32 => (0..n)
                    .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
                    .collect::<Vec<f32>>()
                    .into(),
                other => panic!("'{}': no random-fill rule for dtype {other:?}", spec.label),
            };
            (spec.id, payload)
        })
        .collect()
}
