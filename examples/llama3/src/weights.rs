//! Host-side weight staging, LABEL-DRIVEN (ruling 2026-08-13): since
//! the Ns landing every checkpoint-backed tensor's label IS its HF
//! checkpoint key, so staging enumerates the graph's own input specs
//! and matches labels against the combined bf16 checkpoint (unfused,
//! norms F32), converting to f32 on the way in. Runtime inputs —
//! anonymous `arg.*` (token, q_pos, rope tables, gather/scatter,
//! masks) and the `cache.*` pool — are staged by handle in the decode
//! drivers and are skipped here. Loud bails on a weight-looking label
//! missing from the checkpoint or a shape mismatch; no silent skips.

use crate::hf::tensor_to_f32;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{NodeIndex, TypedBuffer};
use safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Runtime inputs — everything the decode drivers stage by handle each
/// search/step rather than from the checkpoint.
fn is_runtime_input(label: &str) -> bool {
    label.starts_with("arg.") || label.starts_with("cache.")
}

pub fn load_safetensors_weights(
    cx: &Graph,
    model_dir: &Path,
) -> Result<Vec<(NodeIndex, TypedBuffer)>, Box<dyn Error>> {
    let path = model_dir.join("model_combined_bf16_unfused_v1.safetensors");
    let file = std::fs::File::open(&path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    let mut pairs = Vec::new();
    for spec in cx.logical.input_specs() {
        let label = &spec.label;
        let view = match tensors.tensor(label) {
            Ok(view) => view,
            Err(e) if is_runtime_input(label) => {
                let _ = e; // runtime input — staged by handle in the driver
                continue;
            }
            Err(e) => return Err(format!("checkpoint is missing '{label}': {e}").into()),
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
            DType::F32 => pairs.push((spec.id, tensor_to_f32(&view).into())),
            other => {
                return Err(format!(
                    "'{label}': authored dtype {other:?} — this example stages F32 weights only"
                )
                .into());
            }
        }
    }
    Ok(pairs)
}

/// Deterministic pseudo-random parameters for offline runs and smoke
/// tests — anatomy-real, weights fake. Seeded by the weight's position
/// among the graph's checkpoint-backed inputs, which is identical
/// across the position-slots and page-table graphs (both record the
/// model first), so cross-driver proofs see identical parameters.
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
                "'{}': random fill stages F32 weights only",
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
