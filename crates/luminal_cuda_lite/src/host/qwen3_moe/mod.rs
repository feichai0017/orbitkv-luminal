//! Fused Qwen3-30B-A3B sparse-MoE host operation.
//!
//! The router projection intentionally stays outside this operation so the
//! ordinary CUDA rewrite/search path can lower it to cuBLASLt. This operation
//! consumes the router logits and owns softmax, top-k routing, and the experts.

use std::{
    ffi::c_void,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use libloading::Library;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    cudarc::driver::CudaStream,
    host::{DeviceBuffer, HostDeviceMemoryPlan, HostOp, ResourceViolation},
    resource::eval_resource_expression,
};

const HIDDEN_SIZE: usize = 2048;
const INTERMEDIATE_SIZE: usize = 768;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 128;
const WORKSPACE_ALIGNMENT: usize = 128;
const DEBUG_SYNC_ENV: &str = "LUMINAL_QWEN3_MOE_DEBUG_SYNC";

/// C ABI implemented by the self-contained CuTeDSL shared library.
///
/// Dtype is 0 for IEEE FP16 and 1 for BF16. Inputs are hidden states, router
/// logits, gate/up expert weights, and down expert weights, in that order.
type Qwen3MoeForward = unsafe extern "C" fn(
    hidden_states: *const c_void,
    router_logits: *const c_void,
    gate_up_proj: *const c_void,
    down_proj: *const c_void,
    output: *mut c_void,
    tokens: i32,
    dtype: i32,
    stream: *mut c_void,
) -> i32;

struct Qwen3MoeLibrary {
    // The library must outlive every copied function pointer obtained from it.
    _library: Library,
    forward: Qwen3MoeForward,
}

impl std::fmt::Debug for Qwen3MoeLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3MoeLibrary").finish_non_exhaustive()
    }
}

// `libloading::Library` is only accessed during construction. The retained
// handle keeps an immutable function pointer alive for concurrent launches.
unsafe impl Send for Qwen3MoeLibrary {}
unsafe impl Sync for Qwen3MoeLibrary {}

impl Qwen3MoeLibrary {
    fn load() -> anyhow::Result<Self> {
        let path = std::env::var_os("LUMINAL_QWEN3_MOE_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libqwen3_moe.so"));

        // SAFETY: this explicit native plugin boundary loads the documented C
        // ABI. The Library is retained for the lifetime of the function pointer.
        let library = unsafe { Library::new(&path) }.map_err(|error| {
            anyhow::anyhow!(
                "failed to load Qwen3-MoE kernel library {}: {error}; set \
                 LUMINAL_QWEN3_MOE_LIBRARY to its absolute path",
                path.display()
            )
        })?;
        let forward = unsafe {
            *library
                .get::<Qwen3MoeForward>(b"qwen3_moe_forward\0")
                .map_err(|error| {
                    anyhow::anyhow!(
                        "{} does not export qwen3_moe_forward: {error}",
                        path.display()
                    )
                })?
        };
        Ok(Self {
            _library: library,
            forward,
        })
    }
}

/// Fused post-router Qwen3-MoE operation for Qwen3-30B-A3B dimensions.
#[derive(Debug)]
pub struct Qwen3Moe {
    tokens: Expression,
    dtype: DType,
    library: OnceLock<Arc<Qwen3MoeLibrary>>,
}

impl Default for Qwen3Moe {
    fn default() -> Self {
        Self {
            tokens: Expression::default(),
            dtype: DType::Bf16,
            library: OnceLock::new(),
        }
    }
}

impl Clone for Qwen3Moe {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens,
            dtype: self.dtype,
            library: OnceLock::new(),
        }
    }
}

impl EgglogOp for Qwen3Moe {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "Qwen3Moe",
            &[("tokens", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        4
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![include_str!("qwen3_moe_declarations.egg").to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Debugging the original PT2 lowering requires constructing the same
        // CUDA search space without introducing this HostOp alternative.
        if std::env::var_os("LUMINAL_DISABLE_QWEN3_MOE").is_some() {
            return Vec::new();
        }
        vec![
            // `dtype(IR)` is a separate Egglog table from the dtype field in
            // `Qwen3Moe`. The fused node is unioned with an F32-accumulating
            // reference reduction, so a one-shot dtype assignment in the
            // fusion action is not enough: a later saturated `dtype_prop`
            // phase can otherwise restore F32 on the shared e-class. Derive
            // the storage dtype from the HostOp kind on every dtype fixed
            // point, just as Cast and other dtype-carrying OpKinds do.
            Rule::raw(
                r#"
(rule
    (
        (= ?qwen3_moe (Op (Qwen3Moe ?tokens ?storage_dtype) ?inputs))
    )
    (
        (set (dtype ?qwen3_moe) ?storage_dtype)
    )
    :name "Qwen3-MoE dtype from ABI field"
    :ruleset dtype_prop
)
"#,
            ),
            // PT2 materializes a contiguous standalone block output as
            // `Qwen3Moe + tensor(0)`. The Add is numerically a view/copy, but
            // its dtype row may have been populated from the original F32
            // reduction before that reduction was replaced. Fold this exact
            // identity after fusion so the HostOp writes directly into the
            // model-width output buffer. Full-model residual Adds are not
            // affected because their second input is not a zero constant.
            Rule::raw(
                r#"
(rule
    (
        (= ?qwen3_moe (Op (Qwen3Moe ?tokens ?storage_dtype) ?inputs))
        (= ?zero (Op (Constant 0.0) (INil)))
        (= ?output_view
            (Op (Add ?shape ?qwen_strides ?zero_strides ?output_strides)
                (ICons ?qwen3_moe (ICons ?zero (INil)))))
    )
    (
        (union ?output_view ?qwen3_moe)
        (delete (dtype ?output_view))
        (set (dtype ?output_view) ?storage_dtype)
        (subsume
            (Op (Add ?shape ?qwen_strides ?zero_strides ?output_strides)
                (ICons ?qwen3_moe (ICons ?zero (INil)))))
    )
    :name "Qwen3-MoE remove PT2 zero-add output view"
    :ruleset glumoe
)
"#,
            ),
            Rule::raw(include_str!("qwen3_moe_rewrite.egg")),
        ]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let extracted = Self {
            tokens: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
            dtype: extract_dtype(egraph, kind_children[1]),
            library: OnceLock::new(),
        };
        (
            LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>),
            input_enodes,
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl Qwen3Moe {
    fn library(&self) -> anyhow::Result<&Arc<Qwen3MoeLibrary>> {
        if let Some(library) = self.library.get() {
            return Ok(library);
        }
        let loaded = Arc::new(Qwen3MoeLibrary::load()?);
        let _ = self.library.set(loaded);
        Ok(self.library.get().expect("Qwen3-MoE library was just set"))
    }

    fn workspace_bytes(tokens: usize) -> Option<usize> {
        let assignments = tokens.checked_mul(TOP_K)?;
        let mut cursor = 0usize;
        let mut reserve = |bytes: usize| -> Option<()> {
            cursor = cursor.checked_add(WORKSPACE_ALIGNMENT - 1)? & !(WORKSPACE_ALIGNMENT - 1);
            cursor = cursor.checked_add(bytes)?;
            Some(())
        };

        reserve(assignments.checked_mul(2)?)?; // router scores
        reserve(assignments.checked_mul(4)?)?; // router indices
        reserve((NUM_EXPERTS + 1).checked_mul(4)?)?; // expert offsets
        reserve(assignments.checked_mul(4)?)?; // m_a_idx
        reserve(assignments.checked_mul(4)?)?; // routing weights
        reserve(4)?; // max rows per expert
        reserve(assignments.checked_mul(4)?)?; // expert tile map
        reserve(4)?; // total work tiles
        reserve(4)?; // tile counter
        reserve(NUM_EXPERTS.checked_mul(4)?)?; // expert counts
        reserve(NUM_EXPERTS.checked_mul(4)?)?; // routing write pointers

        cursor
            .checked_add(WORKSPACE_ALIGNMENT - 1)
            .map(|bytes| bytes & !(WORKSPACE_ALIGNMENT - 1))
    }
}

impl HostOp for Qwen3Moe {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() != 4 {
            anyhow::bail!("Qwen3Moe expected 4 inputs, got {}", inputs.len());
        }
        let tokens = self
            .tokens
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("could not evaluate Qwen3Moe token expression"))?;
        let tokens = i32::try_from(tokens)
            .map_err(|_| anyhow::anyhow!("Qwen3Moe token count does not fit i32"))?;
        let dtype = match self.dtype {
            DType::F16 => 0,
            DType::Bf16 => 1,
            other => anyhow::bail!("Qwen3Moe supports F16 and BF16, got {other:?}"),
        };
        let buffer = |node: NodeIndex| {
            buffers
                .get(&node)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Qwen3Moe is missing buffer for node {node:?}"))
        };
        let hidden_states = buffer(inputs[0])?;
        let router_logits = buffer(inputs[1])?;
        let gate_up_proj = buffer(inputs[2])?;
        let down_proj = buffer(inputs[3])?;
        let output = buffer(self_node)?;

        let tokens_usize = usize::try_from(tokens)
            .map_err(|_| anyhow::anyhow!("Qwen3Moe token count must be non-negative"))?;
        let tensor_bytes = |elements: usize, label: &'static str| {
            elements
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Qwen3Moe {label} byte size overflow"))
        };
        let dynamic_elements = |width: usize, label: &'static str| {
            tokens_usize
                .checked_mul(width)
                .ok_or_else(|| anyhow::anyhow!("Qwen3Moe {label} element count overflow"))
        };
        let required = [
            (
                "hidden_states",
                hidden_states,
                tensor_bytes(
                    dynamic_elements(HIDDEN_SIZE, "hidden_states")?,
                    "hidden_states",
                )?,
            ),
            (
                "router_logits",
                router_logits,
                tensor_bytes(
                    dynamic_elements(NUM_EXPERTS, "router_logits")?,
                    "router_logits",
                )?,
            ),
            (
                "gate_up_proj",
                gate_up_proj,
                tensor_bytes(
                    NUM_EXPERTS
                        .checked_mul(2 * INTERMEDIATE_SIZE)
                        .and_then(|elements| elements.checked_mul(HIDDEN_SIZE))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Qwen3Moe gate_up_proj element count overflow")
                        })?,
                    "gate_up_proj",
                )?,
            ),
            (
                "down_proj",
                down_proj,
                tensor_bytes(
                    NUM_EXPERTS
                        .checked_mul(HIDDEN_SIZE)
                        .and_then(|elements| elements.checked_mul(INTERMEDIATE_SIZE))
                        .ok_or_else(|| {
                            anyhow::anyhow!("Qwen3Moe down_proj element count overflow")
                        })?,
                    "down_proj",
                )?,
            ),
            (
                "output",
                output,
                tensor_bytes(dynamic_elements(HIDDEN_SIZE, "output")?, "output")?,
            ),
        ];
        for (label, buffer, required_bytes) in required {
            if buffer.len() < required_bytes {
                anyhow::bail!(
                    "Qwen3Moe {label} buffer is too small: node pointer=0x{:x}, available={} bytes, required={} bytes for tokens={tokens}",
                    buffer.ptr(),
                    buffer.len(),
                    required_bytes,
                );
            }
        }

        let debug_sync = std::env::var_os(DEBUG_SYNC_ENV).is_some();
        if debug_sync {
            eprintln!(
                "QWEN3_MOE_DEBUG begin tokens={tokens} dtype={dtype} stream={:?}",
                stream.cu_stream()
            );
            for (label, buffer, required_bytes) in required {
                eprintln!(
                    "QWEN3_MOE_DEBUG buffer={label} ptr=0x{:x} end=0x{:x} available={} required={required_bytes}",
                    buffer.ptr(),
                    buffer.ptr().saturating_add(buffer.len() as u64),
                    buffer.len(),
                );
            }
            stream.synchronize().map_err(|error| {
                anyhow::anyhow!(
                    "Qwen3Moe pre-launch synchronization failed at tokens={tokens}: {error:?}"
                )
            })?;
            eprintln!("QWEN3_MOE_DEBUG pre-launch sync passed tokens={tokens}");
        }

        let library = self.library()?;
        let status = unsafe {
            (library.forward)(
                hidden_states.ptr() as *const c_void,
                router_logits.ptr() as *const c_void,
                gate_up_proj.ptr() as *const c_void,
                down_proj.ptr() as *const c_void,
                output.ptr() as *mut c_void,
                tokens,
                dtype,
                stream.cu_stream() as *mut c_void,
            )
        };
        if status != 0 {
            anyhow::bail!("qwen3_moe_forward returned status {status}");
        }

        if debug_sync {
            stream.synchronize().map_err(|error| {
                anyhow::anyhow!(
                    "Qwen3Moe post-launch synchronization failed at tokens={tokens}: {error:?}"
                )
            })?;
            eprintln!("QWEN3_MOE_DEBUG post-launch sync passed tokens={tokens}");
        }

        // All C-ABI launches use Luminal's stream. The Python raw-pointer
        // boundary orders that stream after PyTorch's producer stream, and
        // normal stream ordering carries this output into downstream graphs.
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.tokens * HIDDEN_SIZE
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 2
    }

    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffer_lengths: &FxHashMap<NodeIndex, usize>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        let tokens = eval_resource_expression(self.tokens, dyn_map, "Qwen3Moe tokens")?;
        let workspace_bytes =
            Self::workspace_bytes(tokens).ok_or(ResourceViolation::ArithmeticOverflow {
                resource: "Qwen3Moe workspace",
            })?;

        // The shared object caches this allocation by host thread and stream.
        // Model it as the peak host-op workspace rather than a keyed shared
        // allocation because different dynamic-shape buckets require different
        // capacities; keyed allocations require an identical byte count.
        Ok(HostDeviceMemoryPlan {
            transient_peak_bytes: workspace_bytes,
            ..Default::default()
        })
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("Qwen3Moe")
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Moe;

    #[test]
    fn workspace_layout_matches_wrapper_formula() {
        assert_eq!(Qwen3Moe::workspace_bytes(1), Some(2688));
        assert_eq!(Qwen3Moe::workspace_bytes(256), Some(38912));
        assert_eq!(Qwen3Moe::workspace_bytes(1025), Some(150144));
    }
}
