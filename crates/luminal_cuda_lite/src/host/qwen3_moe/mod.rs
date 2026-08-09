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
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr},
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

type Qwen3MoePrepare = unsafe extern "C" fn(dtype: i32) -> i32;
type Qwen3MoeWorkspaceBytes = unsafe extern "C" fn(tokens: i32) -> usize;
type Qwen3MoeEnqueue = unsafe extern "C" fn(
    hidden_states: *const c_void,
    router_logits: *const c_void,
    gate_up_proj: *const c_void,
    down_proj: *const c_void,
    output: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    tokens: i32,
    dtype: i32,
    stream: *mut c_void,
    prefill_mode: i32,
) -> i32;
struct Qwen3MoeLibrary {
    // The library must outlive every copied function pointer obtained from it.
    _library: Library,
    forward: Qwen3MoeForward,
    prepare: Qwen3MoePrepare,
    workspace_bytes: Qwen3MoeWorkspaceBytes,
    enqueue: Qwen3MoeEnqueue,
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
        let prepare = unsafe {
            *library
                .get::<Qwen3MoePrepare>(b"qwen3_moe_prepare\0")
                .map_err(|error| {
                    anyhow::anyhow!(
                        "{} does not export qwen3_moe_prepare: {error}",
                        path.display()
                    )
                })?
        };
        let workspace_bytes = unsafe {
            *library
                .get::<Qwen3MoeWorkspaceBytes>(b"qwen3_moe_workspace_bytes\0")
                .map_err(|error| {
                    anyhow::anyhow!(
                        "{} does not export qwen3_moe_workspace_bytes: {error}",
                        path.display()
                    )
                })?
        };
        let enqueue = unsafe {
            *library
                .get::<Qwen3MoeEnqueue>(b"qwen3_moe_enqueue\0")
                .map_err(|error| {
                    anyhow::anyhow!(
                        "{} does not export qwen3_moe_enqueue: {error}",
                        path.display()
                    )
                })?
        };
        Ok(Self {
            _library: library,
            forward,
            prepare,
            workspace_bytes,
            enqueue,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3MoePointers {
    hidden_states: u64,
    router_logits: u64,
    gate_up_proj: u64,
    down_proj: u64,
    output: u64,
}

impl Qwen3MoePointers {
    pub(crate) fn changed_fields(self, other: Self) -> Vec<&'static str> {
        [
            ("hidden_states", self.hidden_states != other.hidden_states),
            ("router_logits", self.router_logits != other.router_logits),
            ("gate_up_proj", self.gate_up_proj != other.gate_up_proj),
            ("down_proj", self.down_proj != other.down_proj),
            ("output", self.output != other.output),
        ]
        .into_iter()
        .filter_map(|(name, changed)| changed.then_some(name))
        .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3MoeCaptureSignature {
    pub(crate) ptrs: Qwen3MoePointers,
    pub(crate) tokens: i32,
    pub(crate) dtype: i32,
    pub(crate) prefill_mode: i32,
}

pub(crate) struct Qwen3MoeResolved {
    signature: Qwen3MoeCaptureSignature,
    workspace_bytes: usize,
}

impl Qwen3MoeResolved {
    pub(crate) fn signature(&self) -> Qwen3MoeCaptureSignature {
        self.signature
    }
}

pub(crate) struct PreparedQwen3Moe {
    library: Arc<Qwen3MoeLibrary>,
    // Owns the allocation addressed by `workspace_ptr` for every capture and
    // replay.
    _workspace: CudaSlice<u8>,
    workspace_bytes: usize,
    workspace_ptr: u64,
}

impl PreparedQwen3Moe {
    pub(crate) fn enqueue(
        &self,
        stream: &Arc<CudaStream>,
        signature: Qwen3MoeCaptureSignature,
    ) -> anyhow::Result<()> {
        let status = unsafe {
            (self.library.enqueue)(
                signature.ptrs.hidden_states as *const c_void,
                signature.ptrs.router_logits as *const c_void,
                signature.ptrs.gate_up_proj as *const c_void,
                signature.ptrs.down_proj as *const c_void,
                signature.ptrs.output as *mut c_void,
                self.workspace_ptr as *mut c_void,
                self.workspace_bytes,
                signature.tokens,
                signature.dtype,
                stream.cu_stream() as *mut c_void,
                signature.prefill_mode,
            )
        };
        if status != 0 {
            anyhow::bail!("qwen3_moe_enqueue returned status {status}");
        }
        Ok(())
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
            // The generated PT2 rewrite subsumes the semantic HLIR Sum that
            // Qwen3Moe replaces. CUDA lowering subsequently introduces a
            // KernelSum spelling in the same output e-class, including one
            // spelling for a rolled transformer body and one for its peeled
            // boundary layer. Commit only that lowered reduction after all
            // specialized HostOps have been produced; retaining it makes
            // extraction randomly choose 1 or 47 fused MoE blocks.
            Rule::raw(
                r#"
(rule
    (
        (= ?qwen3_moe (Op (Qwen3Moe ?tokens ?storage_dtype) ?qwen_inputs))
        (= ?kernel_sum (Op
            (KernelSum ?shape ?iters ?strides ?iter_stride ?out_strides ?sum_dtype)
            ?sum_inputs))
        (= ?kernel_sum ?qwen3_moe)
    )
    (
        (subsume (Op
            (KernelSum ?shape ?iters ?strides ?iter_stride ?out_strides ?sum_dtype)
            ?sum_inputs))
    )
    :name "Qwen3-MoE commit lowered expert reduction"
    :ruleset kernel_commit
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
    /// Opaque CuTeDSL captures bake the resolved token count into launch
    /// parameters. They can therefore live inside a reusable CUDA graph only
    /// when the token expression is static. Dynamic prefill remains a normal
    /// HostOp; fixed-token decode (and fixed-shape prefill) can be captured.
    pub(crate) fn has_static_graph_signature(&self) -> bool {
        self.tokens.exec(&FxHashMap::default()).is_some()
    }

    fn library(&self) -> anyhow::Result<&Arc<Qwen3MoeLibrary>> {
        if let Some(library) = self.library.get() {
            return Ok(library);
        }
        let loaded = Arc::new(Qwen3MoeLibrary::load()?);
        let _ = self.library.set(loaded);
        Ok(self.library.get().expect("Qwen3-MoE library was just set"))
    }

    pub(crate) fn workspace_bytes(tokens: usize) -> Option<usize> {
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

    pub(crate) fn workspace_bytes_for_resources(
        &self,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Result<usize, ResourceViolation> {
        let tokens = eval_resource_expression(self.tokens, dyn_map, "Qwen3Moe tokens")?;
        Self::workspace_bytes(tokens).ok_or(ResourceViolation::ArithmeticOverflow {
            resource: "Qwen3Moe workspace",
        })
    }

    fn prefill_mode(tokens: i32) -> i32 {
        if tokens > 1024 { 1 } else { 0 }
    }

    pub(crate) fn resolve_for_graph(
        &self,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<Qwen3MoeResolved> {
        if inputs.len() != 4 {
            anyhow::bail!("Qwen3Moe expected 4 inputs, got {}", inputs.len());
        }
        let tokens_usize = self
            .tokens
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("could not evaluate Qwen3Moe token expression"))?;
        let tokens = i32::try_from(tokens_usize)
            .map_err(|_| anyhow::anyhow!("Qwen3Moe token count does not fit i32"))?;
        if tokens <= 0 {
            anyhow::bail!("Qwen3Moe token count must be positive, got {tokens}");
        }
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

        let workspace_bytes = Self::workspace_bytes(tokens_usize)
            .ok_or_else(|| anyhow::anyhow!("Qwen3Moe workspace byte size overflow"))?;
        // Select the same crossover as the Python dispatcher, but make the
        // choice explicit so the captured launch topology cannot change.
        let prefill_mode = Self::prefill_mode(tokens);
        Ok(Qwen3MoeResolved {
            signature: Qwen3MoeCaptureSignature {
                ptrs: Qwen3MoePointers {
                    hidden_states: hidden_states.ptr(),
                    router_logits: router_logits.ptr(),
                    gate_up_proj: gate_up_proj.ptr(),
                    down_proj: down_proj.ptr(),
                    output: output.ptr(),
                },
                tokens,
                dtype,
                prefill_mode,
            },
            workspace_bytes,
        })
    }

    pub(crate) fn prepare_resolved_for_graph(
        &self,
        stream: &Arc<CudaStream>,
        resolved: &Qwen3MoeResolved,
    ) -> anyhow::Result<PreparedQwen3Moe> {
        let library = Arc::clone(self.library()?);
        let status = unsafe { (library.prepare)(resolved.signature.dtype) };
        if status != 0 {
            anyhow::bail!("qwen3_moe_prepare returned status {status}");
        }
        let abi_workspace_bytes = unsafe { (library.workspace_bytes)(resolved.signature.tokens) };
        if abi_workspace_bytes != resolved.workspace_bytes {
            anyhow::bail!(
                "Qwen3Moe workspace ABI mismatch for tokens={}: Rust computed {} bytes but shared object reported {} bytes",
                resolved.signature.tokens,
                resolved.workspace_bytes,
                abi_workspace_bytes,
            );
        }
        let workspace = stream
            .alloc_zeros::<u8>(abi_workspace_bytes)
            .map_err(|error| {
                anyhow::anyhow!("failed to allocate Qwen3Moe graph workspace: {error:?}")
            })?;
        let workspace_ptr = workspace.device_ptr(stream).0 as usize;
        if !workspace_ptr.is_multiple_of(WORKSPACE_ALIGNMENT) {
            anyhow::bail!(
                "Qwen3Moe graph workspace pointer 0x{workspace_ptr:x} is not {WORKSPACE_ALIGNMENT}-byte aligned"
            );
        }
        Ok(PreparedQwen3Moe {
            library,
            _workspace: workspace,
            workspace_bytes: abi_workspace_bytes,
            workspace_ptr: workspace_ptr as u64,
        })
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
        let resolved = self.resolve_for_graph(self_node, inputs, buffers, dyn_map)?;
        let signature = resolved.signature();
        let tokens = signature.tokens;
        let dtype = signature.dtype;
        let ptrs = signature.ptrs;

        let debug_sync = std::env::var_os(DEBUG_SYNC_ENV).is_some();
        if debug_sync {
            eprintln!(
                "QWEN3_MOE_DEBUG begin tokens={tokens} dtype={dtype} stream={:?}",
                stream.cu_stream()
            );
            eprintln!("QWEN3_MOE_DEBUG pointers={ptrs:?}");
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
                ptrs.hidden_states as *const c_void,
                ptrs.router_logits as *const c_void,
                ptrs.gate_up_proj as *const c_void,
                ptrs.down_proj as *const c_void,
                ptrs.output as *mut c_void,
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
        let workspace_bytes = self.workspace_bytes_for_resources(dyn_map)?;

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
    use std::sync::{Arc, OnceLock};

    use super::{
        DType, Expression, PreparedQwen3Moe, Qwen3Moe, Qwen3MoeCaptureSignature, Qwen3MoePointers,
    };
    use crate::{
        cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr},
        kernel::CudaGraphHandle,
    };

    struct CaptureFixture {
        stream: Arc<CudaStream>,
        prepared: PreparedQwen3Moe,
        signature: Qwen3MoeCaptureSignature,
        // Keep every pointer embedded in `signature` alive.
        _buffers: Vec<CudaSlice<u8>>,
    }

    fn capture_fixture_for_tokens(tokens: i32) -> Option<CaptureFixture> {
        if std::env::var_os("LUMINAL_QWEN3_MOE_LIBRARY").is_none() {
            return None;
        }
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.new_stream().ok()?;
        let library = Arc::new(super::Qwen3MoeLibrary::load().ok()?);
        let dtype = 1;
        let status = unsafe { (library.prepare)(dtype) };
        assert_eq!(status, 0, "qwen3_moe_prepare failed");

        let tokens_usize = usize::try_from(tokens).ok()?;
        let sizes = [
            tokens_usize * 2048 * 2,
            tokens_usize * 128 * 2,
            128 * 1536 * 2048 * 2,
            128 * 2048 * 768 * 2,
            tokens_usize * 2048 * 2,
        ];
        let buffers = sizes
            .into_iter()
            .map(|bytes| stream.alloc_zeros::<u8>(bytes).unwrap())
            .collect::<Vec<_>>();
        let ptr = |index: usize| buffers[index].device_ptr(&stream).0;
        let workspace_bytes = Qwen3Moe::workspace_bytes(tokens as usize).unwrap();
        let workspace = stream.alloc_zeros::<u8>(workspace_bytes).unwrap();
        let workspace_ptr = workspace.device_ptr(&stream).0;
        let prepared = PreparedQwen3Moe {
            library,
            _workspace: workspace,
            workspace_bytes,
            workspace_ptr,
        };
        let signature = Qwen3MoeCaptureSignature {
            ptrs: Qwen3MoePointers {
                hidden_states: ptr(0),
                router_logits: ptr(1),
                gate_up_proj: ptr(2),
                down_proj: ptr(3),
                output: ptr(4),
            },
            tokens,
            dtype,
            prefill_mode: Qwen3Moe::prefill_mode(tokens),
        };
        Some(CaptureFixture {
            stream,
            prepared,
            signature,
            _buffers: buffers,
        })
    }

    #[test]
    fn workspace_layout_matches_wrapper_formula() {
        assert_eq!(Qwen3Moe::workspace_bytes(1), Some(2688));
        assert_eq!(Qwen3Moe::workspace_bytes(256), Some(38912));
        assert_eq!(Qwen3Moe::workspace_bytes(1025), Some(150144));
    }

    #[test]
    fn capture_dispatch_uses_the_benchmarked_persistence_crossover() {
        assert_eq!(Qwen3Moe::prefill_mode(1), 0);
        assert_eq!(Qwen3Moe::prefill_mode(1024), 0);
        assert_eq!(Qwen3Moe::prefill_mode(1025), 1);
    }

    #[test]
    fn capture_pointer_changes_report_only_changed_abi_fields() {
        let old = Qwen3MoePointers {
            hidden_states: 1,
            router_logits: 2,
            gate_up_proj: 3,
            down_proj: 4,
            output: 5,
        };
        let new = Qwen3MoePointers {
            hidden_states: 10,
            output: 50,
            ..old
        };
        assert_eq!(old.changed_fields(new), vec!["hidden_states", "output"]);
    }

    #[test]
    fn cuda_graph_capture_requires_a_static_qwen_token_expression() {
        let static_decode = Qwen3Moe {
            tokens: Expression::from(1),
            dtype: DType::Bf16,
            library: OnceLock::new(),
        };
        let dynamic_prefill = Qwen3Moe {
            tokens: Expression::from('a'),
            dtype: DType::Bf16,
            library: OnceLock::new(),
        };

        assert!(static_decode.has_static_graph_signature());
        assert!(!dynamic_prefill.has_static_graph_signature());
    }

    /// Exercises the production capture boundary without PT2, egglog, search,
    /// or arenas: enqueue Driver-API Qwen work directly into an existing
    /// cudarc graph, instantiate it, and replay it.
    #[test]
    #[ignore = "requires CUDA and LUMINAL_QWEN3_MOE_LIBRARY; allocates production expert weights"]
    fn qwen3_moe_direct_capture_to_existing_graph_replays() {
        let tokens = std::env::var("LUMINAL_QWEN3_MOE_CAPTURE_TEST_TOKENS")
            .ok()
            .map(|value| {
                value
                    .parse::<i32>()
                    .expect("capture test tokens must be i32")
            })
            .unwrap_or(16);
        assert!(tokens > 0, "capture test tokens must be positive");
        let Some(fixture) = capture_fixture_for_tokens(tokens) else {
            return;
        };
        let capture_stream = fixture
            .stream
            .context()
            .new_stream()
            .expect("capture stream creation failed");
        capture_stream
            .join(&fixture.stream)
            .expect("capture stream join failed");
        let mut graph = CudaGraphHandle::new(fixture.stream.context().clone())
            .expect("parent graph creation failed");
        let entry = graph.add_empty_node(&[]).expect("entry node failed");
        graph
            .begin_capture_to_graph(&capture_stream, &[entry])
            .expect("cuStreamBeginCaptureToGraph failed");
        let enqueue = fixture.prepared.enqueue(&capture_stream, fixture.signature);
        let end = graph.end_capture(&capture_stream);
        assert!(
            enqueue.is_ok(),
            "direct capture enqueue: {enqueue:?}; end: {end:?}"
        );
        end.expect("cuStreamEndCapture failed");
        let executable = graph
            .instantiate()
            .expect("parent graph instantiate failed");
        executable
            .launch(&fixture.stream)
            .expect("parent graph launch failed");
        fixture
            .stream
            .synchronize()
            .expect("captured Qwen3Moe replay failed asynchronously");
    }

    /// A single captured island is not representative of a transformer: the
    /// full Qwen3 model appends one MoE island per layer to the same mutable
    /// CUDA graph before instantiating it. Exercise that composition directly
    /// so regressions in repeated `cuStreamBeginCaptureToGraph` use fail here,
    /// rather than only after loading the 30B-parameter model.
    #[test]
    #[ignore = "requires CUDA and LUMINAL_QWEN3_MOE_LIBRARY; allocates production expert weights"]
    fn qwen3_moe_repeated_islands_in_one_graph_instantiate_and_replay() {
        let tokens = std::env::var("LUMINAL_QWEN3_MOE_CAPTURE_TEST_TOKENS")
            .ok()
            .map(|value| {
                value
                    .parse::<i32>()
                    .expect("capture test tokens must be i32")
            })
            // The full-model failure occurs while materializing the first
            // decode graph: `a=19` is the cache length, while every MoE island
            // itself still receives one token.
            .unwrap_or(1);
        let islands = std::env::var("LUMINAL_QWEN3_MOE_CAPTURE_TEST_ISLANDS")
            .ok()
            .map(|value| {
                value
                    .parse::<usize>()
                    .expect("capture test island count must be usize")
            })
            .unwrap_or(48);
        assert!(tokens > 0, "capture test tokens must be positive");
        assert!(islands > 0, "capture test island count must be positive");
        let Some(fixture) = capture_fixture_for_tokens(tokens) else {
            return;
        };
        let capture_stream = fixture
            .stream
            .context()
            .new_stream()
            .expect("capture stream creation failed");
        let mut graph = CudaGraphHandle::new(fixture.stream.context().clone())
            .expect("parent graph creation failed");
        let mut tail = graph.add_empty_node(&[]).expect("entry node failed");

        // Mirror production's one independently-owned workspace per fused MoE
        // layer. Tensor/weight pointers may be shared here: graph validity is
        // about capture composition, while the single-island correctness test
        // already checks the launch ABI and asynchronous execution.
        let workspace_bytes = Qwen3Moe::workspace_bytes(tokens as usize).unwrap();
        let mut prepared_islands = Vec::with_capacity(islands);
        for _ in 0..islands {
            let workspace = fixture
                .stream
                .alloc_zeros::<u8>(workspace_bytes)
                .expect("per-island workspace allocation failed");
            let workspace_ptr = workspace.device_ptr(&fixture.stream).0;
            prepared_islands.push(PreparedQwen3Moe {
                library: Arc::clone(&fixture.prepared.library),
                _workspace: workspace,
                workspace_bytes,
                workspace_ptr,
            });
        }

        for (island, prepared) in prepared_islands.iter().enumerate() {
            capture_stream
                .join(&fixture.stream)
                .unwrap_or_else(|error| panic!("island {island} capture stream join: {error:?}"));
            let entry = graph
                .add_empty_node(&[tail])
                .unwrap_or_else(|error| panic!("island {island} entry node: {error:?}"));
            let before = graph
                .nodes()
                .expect("graph node enumeration before capture failed")
                .into_iter()
                .map(|node| node as usize)
                .collect::<std::collections::HashSet<_>>();
            graph
                .begin_capture_to_graph(&capture_stream, &[entry])
                .unwrap_or_else(|error| panic!("island {island} begin capture: {error:?}"));
            let enqueue = prepared.enqueue(&capture_stream, fixture.signature);
            let end = graph.end_capture(&capture_stream);
            assert!(
                enqueue.is_ok(),
                "island {island} enqueue: {enqueue:?}; end: {end:?}"
            );
            end.unwrap_or_else(|error| panic!("island {island} end capture: {error:?}"));

            let captured = graph
                .nodes()
                .expect("graph node enumeration after capture failed")
                .into_iter()
                .filter(|node| !before.contains(&(*node as usize)))
                .collect::<Vec<_>>();
            assert_eq!(
                captured.len(),
                4,
                "island {island} should capture memset + topk + routing + prefill"
            );
            let captured_set = captured
                .iter()
                .map(|node| *node as usize)
                .collect::<std::collections::HashSet<_>>();
            let leaves = captured
                .iter()
                .copied()
                .filter(|node| {
                    graph
                        .dependent_nodes(*node)
                        .expect("captured-node dependent query failed")
                        .iter()
                        .all(|dependent| !captured_set.contains(&(*dependent as usize)))
                })
                .collect::<Vec<_>>();
            assert_eq!(leaves.len(), 1, "island {island} must have one leaf");
            tail = graph
                .add_empty_node(&leaves)
                .unwrap_or_else(|error| panic!("island {island} exit node: {error:?}"));
        }

        let node_count = graph.nodes().expect("final graph node query failed").len();
        let executable = graph.instantiate().unwrap_or_else(|error| {
            panic!(
                "failed to instantiate graph containing {islands} Qwen3Moe islands ({node_count} total nodes, tokens={tokens}): {error:?}"
            )
        });
        executable
            .launch(&fixture.stream)
            .expect("multi-island graph launch failed");
        fixture
            .stream
            .synchronize()
            .expect("multi-island Qwen3Moe replay failed asynchronously");
    }
}
