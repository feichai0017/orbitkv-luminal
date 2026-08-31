//! CL-2: the device executor — the reference runtime's four execution
//! phases reimplemented over cudarc, consuming the identical
//! `BufferIrGraph`.
//!
//! Phase 1 materializes every plan buffer on the device up front
//! (loud on missing geometry/dtype, exactly like the reference; alloc
//! and free plan ops are no-ops in this first cut). Phase 2 toposorts
//! the dag — `Anti` (WAR) edges ride petgraph, so the order is
//! load-bearing for free. Phase 3 dispatches: D2D for copies,
//! NVRTC-compiled launches for compute (out-of-place: inputs are the
//! operand buffers, the destination is a fresh zeroed slice swapped in
//! after the launch — mirroring the reference's alias-safety
//! convention; `ties` are ordering-only in CL-2). Phase 4 copies each
//! output SLOT's backing buffer back to a host `TypedBuffer`, keyed by
//! slot index and paired with the slot's [`OutputBinding`] — the
//! escape-and-disclose contract (ruling 2026-08-27): the caller gets
//! the backing bytes (possibly parent-sized, for an escaped view
//! election) plus the layout to interpret them under.

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind, OutputBinding};
use luminal::dtype::PlanDtype;
use luminal::prelude::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

use crate::kernels::{codegen_for, CodegenCtx};

fn dtype_bytes(dtype: PlanDtype) -> Result<usize> {
    Ok(match dtype {
        PlanDtype::F32 => 4,
        PlanDtype::Int => 4,
        PlanDtype::Int64 => 8,
        PlanDtype::Bool | PlanDtype::Bool8 => 1,
        other => bail!("cuda-lite CL-2 has no device representation for {other:?}"),
    })
}

fn typed_to_bytes(data: &TypedBuffer) -> &[u8] {
    match data {
        TypedBuffer::F32(v) => bytemuck_cast(v),
        TypedBuffer::I32(v) => bytemuck_cast(v),
        TypedBuffer::I64(v) => bytemuck_cast(v),
        TypedBuffer::Bool8(v) => v.as_slice(),
        TypedBuffer::F8E4M3(_) => unreachable!("dtype_bytes refuses F8 first"),
    }
}

fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    // Plain-old-data reinterpretation for f32/i32/i64 payloads.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_typed(bytes: &[u8], dtype: PlanDtype) -> Result<TypedBuffer> {
    Ok(match dtype {
        PlanDtype::F32 => TypedBuffer::F32(
            bytes.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect(),
        ),
        PlanDtype::Int => TypedBuffer::I32(
            bytes.chunks_exact(4).map(|c| i32::from_ne_bytes(c.try_into().unwrap())).collect(),
        ),
        PlanDtype::Int64 => TypedBuffer::I64(
            bytes.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect(),
        ),
        PlanDtype::Bool | PlanDtype::Bool8 => TypedBuffer::bool8(bytes.to_vec())?,
        other => bail!("cuda-lite CL-2 cannot read back {other:?}"),
    })
}

struct KernelCache {
    ctx: Arc<CudaContext>,
    modules: HashMap<u64, (Arc<CudaModule>, CudaFunction)>,
}

impl KernelCache {
    fn function(&mut self, source: &str) -> Result<CudaFunction> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, func)) = self.modules.get(&key) {
            return Ok(func.clone());
        }
        let ptx = compile_ptx(source)
            .map_err(|e| anyhow!("NVRTC failed: {e:?}\nsource:\n{source}"))?;
        let module = self.ctx.load_module(ptx).context("module load")?;
        let func = module.load_function("k").context("entry `k` missing")?;
        self.modules.insert(key, (module, func.clone()));
        Ok(func)
    }
}

/// Bring-up/test helper: NVRTC-compile `source` (entry `k`), launch it
/// once over `n` threads on device 0 with the given input byte buffers
/// followed by one zeroed `out_bytes` output and the `n` argument (the
/// standard generated-kernel signature, no scratch), and return the
/// output bytes. Used by the Phase-4 synthetic-descriptor device gates
/// to launch strided-read kernels outside a plan; a `__trap()` in the
/// kernel surfaces as an `Err` from the synchronize.
pub fn launch_single(source: &str, inputs: &[&[u8]], out_bytes: usize, n: usize) -> Result<Vec<u8>> {
    let ctx = CudaContext::new(0).context("no CUDA device 0")?;
    let stream = ctx.default_stream();
    let mut cache = KernelCache { ctx: ctx.clone(), modules: HashMap::new() };
    let func = cache.function(source)?;
    let mut device_inputs = Vec::with_capacity(inputs.len());
    for host in inputs {
        let mut slice = stream.alloc_zeros::<u8>(host.len().max(1)).context("input alloc")?;
        if !host.is_empty() {
            stream.memcpy_htod(*host, &mut slice).context("H2D")?;
        }
        device_inputs.push(slice);
    }
    let mut dest = stream.alloc_zeros::<u8>(out_bytes.max(1)).context("dest alloc")?;
    let n_arg = n as u64;
    let cfg = LaunchConfig {
        grid_dim: (((n as u32).max(1) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    for input in &device_inputs {
        builder.arg(input);
    }
    builder.arg(&mut dest);
    builder.arg(&n_arg);
    unsafe { builder.launch(cfg) }.context("launch")?;
    stream.synchronize().context("stream sync")?;
    let mut host = vec![0u8; dest.len()];
    stream.memcpy_dtoh(&dest, &mut host).context("D2H")?;
    Ok(host)
}

/// Execute a bufferized plan on device 0. Returns, per output slot
/// index, a host copy of the slot's BACKING buffer plus its
/// [`OutputBinding`] (the elected layout) — the escape-and-disclose
/// fetch, universal over dense and view elections.
pub fn execute_plan(
    plan: &BufferIrGraph<crate::layouts::CudaLayout>,
    staged: &FxHashMap<i64, TypedBuffer>,
) -> Result<FxHashMap<usize, (TypedBuffer, OutputBinding)>> {
    // ESCAPE GUARD (ruling 2026-08-27): an output slot's backing storage
    // must SURVIVE the call — FreedBy::Caller, whatever the owner.
    // FreedBy::Program backing an output hands the caller bytes the
    // program destroys: minted non-escaping storage (Owner::System) and
    // DONATED boundary storage (Owner::Caller — validate()'s donated arm
    // forbids exactly this plan shape) alike. The pre-lowering
    // certificate enforces this for planner-built plans; hand-built /
    // externally loaded plans never met it — re-check here, loudly,
    // before any bytes move.
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let buffer = plan
                    .buffers
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("output slot {} names unknown buffer", slot.index))?;
                if buffer.freed_by != luminal::layout_ir::FreedBy::Caller {
                    bail!(
                        "output slot {} is backed by NON-ESCAPING buffer {} \
                         (FreedBy::Program, {:?}-owned) — escaped output storage \
                         must be FreedBy::Caller; refusing to hand the caller bytes \
                         the program destroys",
                        slot.index,
                        buffer.label,
                        buffer.owner,
                    );
                }
            }
        }
    }

    let ctx = CudaContext::new(0).context("no CUDA device 0")?;
    let stream = ctx.default_stream();
    let mut cache = KernelCache { ctx: ctx.clone(), modules: HashMap::new() };

    // Phase 1: materialize every buffer on device.
    let mut storage: FxHashMap<BufferId, CudaSlice<u8>> = FxHashMap::default();
    let mut geometry: FxHashMap<BufferId, (Vec<usize>, PlanDtype)> = FxHashMap::default();
    for (id, buffer) in &plan.buffers {
        let dims = buffer
            .dims
            .as_ref()
            .ok_or_else(|| anyhow!("buffer {:?} has no numeric geometry", buffer.label))?;
        let dtype = buffer
            .dtype
            .ok_or_else(|| anyhow!("buffer {:?} has no dtype", buffer.label))?;
        let dims: Vec<usize> = dims.iter().map(|&d| usize::try_from(d).unwrap_or(0)).collect();
        let numel: usize = dims.iter().product();
        let bytes = numel * dtype_bytes(dtype)?;
        let mut slice = stream
            .alloc_zeros::<u8>(bytes.max(1))
            .with_context(|| format!("device alloc {} bytes for {:?}", bytes, buffer.label))?;
        if let Some(lit) = buffer.lit {
            if let Some(data) = staged.get(&lit) {
                let host = typed_to_bytes(data);
                if host.len() != bytes {
                    bail!(
                        "staged buffer {lit} is {} bytes, plan expects {bytes} for {:?}",
                        host.len(),
                        buffer.label
                    );
                }
                stream.memcpy_htod(host, &mut slice).context("H2D")?;
            }
        }
        storage.insert(id.clone(), slice);
        geometry.insert(id.clone(), (dims, dtype));
    }

    // CONTRACT-1 (bind-time): distinct BufferIds must be backed by
    // disjoint device ranges — folded-view reads and WAR ordering are
    // both keyed on BufferId identity. Fresh `alloc_zeros` per buffer
    // makes this hold by construction today; the assert is the
    // contract's enforcement face for when raw caller pointers arrive
    // at this binding surface. Loud refusal, never mistranslation.
    {
        let bound: Vec<crate::binding_check::BoundRange> = storage
            .iter()
            .map(|(id, slice)| {
                let (base, _sync) = slice.device_ptr(&stream);
                crate::binding_check::BoundRange {
                    buffer: format!("{id:?}"),
                    base: base as u64,
                    bytes: slice.len() as u64,
                }
            })
            .collect();
        crate::binding_check::assert_disjoint(&bound)
            .context("CONTRACT-1 bind-time check")?;
    }

    // Phase 2: toposort — Anti edges are ordinary edges here, so WAR
    // ordering is enforced by construction.
    let order = luminal::prelude::petgraph::algo::toposort(&plan.dag, None)
        .map_err(|_| anyhow!("plan dag has a cycle"))?;
    debug_assert!(plan
        .dag
        .edge_weights()
        .all(|e| matches!(e.kind, EdgeKind::Data | EdgeKind::Anti)));

    // Phase 3: dispatch.
    for node in order {
        match &plan.dag[node] {
            BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
            BufferNode::BufferCopy { src, dst, .. } => {
                let (src_geo, src_dtype) =
                    geometry.get(src).ok_or_else(|| anyhow!("copy src unknown"))?.clone();
                let (dst_geo, dst_dtype) =
                    geometry.get(dst).ok_or_else(|| anyhow!("copy dst unknown"))?.clone();
                // RULING 2026-08-27: a BufferCopy is only ever a dumb
                // whole-buffer memcpy — the Phase-5 copy_through_fold path
                // is deleted. This geometry/dtype equality check is the
                // PERMANENT FENCE: a folded delivery smuggled past the
                // bufferizer's refusal would arrive with a parent-shaped
                // src and an output-shaped dst and must fail HERE, loudly,
                // never move bytes.
                if src_geo.iter().product::<usize>() != dst_geo.iter().product::<usize>()
                    || src_dtype != dst_dtype
                {
                    bail!("copy geometry/dtype mismatch: {src_geo:?}/{src_dtype:?} -> {dst_geo:?}/{dst_dtype:?}");
                }
                let src_slice = storage.get(src).unwrap().clone();
                let dst_slice = storage.get_mut(dst).unwrap();
                stream.memcpy_dtod(&src_slice, dst_slice).context("D2D copy")?;
            }
            BufferNode::Compute { op, reads, writes, operand_info, result_info, .. } => {
                let label = op.label();
                if label == "BufferAlloc" || label == "BufferFree" {
                    continue; // storage is pre-materialized in CL-2
                }
                // Train 3: the HOST-CALL arm — cuBLASLt contracts
                // dispatch as one `cublasLtMatmul` library call on the
                // SAME stream as the surrounding kernels, never an
                // NVRTC kernel. The destination follows the executor's
                // out-of-place convention (fresh zeroed slice, swapped
                // into storage after the call), so the C-fold forms
                // read their C operand buffer and write fresh D
                // (C != D pointers, beta = 1.0f — legal, identical
                // layouts by the marker's rule guard).
                if let Some(dps) =
                    op.as_any().downcast_ref::<crate::ops::cublaslt::CublasLtDps>()
                {
                    let call = crate::ops::cublaslt::exec::plan_call(&dps.op)
                        .with_context(|| format!("cuBLASLt call planning for {label}"))?;
                    if writes.len() != 1 {
                        bail!("{label}: single-destination contract, got {}", writes.len());
                    }
                    let input_count = reads.len().saturating_sub(writes.len());
                    let inputs: Vec<CudaSlice<u8>> = reads[..input_count]
                        .iter()
                        .map(|id| storage.get(id).unwrap().clone())
                        .collect();
                    let operand_refs: Vec<&CudaSlice<u8>> = inputs.iter().collect();
                    // F32-only scope end to end (contract 1): every
                    // operand slot and the destination must be F32.
                    let (dest_dims, dest_dtype) = geometry.get(&writes[0]).unwrap().clone();
                    if dest_dtype != PlanDtype::F32
                        || operand_info.iter().any(|s| s.dtype != Some(PlanDtype::F32))
                    {
                        let bad_operands: Vec<String> = operand_info
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| s.dtype != Some(PlanDtype::F32))
                            .map(|(i, s)| format!("operand {i}: {:?}", s.dtype))
                            .collect();
                        bail!(
                            "{label}: cuBLASLt scope is F32-only end to end \
                             (contract 1); dest {dest_dtype:?}, offending \
                             operands: [{}]",
                            bad_operands.join(", ")
                        );
                    }
                    // ROW-CONVENTION FRAME CHECK (the orientation-bug
                    // fence): the fresh dest is written as a dense
                    // ROW-major m x n matrix, and the plan's disclosure
                    // walks the result buffer as row-major over the
                    // RESULT VALUE's dims — the two agree only when the
                    // planned dims ARE [m, n]. A mismatch means the
                    // call frame and the plan frame diverged: refuse
                    // loudly, never land transposed bytes.
                    if dest_dims != [call.m as usize, call.n as usize] {
                        bail!(
                            "{label}: planned destination dims {dest_dims:?} disagree \
                             with the call frame [m, n] = [{}, {}] — the ROW-major D \
                             write would not match the disclosed layout",
                            call.m,
                            call.n
                        );
                    }
                    // The C-fold forms read a REAL C operand through the
                    // same ROW m x n descriptor as D (Cdesc == Ddesc by
                    // rule guard). That read is only correct when the C
                    // operand buffer holds the call-frame C dense
                    // row-major: a slot arriving as a FOLDED VIEW
                    // (composed access over a parent buffer) presents
                    // different bytes and there is no transC to absorb
                    // it — refuse loudly.
                    if let crate::ops::cublaslt::exec::CSource::Operand(ci) = call.c_source {
                        let slot = operand_info.get(ci).ok_or_else(|| {
                            anyhow!("{label}: C operand slot {ci} missing from operand_info")
                        })?;
                        if slot.composed_access.is_some() {
                            bail!(
                                "{label}: C operand arrives as a folded VIEW over its \
                                 buffer — the ROW-major C descriptor (== D) requires a \
                                 materialized dense C operand; refusing before dispatch"
                            );
                        }
                        if slot.dims.as_deref() != Some(&[call.m, call.n][..]) {
                            bail!(
                                "{label}: C operand dims {:?} disagree with the call \
                                 frame [m, n] = [{}, {}] — refusing before dispatch",
                                slot.dims,
                                call.m,
                                call.n
                            );
                        }
                    }
                    let dest_bytes =
                        dest_dims.iter().product::<usize>() * dtype_bytes(dest_dtype)?;
                    let mut dest =
                        stream.alloc_zeros::<u8>(dest_bytes.max(1)).context("dest alloc")?;
                    crate::ops::cublaslt::device_call::dispatch(
                        &call,
                        &operand_refs,
                        &mut dest,
                        &stream,
                    )
                    .with_context(|| format!("cuBLASLt dispatch for {label}"))?;
                    storage.insert(writes[0].clone(), dest);
                    continue;
                }
                let Some(kernel) = codegen_for(op.as_ref()) else {
                    bail!("no cuda codegen for {label}");
                };
                // Phase 3: codegen geometry comes from the node's OWN slot
                // descriptors, never the shared buffer table — `geometry`
                // stays for allocation sizing and the copy check only. A
                // compute node arriving without its descriptors is
                // malformed: bail loudly (mirror of the None-dims bail).
                if operand_info.len() != reads.len() || result_info.len() != writes.len() {
                    bail!(
                        "{label}: compute node lacks slot descriptors \
                         (operand_info {}/{}, result_info {}/{})",
                        operand_info.len(),
                        reads.len(),
                        result_info.len(),
                        writes.len()
                    );
                }
                let ctxinfo = CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                let launches = (kernel.codegen)(op.as_ref(), &ctxinfo)
                    .with_context(|| format!("codegen for {label}"))?;

                // Kernel inputs are the non-destination operands; the
                // destination is a fresh zeroed slice (out-of-place),
                // swapped into storage after the sequence. Launches in
                // one sequence share the stream, so phase ordering
                // (e.g. scatter's init-copy then writes) is free.
                if writes.len() != 1 {
                    bail!("{label}: CL-2 handles single-destination ops, got {}", writes.len());
                }
                let input_count = reads.len().saturating_sub(writes.len());
                let inputs: Vec<CudaSlice<u8>> =
                    reads[..input_count].iter().map(|id| storage.get(id).unwrap().clone()).collect();
                let (dest_dims, dest_dtype) = geometry.get(&writes[0]).unwrap().clone();
                let dest_bytes =
                    dest_dims.iter().product::<usize>() * dtype_bytes(dest_dtype)?;
                let mut dest = stream.alloc_zeros::<u8>(dest_bytes.max(1)).context("dest alloc")?;
                let mut scratch: Option<CudaSlice<u8>> = None;

                for generated in &launches {
                    let func = cache.function(&generated.source)?;
                    if generated.scratch_bytes > 0 && scratch.is_none() {
                        scratch = Some(
                            stream
                                .alloc_zeros::<u8>(generated.scratch_bytes)
                                .context("scratch alloc")?,
                        );
                    }
                    let n = generated.n as u64;
                    let cfg = LaunchConfig {
                        grid_dim: (((generated.n as u32).max(1) + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = stream.launch_builder(&func);
                    for input in &inputs {
                        builder.arg(input);
                    }
                    if generated.scratch_bytes > 0 {
                        builder.arg(scratch.as_mut().unwrap());
                    }
                    builder.arg(&mut dest);
                    builder.arg(&n);
                    unsafe { builder.launch(cfg) }.with_context(|| format!("launch {label}"))?;
                }
                storage.insert(writes[0].clone(), dest);
            }
        }
    }
    stream.synchronize().context("stream sync")?;

    // Phase 4: D2H each output SLOT's backing buffer — the escaped
    // buffer for a view election, the boundary buffer for a dense one —
    // keyed by slot index and paired with the binding's layout. (The
    // declared-but-unused Boundary buffer of an escaped slot never
    // reaches this plan: buffer DCE dropped it, so Phase 1 never
    // allocated it; and no free node exists for an escaping buffer.)
    let mut outputs = FxHashMap::default();
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let slice = storage
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("output slot {} names unknown buffer", slot.index))?;
                let mut host = vec![0u8; slice.len()];
                stream.memcpy_dtoh(slice, &mut host).context("D2H")?;
                let (_, dtype) = geometry.get(&slot.buffer).unwrap();
                outputs.insert(slot.index, (bytes_to_typed(&host, *dtype)?, slot.clone()));
            }
        }
    }
    Ok(outputs)
}
