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
//! convention; `ties` are ordering-only in CL-2). Phase 4 copies every
//! output-role buffer back to host `TypedBuffer`s keyed by BufferLit.

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind};
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

/// Execute a bufferized plan on device 0. Returns host copies of every
/// output-role buffer, keyed by BufferLit.
pub fn execute_plan(
    plan: &BufferIrGraph,
    staged: &FxHashMap<i64, TypedBuffer>,
) -> Result<FxHashMap<i64, TypedBuffer>> {
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
            BufferNode::BufferCopy { src, dst } => {
                let (src_geo, src_dtype) =
                    geometry.get(src).ok_or_else(|| anyhow!("copy src unknown"))?.clone();
                let (dst_geo, dst_dtype) =
                    geometry.get(dst).ok_or_else(|| anyhow!("copy dst unknown"))?.clone();
                if src_geo.iter().product::<usize>() != dst_geo.iter().product::<usize>()
                    || src_dtype != dst_dtype
                {
                    bail!("copy geometry/dtype mismatch: {src_geo:?}/{src_dtype:?} -> {dst_geo:?}/{dst_dtype:?}");
                }
                let src_slice = storage.get(src).unwrap().clone();
                let dst_slice = storage.get_mut(dst).unwrap();
                stream.memcpy_dtod(&src_slice, dst_slice).context("D2D copy")?;
            }
            BufferNode::Compute { op, reads, writes, .. } => {
                let label = op.label();
                if label == "BufferAlloc" || label == "BufferFree" {
                    continue; // storage is pre-materialized in CL-2
                }
                let Some(kernel) = codegen_for(op.as_ref()) else {
                    bail!("no cuda codegen for {label}");
                };
                let ctxinfo = CodegenCtx {
                    operand_dims: reads
                        .iter()
                        .map(|id| geometry.get(id).map(|(d, _)| d.clone()))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| anyhow!("{label} operand lacks geometry"))?,
                    operand_dtypes: reads
                        .iter()
                        .map(|id| geometry.get(id).map(|(_, t)| *t))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| anyhow!("{label} operand lacks dtype"))?,
                    dest_dims: writes
                        .iter()
                        .map(|id| geometry.get(id).map(|(d, _)| d.clone()))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| anyhow!("{label} dest lacks geometry"))?,
                    dest_dtypes: writes
                        .iter()
                        .map(|id| geometry.get(id).map(|(_, t)| *t))
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| anyhow!("{label} dest lacks dtype"))?,
                };
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

    // Phase 4: D2H every output-role buffer, keyed by lit.
    let mut outputs = FxHashMap::default();
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let buffer = plan
                    .buffers
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("output slot names unknown buffer"))?;
                let Some(lit) = buffer.lit else { continue };
                let slice = storage.get(&slot.buffer).unwrap();
                let mut host = vec![0u8; slice.len()];
                stream.memcpy_dtoh(slice, &mut host).context("D2H")?;
                let (_, dtype) = geometry.get(&slot.buffer).unwrap();
                outputs.insert(lit, bytes_to_typed(&host, *dtype)?);
            }
        }
    }
    Ok(outputs)
}
