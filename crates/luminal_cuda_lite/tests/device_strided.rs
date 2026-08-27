//! M4 Phase 4 device gates (`device` feature only): synthetic-descriptor
//! STRIDED-READ launches, byte-compared against the REFERENCE
//! MATERIALIZE ROUTE on identical inputs. The reference deliberately
//! stays materialize-only — its kernel evaluates the same parsed
//! `IotaExpr` entries per OUT coordinate on the host
//! (`IotaExpr::eval`, see luminal_reference ops/index_map_apply_materialize)
//! — so the oracle here is exactly that evaluation, one materialize per
//! hop, composed outermost-first. Copies move bits, so agreement is
//! byte-exact; the reduce case folds in the same linear order on both
//! sides, so it is byte-exact too.
#![cfg(feature = "device")]

use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::bufferize::{AccessHop, BufferId, ComposedAccess, SlotDescriptor};
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use luminal_cuda_lite::{device, kernels, ops};

fn slot(dims: Vec<i64>, access: Option<ComposedAccess>) -> SlotDescriptor {
    SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$device_synthetic"),
        buffer: BufferId::Allocated(0),
        dims: Some(dims),
        element_bits: Some(32),
        dtype: Some(PlanDtype::F32),
        composed_access: access,
    }
}

fn one_hop(entries: Vec<IotaExpr>, parent_dims: Vec<i64>) -> ComposedAccess {
    ComposedAccess {
        hops: vec![AccessHop { entries: Some(entries), parent_dims: Some(parent_dims) }],
    }
}

fn generate(
    op: &dyn BufferTensorIrOp,
    operand_info: &[SlotDescriptor],
    result_info: &[SlotDescriptor],
) -> kernels::KernelSource {
    let ctx = kernels::CodegenCtx::from_descriptors(op.label(), operand_info, result_info)
        .expect("descriptor ctx builds");
    let row = kernels::codegen_for(op).expect("codegen row");
    let mut launches = (row.codegen)(op, &ctx).expect("codegen succeeds");
    assert_eq!(launches.len(), 1, "single-launch op");
    launches.pop().unwrap()
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn floats_of(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

/// The reference materialize route, hop by hop: evaluate each hop's
/// entries at the current coordinates (bounds-checked, exactly the
/// reference kernel's ensure) to produce the next hop's coordinates;
/// the LAST hop's flat parent offset selects the element.
fn oracle_gather(parent: &[f32], access: &ComposedAccess, out_dims: &[usize]) -> Vec<f32> {
    let numel: usize = out_dims.iter().product();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let mut remainder = flat;
        let mut coords = vec![0usize; out_dims.len()];
        for axis in (0..out_dims.len()).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        let mut parent_dims: Vec<usize> = Vec::new();
        for hop in &access.hops {
            let entries = hop.entries.as_ref().expect("oracle hops are parsed");
            parent_dims = hop
                .parent_dims
                .as_ref()
                .expect("oracle hops are numeric")
                .iter()
                .map(|&d| usize::try_from(d).unwrap())
                .collect();
            let next: Vec<usize> = entries
                .iter()
                .zip(&parent_dims)
                .map(|(entry, &ext)| {
                    let index = entry.eval(&coords);
                    assert!(index >= 0 && (index as usize) < ext, "oracle index in bounds");
                    index as usize
                })
                .collect();
            coords = next;
        }
        let mut strides = vec![1usize; parent_dims.len()];
        for k in (0..parent_dims.len().saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * parent_dims[k + 1];
        }
        let offset: usize = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
        out.push(parent[offset]);
    }
    out
}

fn assert_bytes_equal(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            w.to_ne_bytes(),
            g.to_ne_bytes(),
            "{what}: element {i} — oracle {w} vs device {g}"
        );
    }
}

/// Launch a strided COPY through `access` and byte-compare against the
/// materialize oracle.
fn copy_case(what: &str, parent_dims_i64: Vec<i64>, out_dims: &[usize], access: ComposedAccess) {
    let parent_numel: usize =
        parent_dims_i64.iter().map(|&d| usize::try_from(d).unwrap()).product();
    let parent: Vec<f32> = (0..parent_numel).map(|v| v as f32 * 1.5 + 3.0).collect();
    let out_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
    let launch = generate(
        &op,
        &[slot(out_i64.clone(), Some(access.clone())), slot(out_i64.clone(), None)],
        &[slot(out_i64, None)],
    );
    let out_bytes = out_dims.iter().product::<usize>() * 4;
    let got = device::launch_single(&launch.source, &[bytes_of(&parent)], out_bytes, launch.n)
        .expect("strided launch");
    let want = oracle_gather(&parent, &access, out_dims);
    assert_bytes_equal(&want, &floats_of(&got), what);
}

#[test]
fn transpose_strided_read_matches_the_materialize_oracle() {
    copy_case(
        "transpose",
        vec![2, 3],
        &[3, 2],
        one_hop(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)], vec![2, 3]),
    );
}

#[test]
fn pitched_slice_strided_read_matches_the_materialize_oracle() {
    // Zero-base slice, pitch 8 > cols 5 — plus a flat second operand
    // through the binary add template.
    let access = one_hop(vec![IotaExpr::Coord(1), IotaExpr::Coord(0)], vec![4, 8]);
    let parent: Vec<f32> = (0..32).map(|v| v as f32 - 7.25).collect();
    let b: Vec<f32> = (0..20).map(|v| v as f32 * 0.125).collect();
    let out_dims = [4usize, 5usize];
    let op = ops::add::AddFunctionalDps;
    let launch = generate(
        &op,
        &[
            slot(vec![4, 5], Some(access.clone())),
            slot(vec![4, 5], None),
            slot(vec![4, 5], None),
        ],
        &[slot(vec![4, 5], None)],
    );
    let got = device::launch_single(
        &launch.source,
        &[bytes_of(&parent), bytes_of(&b)],
        20 * 4,
        launch.n,
    )
    .expect("strided add launch");
    let gathered = oracle_gather(&parent, &access, &out_dims);
    let want: Vec<f32> = gathered.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_bytes_equal(&want, &floats_of(&got), "pitched slice + flat add");
}

#[test]
fn broadcast_strided_read_matches_the_materialize_oracle() {
    copy_case(
        "broadcast",
        vec![1, 3],
        &[2, 3],
        one_hop(vec![IotaExpr::Lit(0), IotaExpr::Coord(0)], vec![1, 3]),
    );
}

#[test]
fn two_hop_chain_matches_the_hopwise_materialize_oracle() {
    let access = ComposedAccess {
        hops: vec![
            AccessHop {
                entries: Some(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)]),
                parent_dims: Some(vec![2, 3]),
            },
            AccessHop {
                entries: Some(vec![
                    IotaExpr::Add(Box::new(IotaExpr::Coord(1)), Box::new(IotaExpr::Lit(1))),
                    IotaExpr::Coord(0),
                ]),
                parent_dims: Some(vec![4, 3]),
            },
        ],
    };
    copy_case("two-hop chain", vec![4, 3], &[3, 2], access);
}

#[test]
fn reduce_over_a_strided_read_matches_the_materialize_then_reduce_route() {
    // ReduceSum(axis_from_end=0) over a [2,3] value that is a transpose
    // of parent [3,2]; the oracle materializes first (the reference's
    // only route), then folds in the same linear order.
    let access = one_hop(vec![IotaExpr::Coord(0), IotaExpr::Coord(1)], vec![3, 2]);
    let parent: Vec<f32> = (0..6).map(|v| (v as f32).exp()).collect();
    let op = ops::reduce_sum::ReduceSumDps { axis: 0 };
    let launch = generate(
        &op,
        &[slot(vec![2, 3], Some(access.clone())), slot(vec![2], None)],
        &[slot(vec![2], None)],
    );
    let got = device::launch_single(&launch.source, &[bytes_of(&parent)], 2 * 4, launch.n)
        .expect("strided reduce launch");
    let dense = oracle_gather(&parent, &access, &[2, 3]);
    let want: Vec<f32> = dense
        .chunks_exact(3)
        .map(|row| row.iter().fold(0.0f32, |acc, v| acc + v))
        .collect();
    assert_bytes_equal(&want, &floats_of(&got), "strided reduce_sum");
}
