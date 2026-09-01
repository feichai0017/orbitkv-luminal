//! M4 Phase 4 device gates (`device` feature only), RESTATED for the
//! corrected contract (2026-08-31): synthetic-descriptor STRIDED-READ
//! launches, byte-compared against an INDEPENDENT HOST ORACLE on
//! identical inputs.
//!
//! What changed. These gates used to build a hop chain and check the
//! device against the reference materialize route, which evaluated the
//! same parsed `IotaExpr` entries hop by hop on the host. The hop chain
//! is deleted: an operand now carries ONE composed layout (the e-graph
//! composed it at view creation) and the kernel lowers that layout's own
//! offset expression.
//!
//! Keeping the differential HONEST therefore requires care: if the
//! oracle also evaluated the carried layout, both sides would share the
//! lowering and the test would prove nothing. So each case states its
//! index map TWICE — once as the layout handed to codegen, and once as a
//! hand-written host closure — and the gate is that the two agree
//! element for element. Copies move bits, so agreement is byte-exact;
//! the reduce case folds in the same linear order on both sides, so it
//! is byte-exact too.
#![cfg(feature = "device")]

use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::bufferize::{BufferId, SlotDescriptor};
use luminal::dtype::PlanDtype;
use luminal::layouts::{
    BitWidthTerm, ElementOffsetExpressionLayout, IntExprTerm, MirrorLayout,
    RightMajorContiguousElementLayout, ShapeTerm, StridedElementLayout,
};
use luminal_cuda_lite::{device, kernels, ops, CudaLayout};

fn lit(v: i64) -> IntExprTerm {
    IntExprTerm::Lit(v)
}
fn coord(axis_from_end: i64) -> IntExprTerm {
    IntExprTerm::Coord { axis_from_end }
}
fn mul(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
    IntExprTerm::Mul(Box::new(a), Box::new(b))
}
fn add(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
    IntExprTerm::Add(Box::new(a), Box::new(b))
}
fn shape(dims: &[i64]) -> ShapeTerm {
    ShapeTerm(dims.iter().map(|&d| lit(d)).collect())
}

fn typed(mirror: MirrorLayout) -> CudaLayout {
    CudaLayout { mirror, dtype: Some(PlanDtype::F32) }
}
fn rm_layout(dims: &[i64]) -> CudaLayout {
    typed(MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
        shape: shape(dims),
        width: BitWidthTerm(32),
    }))
}
fn strided_layout(dims: &[i64], chain: Vec<IntExprTerm>) -> CudaLayout {
    typed(MirrorLayout::Strided(StridedElementLayout {
        shape: shape(dims),
        chain,
        width: BitWidthTerm(32),
    }))
}
fn offset_layout(dims: &[i64], offset: IntExprTerm) -> CudaLayout {
    typed(MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
        offset,
        shape: shape(dims),
        width: BitWidthTerm(32),
    }))
}

fn slot_l(layout: CudaLayout) -> SlotDescriptor<CudaLayout> {
    SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$device_synthetic"),
        buffer: BufferId::Allocated(0),
        layout,
    }
}

/// The direct row-major read for `dims` (the flat fast path).
fn slot(dims: Vec<i64>) -> SlotDescriptor<CudaLayout> {
    slot_l(rm_layout(&dims))
}

fn generate(
    op: &dyn BufferTensorIrOp,
    operand_info: &[SlotDescriptor<CudaLayout>],
    result_info: &[SlotDescriptor<CudaLayout>],
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

/// THE INDEPENDENT ORACLE: gather `parent` at every out coordinate using
/// a HAND-WRITTEN flat-index closure — deliberately not the carried
/// layout, so the two sides of the differential are two statements of
/// the same map rather than one statement checked against itself. Out
/// coordinates enumerate row-major over `out_dims`.
fn oracle_gather(
    parent: &[f32],
    out_dims: &[usize],
    index: impl Fn(&[usize]) -> usize,
) -> Vec<f32> {
    let numel: usize = out_dims.iter().product();
    let rank = out_dims.len();
    let mut out = Vec::with_capacity(numel);
    let mut coords = vec![0usize; rank];
    for _ in 0..numel {
        let flat = index(&coords);
        assert!(flat < parent.len(), "oracle index {flat} in bounds");
        out.push(parent[flat]);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < out_dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
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

/// Launch a COPY reading through `layout` and byte-compare against the
/// hand-written oracle.
fn copy_case(
    what: &str,
    parent_numel: usize,
    out_dims: &[usize],
    layout: CudaLayout,
    index: impl Fn(&[usize]) -> usize,
) {
    let parent: Vec<f32> = (0..parent_numel).map(|v| v as f32 * 1.5 + 3.0).collect();
    let out_i64: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
    let launch = generate(
        &op,
        &[slot_l(layout), slot(out_i64.clone())],
        &[slot(out_i64)],
    );
    let out_bytes = out_dims.iter().product::<usize>() * 4;
    let got = device::launch_single(&launch.source, &[bytes_of(&parent)], out_bytes, launch.n)
        .expect("strided launch");
    let want = oracle_gather(&parent, out_dims, index);
    assert_bytes_equal(&want, &floats_of(&got), what);
}

#[test]
fn transpose_strided_read_matches_the_host_oracle() {
    // out [3,2] over parent [2,3]: (i,j) is parent (j,i) = flat j*3 + i.
    copy_case(
        "transpose",
        6,
        &[3, 2],
        strided_layout(&[3, 2], vec![mul(coord(0), lit(3)), coord(1)]),
        |c| c[1] * 3 + c[0],
    );
}

#[test]
fn pitched_slice_strided_read_matches_the_host_oracle() {
    // Zero-base slice, pitch 8 > cols 5 — plus a flat second operand
    // through the binary add template. (i,j) is parent flat i*8 + j.
    let layout = strided_layout(&[4, 5], vec![coord(0), mul(coord(1), lit(8))]);
    let parent: Vec<f32> = (0..32).map(|v| v as f32 - 7.25).collect();
    let b: Vec<f32> = (0..20).map(|v| v as f32 * 0.125).collect();
    let out_dims = [4usize, 5usize];
    let op = ops::add::AddFunctionalDps;
    let launch = generate(
        &op,
        &[slot_l(layout), slot(vec![4, 5]), slot(vec![4, 5])],
        &[slot(vec![4, 5])],
    );
    let got = device::launch_single(
        &launch.source,
        &[bytes_of(&parent), bytes_of(&b)],
        20 * 4,
        launch.n,
    )
    .expect("strided add launch");
    let gathered = oracle_gather(&parent, &out_dims, |c| c[0] * 8 + c[1]);
    let want: Vec<f32> = gathered.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_bytes_equal(&want, &floats_of(&got), "pitched slice + flat add");
}

#[test]
fn broadcast_strided_read_matches_the_host_oracle() {
    // out [2,3] over parent [1,3]: every row reads the same parent row,
    // so (i,j) is parent flat j (the dead axis contributes the zero
    // residue).
    copy_case(
        "broadcast",
        3,
        &[2, 3],
        strided_layout(&[2, 3], vec![coord(0), lit(0)]),
        |c| c[1],
    );
}

#[test]
fn composed_two_view_chain_matches_the_host_oracle() {
    // Was "two-hop chain". The e-graph composes at view creation, so the
    // slot carries ONE layout for the whole composite: out [3,2] over
    // parent [4,3], (a,b) -> parent (b+1, a) = flat (b+1)*3 + a. It
    // renders as an offset-EXPRESSION form, which discloses no reach —
    // hence the device's only trap here is non-negativity, and the ORACLE
    // (bounds-checked) is what actually catches an escape.
    copy_case(
        "composed two-view chain",
        12,
        &[3, 2],
        offset_layout(&[3, 2], add(mul(add(coord(0), lit(1)), lit(3)), coord(1))),
        |c| (c[1] + 1) * 3 + c[0],
    );
}

#[test]
fn reduce_over_a_strided_read_matches_the_gather_then_reduce_oracle() {
    // ReduceSum(axis_from_end=0) over a [2,3] value that is a transpose
    // of parent [3,2]; the oracle gathers first, then folds in the same
    // linear order.
    let layout = strided_layout(&[2, 3], vec![mul(coord(0), lit(2)), coord(1)]);
    let parent: Vec<f32> = (0..6).map(|v| (v as f32).exp()).collect();
    let op = ops::reduce_sum::ReduceSumDps { axis: 0 };
    let launch = generate(&op, &[slot_l(layout), slot(vec![2])], &[slot(vec![2])]);
    let got = device::launch_single(&launch.source, &[bytes_of(&parent)], 2 * 4, launch.n)
        .expect("strided reduce launch");
    // (c0,c1) is parent flat c1*2 + c0.
    let dense = oracle_gather(&parent, &[2, 3], |c| c[1] * 2 + c[0]);
    let want: Vec<f32> = dense
        .chunks_exact(3)
        .map(|row| row.iter().fold(0.0f32, |acc, v| acc + v))
        .collect();
    assert_bytes_equal(&want, &floats_of(&got), "strided reduce_sum");
}
