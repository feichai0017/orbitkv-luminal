//! M4 Phase 4 bounds-trap probe (`device` feature only), in its OWN
//! test binary: a `__trap()` poisons the CUDA context for the whole
//! process, so this must not share a process with the other device
//! suites (integration test files are separate binaries, which gives
//! the isolation for free).
//!
//! An out-of-range synthetic map must TRAP, never silently read: the
//! generated per-axis bounds check is load-bearing.
#![cfg(feature = "device")]

use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::bufferize::{AccessHop, BufferId, ComposedAccess, SlotDescriptor};
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use luminal_cuda_lite::{device, kernels, ops};

fn slot(dims: Vec<i64>, access: Option<ComposedAccess>) -> SlotDescriptor {
    SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$trap_probe"),
        buffer: BufferId::Allocated(0),
        dims: Some(dims),
        element_bits: Some(32),
        dtype: Some(PlanDtype::F32),
        composed_access: access,
    }
}

#[test]
fn out_of_range_map_traps_instead_of_reading() {
    // Entry `c0 + 1` over out [4] into parent [4]: rows 0..2 are in
    // range, row 3 lands at index 4 == extent and must __trap().
    let access = ComposedAccess {
        hops: vec![AccessHop {
            entries: Some(vec![IotaExpr::Add(
                Box::new(IotaExpr::Coord(0)),
                Box::new(IotaExpr::Lit(1)),
            )]),
            parent_dims: Some(vec![4]),
        }],
    };
    let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
    let ctx = kernels::CodegenCtx::from_descriptors(
        op.label(),
        &[slot(vec![4], Some(access)), slot(vec![4], None)],
        &[slot(vec![4], None)],
    )
    .expect("descriptor ctx builds");
    let row = kernels::codegen_for(&op as &dyn BufferTensorIrOp).expect("codegen row");
    let launch = (row.codegen)(&op, &ctx).expect("codegen succeeds").pop().unwrap();
    let parent: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let parent_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(parent.as_ptr() as *const u8, 16) };
    let result = device::launch_single(&launch.source, &[parent_bytes], 16, launch.n);
    assert!(
        result.is_err(),
        "an out-of-range strided read must trap loudly, got {:?}",
        result.map(|bytes| bytes.len())
    );
}
