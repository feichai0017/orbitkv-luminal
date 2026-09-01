//! M4 Phase 4 bounds-trap probe (`device` feature only), in its OWN
//! test binary: a `__trap()` poisons the CUDA context for the whole
//! process, so this must not share a process with the other device
//! suites (integration test files are separate binaries, which gives
//! the isolation for free).
//!
//! An out-of-range synthetic LAYOUT must TRAP, never silently read: the
//! generated bounds check is load-bearing.
//!
//! RESTATED for the corrected contract (2026-08-31), and the restatement
//! is a HONESTY DOWNGRADE worth reading. The probe used to build a hop
//! chain whose PER-AXIS check caught an escape at the offending hop.
//! There are no hops now — the slot's composed LAYOUT is the read path —
//! and the emitted fence is one final check on the flat index.
//!
//! What that fence can actually catch:
//!  * NON-NEGATIVITY — live for every form, and what this probe fires.
//!  * the strided SPAN — emitted, but VACUOUS for a well-formed strided
//!    layout: `SpanExpr` derives the span FROM the same chain, so a
//!    coordinate inside the domain can never exceed it. It is a
//!    malformed-layout tripwire, not an out-of-range fence.
//!  * for the offset-EXPRESSION forms — nothing but non-negativity, by
//!    construction: an offset function discloses no reach.
//!
//! So a layout that reads PAST the end of its residence with a
//! non-negative index is NOT caught at this layer any more. The
//! preconditions live in the e-graph; the device fence is a backstop
//! that got smaller. Stated, not implied.
#![cfg(feature = "device")]

use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::bufferize::{BufferId, SlotDescriptor};
use luminal::dtype::PlanDtype;
use luminal::layouts::{
    BitWidthTerm, ElementOffsetExpressionLayout, IntExprTerm, MirrorLayout,
    RightMajorContiguousElementLayout, ShapeTerm,
};
use luminal_cuda_lite::{device, kernels, ops, CudaLayout};

fn shape(dims: &[i64]) -> ShapeTerm {
    ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect())
}

fn slot_l(layout: CudaLayout) -> SlotDescriptor<CudaLayout> {
    SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$trap_probe"),
        buffer: BufferId::Allocated(0),
        layout,
    }
}

/// The direct row-major read for `dims` — the flat fast path.
fn slot(dims: Vec<i64>) -> SlotDescriptor<CudaLayout> {
    slot_l(CudaLayout {
        mirror: MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: shape(&dims),
            width: BitWidthTerm(32),
        }),
        dtype: Some(PlanDtype::F32),
    })
}

#[test]
fn negative_layout_index_traps_instead_of_reading() {
    // A rank-1 ELEMENT-OFFSET layout over out [4] whose offset is
    // `c0 - 1`: coordinate 0 addresses element -1, one before the
    // residence. The generated fence for an offset form is exactly
    // `if (a_idx < 0) __trap();`, so the launch must fault rather than
    // read backwards off the allocation.
    let escaping = CudaLayout {
        mirror: MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
            offset: IntExprTerm::Add(
                Box::new(IntExprTerm::Coord { axis_from_end: 0 }),
                Box::new(IntExprTerm::Lit(-1)),
            ),
            shape: shape(&[4]),
            width: BitWidthTerm(32),
        }),
        dtype: Some(PlanDtype::F32),
    };
    let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
    let ctx = kernels::CodegenCtx::from_descriptors(
        op.label(),
        &[slot_l(escaping), slot(vec![4])],
        &[slot(vec![4])],
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
        "a negative layout index must trap loudly, got {:?}",
        result.map(|bytes| bytes.len())
    );
}
