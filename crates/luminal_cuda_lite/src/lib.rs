//! CUDA-lite on the native ladder.
//!
//! The same six-method ladder as the reference `ReferenceRuntime`
//! (`load → bind_* → search → set_data → execute → get_*`), consuming
//! the same `BufferIrGraph` plans, claiming ops through the same
//! allow-list seam — but executing on a CUDA device with
//! NVRTC-compiled kernels instead of host loops.
//!
//! Stage discipline (M4 kickoff ruling, 2026-08-17: "just focus on
//! getting cuda lite up and running"):
//! - CL-1 (this): plan-level runtime + codegen table, buildable and
//!   testable everywhere; device execution behind the `device`
//!   feature. Zero core-crate edits: the runtime claims a SUBSET of
//!   the reference op inventory via the public allow-list seam, and
//!   candidate profiling stays on the reference host executor (a cost
//!   proxy until the profiler seam is parameterized in CL-3).
//! - CL-2: bring-up on a real device; fidelity vs the reference over
//!   the mini battery.
//! - CL-3: CUDA-native ops (cuBLASLt first) — lands the
//!   matcher-injectable search + profiler trait in core.
//! - CL-4: in-place ties (the Mutating family), views + resident
//!   geometry.
//!
//! Out-of-place by design in CL-1: kernels read operand buffers and
//! write fresh destinations, mirroring the reference executor's
//! alias-safety convention; `ties` and `Anti` edges are honored in the
//! toposort order but no in-place claim is made.

pub mod bindings;
pub mod kernels;
pub mod ops;
pub mod runtime;

#[cfg(feature = "device")]
pub mod device;

pub use bindings::CudaBindings;
pub use runtime::CudaRuntime;

/// The op labels this runtime claims, derived from its codegen table —
/// the CUDA analogue of `reference_allow_list()`: search may only
/// elect ops the backend can actually execute.
pub fn cuda_allow_list() -> Vec<&'static str> {
    kernels::cuda_kernels().iter().map(|k| k.label).collect()
}
