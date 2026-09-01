//! THE TESTRUNTIME's op set: the op shapes the reference runtime
//! deliberately refuses.
//!
//! The reference runtime is functional, out-of-place and view-free — its
//! kernel table carries only `*FunctionalDps` types, and
//! `reference_allow_list()` is derived from that table, so anything
//! mutating or view-shaped could never be selected there anyway (the
//! assertion at `luminal_reference::runtime`'s
//! `allow_list_matches_the_kernel_registry` states it outright). The ops
//! that exercise those *contracts* live here instead.
//!
//! One op folder per op, the same convention the reference registry
//! follows (op-folder ruling 2026-08-13): instance, DPS form, matcher and
//! `.egg` rewrites all in the op's own directory, so `include_str!` stays
//! path-relative and an op moves as one unit.
//!
//! THE WHOLE OP SET, owned here: the 22 plain [`functional`] spellings
//! forked from the reference registry, the metadata view op, the fused
//! multi-output pair, and the [`mutating`] family of 12 in-place forms.
//! This crate depends on no other runtime.
//!
//! Kernels are not carried: the TestRuntime is plan-level and never
//! executes, so what it forks are the declarations the bufferizer
//! reads — matcher, instance, DPS form, `.egg` rewrites.

pub mod add_mul_fused;
pub mod functional;
pub mod index_map_apply_view;
pub mod mutating;

pub use add_mul_fused::{AddMulFused, AddMulFusedDps, AddMulFusedMatcher};
pub use index_map_apply_view::{IndexMapApplyView, IndexMapApplyViewMatcher};
