//! TRAIN-2B: composed-access READ lowering for the three kernel
//! families that refused it in the field (A100, 2026-08-28): Gather,
//! ScatterFunctional, IndexMapApplyMaterialize. Searches were green and
//! plans built — only the CL device codegen refused folded read
//! operands ("operand N carries a composed access this kernel does not
//! lower"). Everything here is host-side: real searched plans through
//! the CUDA ladder, rendered to CUDA source strings (the
//! codegen_identity discipline), pinned for composed index arithmetic +
//! per-axis __trap() bounds + hop-composition order (outermost-first,
//! the walk_layout_index / composed_read_index convention). Numeric
//! truth for the gather family comes from the reference runtime on the
//! SAME graph (materialize-only by ruling aff22598 — flat kernels),
//! pinned against hand-computed values. WRITE sides stay fail-closed:
//! `codegen_identity::strided::expression_kernel_write_sides_stay_fail_closed`.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::{kernels, CudaRuntime};

/// The view fixtures' search budget (mirrors `view_admission`):
/// profiling is static bytes-moved, so folds win deterministically
/// under a fixed seed. The seed is per-fixture: fold-vs-materialize
/// spellings of a movement class are cost-TIED (same bytes), so which
/// one the genome elects is sampling — a fixture that needs a specific
/// tied spelling pins the seed that elects it.
fn view_search_options(seed: u64) -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed,
    }
}

/// Load → search on the CUDA runtime; return the best plan.
fn plan_for(cx: &Graph, inputs: &[(NodeIndex, TypedBuffer)], seed: u64) -> BufferIrGraph {
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let data: FxHashMap<NodeIndex, TypedBuffer> = inputs.iter().cloned().collect();
    let outcome = rt.search(&data, &view_search_options(seed)).expect("cuda search");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

/// Render every compute node through the REAL dispatch path
/// (descriptor ctx → codegen row). Returns
/// (label, launch sources, composed operand slots).
fn rendered(plan: &BufferIrGraph) -> Vec<(String, Vec<String>, Vec<usize>)> {
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute { op, operand_info, result_info, .. } = node else {
            continue;
        };
        let label = op.label().to_string();
        if label == "BufferAlloc" || label == "BufferFree" {
            continue;
        }
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx::from_descriptors(&label, operand_info, result_info)
            .unwrap_or_else(|e| panic!("descriptor ctx for {label}: {e}"));
        let folded: Vec<usize> = operand_info
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.composed_access.as_ref().map(|_| k))
            .collect();
        let sources: Vec<String> = (kernel.codegen)(op.as_ref(), &ctx)
            .unwrap_or_else(|e| panic!("codegen for {label}: {e}"))
            .into_iter()
            .map(|l| l.source)
            .collect();
        out.push((label, sources, folded));
    }
    out
}

/// The single node with `label` in the plan, rendered.
fn the_one(plan: &BufferIrGraph, label: &str) -> (Vec<String>, Vec<usize>) {
    let hits: Vec<_> = rendered(plan).into_iter().filter(|(l, _, _)| l == label).collect();
    assert_eq!(hits.len(), 1, "exactly one {label} in the plan:\n{}", plan.summary());
    let (_, sources, folded) = hits.into_iter().next().unwrap();
    (sources, folded)
}

fn assert_contains(source: &str, needles: &[&str], what: &str) {
    for needle in needles {
        assert!(
            source.contains(needle),
            "{what}: generated source missing `{needle}`:\n{source}"
        );
    }
}

/// Numeric truth from the reference runtime (flat kernels /
/// materialization — it never folds), checked against hand-computed
/// values. The CL side of the differential is textual on CPU; the
/// device half is the A100 pass.
fn reference_values(
    cx: &Graph,
    inputs: &[(NodeIndex, TypedBuffer)],
    out: NodeIndex,
    want: &[f32],
) {
    let reference = luminal_reference::harness::run_reference(cx, inputs);
    let got = reference.get_f32(out).expect("reference output");
    assert_eq!(got.as_slice(), want, "reference numerics diverge from the hand computation");
}

/// EMBEDDING-STYLE GATHER, indices through a fold: data (4,3) gathered
/// at rows broadcast over the out shape — coord0 = rows(2,) expanded to
/// (2,3) (a stride-0 view the search folds), coord1 = a column iota.
/// This is the llama3/qwen3/gemma field shape in miniature; before
/// Train-2B this plan's codegen refused with "Gather: operand 1 carries
/// a composed access this kernel does not lower".
#[test]
fn gather_lowers_a_folded_coordinate_operand() {
    let mut cx = Graph::new();
    let data = cx.tensor((4usize, 3usize));
    let rows = cx.tensor_dtyped(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let row_coord = rows.expand_dim(1, 3usize);
    let out = data.gather(&[row_coord, cols]).output();

    let data_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(data.id, data_vals.into()), (rows.id, vec![2i32, 0].into())];

    // Numeric truth: out[i][j] = data[rows[i]][j] with rows = [2, 0].
    reference_values(&cx, &inputs, out.id, &[6., 7., 8., 0., 1., 2.]);

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "GatherGeneric");
    assert_eq!(folded, vec![1], "the broadcast folds into coord0 (operand 1):\n{}", plan.summary());
    assert_eq!(sources.len(), 1, "gather is a single launch");
    // The full rendered kernel, pinned: coord0 is read through its
    // chain (hop 0 = the broadcast map, parent (2,), entry c0), with
    // the per-axis chain trap AND the gather's own value-extent traps;
    // coord1 stays the flat read; the data read is untouched.
    assert_eq!(
        sources[0],
        r#"extern "C" __global__ void k(const float* data, const int* coord0, const int* coord1, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long rem = i;
    long long c1 = (long long)(rem % 3ULL); rem /= 3ULL;
    long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;
    long long flat = 0;
    long long coord;
    long long coord0_h0_0 = c0;
    if (coord0_h0_0 < 0 || coord0_h0_0 >= 2LL) __trap();
    long long coord0_idx = coord0_h0_0 * 1LL;
    coord = (long long)coord0[coord0_idx];
    if (coord < 0 || coord >= 4LL) __trap();
    flat += coord * 3LL;
    coord = (long long)coord1[i];
    if (coord < 0 || coord >= 3LL) __trap();
    flat += coord * 1LL;
    out[i] = data[flat];
}"#
    );
}

/// GATHER, data through a fold: the data operand is a permute view, so
/// its chain is evaluated at the GATHERED coordinate values — the
/// gather's own indirection composes ON TOP of the folded chain
/// (`data_c* = coord`, then the hops).
#[test]
fn gather_lowers_a_folded_data_operand() {
    let mut cx = Graph::new();
    let base = cx.tensor((3usize, 4usize));
    let rows = cx.tensor_dtyped(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    // data = base^T, shape (4,3): data[i][j] = base[j][i].
    let data = base.permute((1, 0));
    let out = data.gather(&[rows.expand_dim(1, 3usize), cols]).output();

    let base_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(base.id, base_vals.into()), (rows.id, vec![2i32, 0].into())];

    // data[i][j] = base[j][i] = j*4 + i; rows = [2, 0]:
    // out row 0 = data[2][:] = [2, 6, 10]; out row 1 = data[0][:] = [0, 4, 8].
    reference_values(&cx, &inputs, out.id, &[2., 6., 10., 0., 4., 8.]);

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "GatherGeneric");
    assert!(
        folded.contains(&0),
        "the permute folds onto the data operand:\n{}",
        plan.summary()
    );
    assert_contains(
        &sources[0],
        &[
            // The gathered coordinates are trapped against the data
            // VALUE's extents (4,3), then bound as the chain's inputs.
            "if (coord < 0 || coord >= 4LL) __trap();",
            "long long data_c0 = coord;",
            "if (coord < 0 || coord >= 3LL) __trap();",
            "long long data_c1 = coord;",
            // Hop 0 = the permute map into base (3,4), evaluated at
            // the data-value coordinates, per-axis trapped.
            "long long data_h0_0 = data_c1;",
            "if (data_h0_0 < 0 || data_h0_0 >= 3LL) __trap();",
            "long long data_h0_1 = data_c0;",
            "if (data_h0_1 < 0 || data_h0_1 >= 4LL) __trap();",
            // Residence read over base's row-major strides (4,1).
            "long long data_idx = data_h0_0 * 4LL + data_h0_1 * 1LL;",
            "out[i] = data[data_idx];",
        ],
        "gather folded-data",
    );
    assert!(!sources[0].contains("data[flat]"), "no flat data read remains:\n{}", sources[0]);
}

/// SCATTER, coordinate operand through a fold: init (4,3), src (2,3),
/// coord0 = rows(2,) broadcast to (2,3), coord1 = a column iota — the
/// qwen3_moe field shape ("ScatterFunctional: operand 2 carries a
/// composed access") in miniature. The checked-scatter contract is
/// untouched: same flags scratch, same atomicExch trap, same write
/// address arithmetic.
#[test]
fn scatter_lowers_a_folded_coordinate_operand() {
    let mut cx = Graph::new();
    let init = cx.tensor((4usize, 3usize));
    let src = cx.tensor((2usize, 3usize));
    let rows = cx.tensor_dtyped(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let row_coord = rows.expand_dim(1, 3usize);
    let out = init.scatter(&[row_coord, cols], src).output();

    let inputs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (init.id, vec![0.0f32; 12].into()),
        (src.id, (0..6).map(|v| 10.0 + v as f32).collect::<Vec<f32>>().into()),
        (rows.id, vec![2i32, 0].into()),
    ];

    // out = zeros(4,3); out[2][:] = src[0][:] = [10,11,12];
    // out[0][:] = src[1][:] = [13,14,15].
    reference_values(
        &cx,
        &inputs,
        out.id,
        &[13., 14., 15., 0., 0., 0., 10., 11., 12., 0., 0., 0.],
    );

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "ScatterFunctionalGeneric");
    assert_eq!(
        folded,
        vec![2],
        "the broadcast folds into coord0 (operand 2):\n{}",
        plan.summary()
    );
    assert_eq!(sources.len(), 2, "scatter is the two-launch sequence");
    // Launch 1 (init copy) has no folded operand here: byte-identical
    // to the flat template.
    assert_contains(&sources[0], &["if (i < n) out[i] = init[i];"], "scatter copy launch");
    // Launch 2: coord0 read through its chain at the SRC coordinates;
    // the write side (flat address, injectivity flags) untouched.
    assert_contains(
        &sources[1],
        &[
            // src-coordinate prelude over (2,3)
            "long long c1 = (long long)(rem % 3ULL); rem /= 3ULL;",
            "long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;",
            // the broadcast chain + trap, then the strided coord read
            "long long coord0_h0_0 = c0;",
            "if (coord0_h0_0 < 0 || coord0_h0_0 >= 2LL) __trap();",
            "long long coord0_idx = coord0_h0_0 * 1LL;",
            "coord = (long long)coord0[coord0_idx];",
            // the scatter's own value-extent traps stand
            "if (coord < 0 || coord >= 4LL) __trap();",
            "if (coord < 0 || coord >= 3LL) __trap();",
            // coord1 stays flat; the checked-write contract is intact
            "coord = (long long)coord1[i];",
            "if (atomicExch(&flags[flat], 1u) != 0u) __trap();",
            "out[flat] = src[i];",
        ],
        "scatter write launch",
    );
}

/// SCATTER with every read-side operand folded: init a permute view,
/// src a slice view, coord0 a broadcast view. All three fold; the
/// write side stays direct.
#[test]
fn scatter_lowers_all_read_side_folds() {
    let mut cx = Graph::new();
    let init_base = cx.tensor((3usize, 4usize));
    let src_base = cx.tensor((4usize, 3usize));
    let rows = cx.tensor_dtyped(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let init = init_base.permute((1, 0)); // (4,3), init[i][j] = init_base[j][i]
    let src = src_base.slice((1..3, ..)); // (2,3), src[i][j] = src_base[i+1][j]
    let out = init.scatter(&[rows.expand_dim(1, 3usize), cols], src).output();

    let init_vals: Vec<f32> = (0..12).map(|v| 100.0 + v as f32).collect();
    let src_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (init_base.id, init_vals.into()),
        (src_base.id, src_vals.into()),
        (rows.id, vec![3i32, 1].into()),
    ];

    // init^T rows: row i = [100+i, 104+i, 108+i]; src rows 1..3 of
    // src_base: [3,4,5], [6,7,8]. rows = [3,1]: out[3] = [3,4,5],
    // out[1] = [6,7,8], others = init.
    reference_values(
        &cx,
        &inputs,
        out.id,
        &[
            100., 104., 108., // init row 0
            6., 7., 8., // src row 1
            102., 106., 110., // init row 2
            3., 4., 5., // src row 0
        ],
    );

    // Seed 1: the genome that folds all three read-side movements
    // (cost-tied spellings; see view_search_options).
    let plan = plan_for(&cx, &inputs, 1);
    let (sources, folded) = the_one(&plan, "ScatterFunctionalGeneric");
    assert_eq!(
        folded,
        vec![0, 1, 2],
        "init, src, and coord0 all read through folds:\n{}",
        plan.summary()
    );
    // Launch 1: init read through the permute chain at DEST coordinates.
    assert_contains(
        &sources[0],
        &[
            "long long init_h0_0 = c1;",
            "if (init_h0_0 < 0 || init_h0_0 >= 3LL) __trap();",
            "long long init_h0_1 = c0;",
            "if (init_h0_1 < 0 || init_h0_1 >= 4LL) __trap();",
            "long long init_idx = init_h0_0 * 4LL + init_h0_1 * 1LL;",
            "out[i] = init[init_idx];",
        ],
        "scatter all-folds copy launch",
    );
    // Launch 2: src through the slice chain (+1 row offset), coord0
    // through the broadcast chain; write side untouched.
    assert_contains(
        &sources[1],
        &[
            "long long coord0_h0_0 = c0;",
            "coord = (long long)coord0[coord0_idx];",
            "long long src_h0_0 = (c0 + 1LL);",
            "if (src_h0_0 < 0 || src_h0_0 >= 4LL) __trap();",
            "long long src_h0_1 = c1;",
            "if (src_h0_1 < 0 || src_h0_1 >= 3LL) __trap();",
            "long long src_idx = src_h0_0 * 3LL + src_h0_1 * 1LL;",
            "if (atomicExch(&flags[flat], 1u) != 0u) __trap();",
            "out[flat] = src[src_idx];",
        ],
        "scatter all-folds write launch",
    );
}

/// MATERIALIZE with a folded input: a view chain folded onto the
/// materialize's parent operand — the whisper field shape
/// ("IndexMapApplyMaterialize: operand 0 carries a composed access")
/// in miniature. The op's own map lands on the input VALUE's
/// coordinates; the folded chain composes ON TOP down to the residence.
#[test]
fn materialize_lowers_a_folded_input_operand() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize));
    // A pure movement chain into a pinned output: the planner must
    // land the result in the caller's dense buffer, so one movement
    // materializes — and the other folds onto its input operand.
    let out = x.permute((1, 0)).slice((0..2, ..)).output();

    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(x.id, (0..6).map(|v| v as f32).collect::<Vec<f32>>().into())];

    // x^T rows 0..2 of (3,2): [[0,3],[1,4]].
    reference_values(&cx, &inputs, out.id, &[0., 3., 1., 4.]);

    let plan = plan_for(
        &cx,
        &inputs,
        5, // the seed whose genome elects the materialize spelling (cost-tied with copy+fold)
    );
    let (sources, folded) = the_one(&plan, "IndexMapApplyMaterialize");
    assert_eq!(
        folded,
        vec![0],
        "the permute folds onto the materialize input:\n{}",
        plan.summary()
    );
    assert_contains(
        &sources[0],
        &[
            // The op's own map lands on the input VALUE's coordinates,
            // trapped against the value extents (3,2)...
            "if (idx < 0 || idx >= 3LL) __trap();",
            "long long parent_c0 = idx;",
            "if (idx < 0 || idx >= 2LL) __trap();",
            "long long parent_c1 = idx;",
            // ...and the folded permute chain is evaluated at THOSE
            // (composition on top), per-axis trapped against x (2,3).
            "long long parent_h0_0 = parent_c1;",
            "if (parent_h0_0 < 0 || parent_h0_0 >= 2LL) __trap();",
            "long long parent_h0_1 = parent_c0;",
            "if (parent_h0_1 < 0 || parent_h0_1 >= 3LL) __trap();",
            "long long parent_idx = parent_h0_0 * 3LL + parent_h0_1 * 1LL;",
            "out[i] = parent[parent_idx];",
        ],
        "materialize folded input",
    );
    assert!(!sources[0].contains("pflat"), "no flat parent read remains:\n{}", sources[0]);
}

