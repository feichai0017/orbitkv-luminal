use luminal::prelude::*;
use luminal::shape::Expression;

/// Gather entire rows from a 2D tensor using row indices.
///
/// - `data`: (R, D) tensor
/// - `indices`: (N,) Int tensor of row indices
/// - `d`: the number of columns (D), must match data's second dimension
///
/// Returns: (N, D) tensor where output[i] = data[indices[i]]
pub fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // base[i] = indices[i] * D → flat starting position for each row
    let base = (indices * d).expand_dim(1, d); // (N, D) broadcast along cols

    // col[j] = j → column offsets 0..D
    let col = data.graph().arange(d as i32).expand_dim(0, n); // (N, D) broadcast along rows

    // flat_idx[i,j] = indices[i] * D + j
    let flat_idx = base + col;

    data.gather1d(flat_idx)
}

/// Scatter entire rows into a 2D tensor using row indices.
///
/// - `src`: (N, D) tensor of values to write
/// - `indices`: (N,) Int tensor of destination row indices
/// - `dest`: (R, D) tensor to write into (copied first, then overwritten at index positions)
/// - `d`: the number of columns (D)
///
/// Returns: (R, D) tensor where output = copy(dest); output[indices[i]] = src[i]
pub fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // Same index expansion as gather_rows
    let base = (indices * d).expand_dim(1, d);
    let col = src.graph().arange(d as i32).expand_dim(0, n);
    let flat_idx = base + col;

    src.scatter1d(flat_idx, dest)
}

/// Pure HLIR paged attention for one layer with causal masking.
///
/// Inputs:
/// - `q`:           (s, hidden)         f32 — query vectors
/// - `k_new`:       (s, kv_dim)         f32 — new key vectors
/// - `v_new`:       (s, kv_dim)         f32 — new value vectors
/// - `k_cache`:     (num_slots, kv_dim) f32 — key cache (preallocated)
/// - `v_cache`:     (num_slots, kv_dim) f32 — value cache (preallocated)
/// - `gather_idx`:  (ctx_len,)          Int — which cache slots to read
/// - `scatter_idx`: (s,)                Int — which cache slots to write new KV into
/// - `prev_seq`:    number of previously cached tokens (for causal mask offset)
/// - `n_heads`:     number of query heads
/// - `n_kv_heads`:  number of KV heads (for GQA)
/// - `head_dim`:    dimension per head
///
/// Returns: (attn_out, k_cache_new, v_cache_new)
///   - `attn_out`:     (s, hidden)         f32
///   - `k_cache_new`:  (num_slots, kv_dim) f32
///   - `v_cache_new`:  (num_slots, kv_dim) f32
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: Expression,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let s = q.dims()[0];
    let ctx = gather_idx.dims()[0];
    let cx = q.graph();

    // ── Phase 1: Write new KV into cache ──
    let k_cache = scatter_rows(k_new, scatter_idx, k_cache, kv_dim);
    let v_cache = scatter_rows(v_new, scatter_idx, v_cache, kv_dim);

    // ── Phase 2: Gather context KV from cache ──
    let k = gather_rows(k_cache, gather_idx, kv_dim); // (ctx, kv_dim)
    let v = gather_rows(v_cache, gather_idx, kv_dim); // (ctx, kv_dim)

    // ── Phase 3: Reshape for multi-head attention ──
    // Q: (s, hidden) → (s, n_heads, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    //                 → (n_kv_heads, kv_groups, s, head_dim)
    let q = q
        .split_dims(1, head_dim) // (s, n_heads, head_dim)
        .split_dims(1, kv_groups) // (s, n_kv_heads, kv_groups, head_dim)
        .permute((1, 2, 0, 3)); // (n_kv_heads, kv_groups, s, head_dim)

    // K: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, head_dim, ctx)
    let k = k
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 2, 0)); // (n_kv_heads, head_dim, ctx)

    // V: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, ctx, head_dim)
    let v = v
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 0, 2)); // (n_kv_heads, ctx, head_dim)

    // ── Phase 4: Attention ──
    // Broadcast K, V over kv_groups dimension
    let k = k.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, head_dim, ctx)
    let v = v.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, ctx, head_dim)

    // QK^T: (n_kv_heads, kv_groups, s, head_dim) @ (n_kv_heads, kv_groups, head_dim, ctx)
    //     → (n_kv_heads, kv_groups, s, ctx)
    let scores = q.matmul(k) * scale;

    // Build causal mask: query at position prev_seq+i can attend to context j iff j <= prev_seq+i.
    // row_vals[i] = prev_seq + i, col_vals[j] = j
    // mask[i,j] = -1e9 where row_vals[i] < col_vals[j], else 0
    let z = Expression::from('z');
    let row_vals = cx.iota(z + prev_seq, s).expand_dim(1, ctx); // (s, ctx)
    let col_vals = cx.arange(ctx).expand_dim(0, s); // (s, ctx)
    let mask = row_vals
        .cast(DType::F32)
        .lt(col_vals.cast(DType::F32))
        .cast(DType::F32)
        * -1e9;

    // Broadcast (s, ctx) → (n_kv_heads, kv_groups, s, ctx)
    let mask = mask.expand_dim(0, n_kv_heads).expand_dim(1, kv_groups);
    let scores = scores + mask;

    // Softmax over context dimension (axis 3)
    let weights = scores.softmax(3);

    // Weighted sum: (n_kv_heads, kv_groups, s, ctx) @ (n_kv_heads, kv_groups, ctx, head_dim)
    //            → (n_kv_heads, kv_groups, s, head_dim)
    let out = weights.matmul(v);

    // ── Phase 5: Reshape output ──
    // (n_kv_heads, kv_groups, s, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    // Head merge as an EXPLICIT view (A2: no tracker reassignment —
    // the old code silently reinterpreted the permuted view as fresh
    // contiguous storage): (s, n_kv, groups, hd) -> (s, n_heads*hd).
    let out = out.permute((2, 0, 1, 3)).merge_dims(1, 2).merge_dims(1, 2);

    (out, k_cache, v_cache)
}

#[cfg(test)]
mod tests {
    use super::{gather_rows, paged_attention, scatter_rows};
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::shape::Expression;
    use luminal::ssa_reference::SsaReferenceRuntime;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// Row gather through the M3 ladder: out[i] = data[indices[i]].
    #[test]
    fn gather_rows_selects_rows() {
        let mut cx = Graph::new();
        let data = cx.tensor((4, 3));
        let idx = cx.tensor_dtyped(2, DType::Int);
        let out = gather_rows(data, idx, 3).output();

        let data_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let idx_vals = vec![2.0f32, 0.0];
        let expected = vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0];

        let mut inputs = FxHashMap::default();
        inputs.insert(data.id, data_vals.clone());
        inputs.insert(idx.id, idx_vals.clone());
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(&inputs, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(data.id, data_vals);
        rt.set_data(idx.id, idx_vals);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Row scatter through the M3 ladder: copy dest, replace the indexed
    /// rows with src.
    #[test]
    fn scatter_rows_replaces_rows() {
        let mut cx = Graph::new();
        let src = cx.tensor((2, 3));
        let idx = cx.tensor_dtyped(2, DType::Int);
        let dest = cx.tensor((4, 3));
        let out = scatter_rows(src, idx, dest, 3).output();
        assert_eq!(out.dims(), dest.dims());

        let src_vals = vec![100.0f32, 101.0, 102.0, 200.0, 201.0, 202.0];
        let idx_vals = vec![1.0f32, 3.0];
        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let expected = vec![
            0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 6.0, 7.0, 8.0, 200.0, 201.0, 202.0,
        ];

        let mut inputs = FxHashMap::default();
        inputs.insert(src.id, src_vals.clone());
        inputs.insert(idx.id, idx_vals.clone());
        inputs.insert(dest.id, dest_vals.clone());
        let mut rt = SsaReferenceRuntime::load(&cx).expect("native load");
        rt.search(&inputs, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(src.id, src_vals);
        rt.set_data(idx.id, idx_vals);
        rt.set_data(dest.id, dest_vals);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Full paged attention, decode step: one new token (s=1) after one
    /// cached token, 2 KV heads with no grouping (kv_groups=1), head_dim 2.
    /// Reference computed by scalar loops below; the graph runs the plain
    /// extraction path (run_ssa) — the search ladder is exercised by the
    /// smaller units above.
    #[test]
    fn paged_attention_decode_step_matches_scalar_reference() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM; // == kv_dim here
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        let prev_seq = 1usize;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN));
        let k_new = cx.tensor((1, HIDDEN));
        let v_new = cx.tensor((1, HIDDEN));
        let k_cache = cx.tensor((SLOTS, HIDDEN));
        let v_cache = cx.tensor((SLOTS, HIDDEN));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let (attn, k_cache_new, v_cache_new) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            Expression::from(prev_seq),
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
        );
        let attn = attn.output();
        let k_cache_new = k_cache_new.output();
        let v_cache_new = v_cache_new.output();

        let q_vals = vec![0.5f32, -0.3, 0.8, 0.1];
        let k_new_vals = vec![0.2f32, 0.4, -0.1, 0.3];
        let v_new_vals = vec![1.0f32, -1.0, 0.5, 2.0];
        let k_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.1).collect();
        let v_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.2 + 1.0).collect();
        let gather_vals = vec![0.0f32, 1.0]; // context = slots 0, 1
        let scatter_vals = vec![1.0f32]; // new KV lands in slot 1

        // Scalar reference. Cache update: row 1 replaced by k_new/v_new.
        let mut k_cache_ref = k_cache_vals.clone();
        let mut v_cache_ref = v_cache_vals.clone();
        k_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&k_new_vals);
        v_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&v_new_vals);
        // Attention per head over the gathered context rows (slots 0, 1).
        // The query is token index prev_seq = 1, so both context positions
        // are causally visible.
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_ref = vec![0.0f32; HIDDEN];
        for h in 0..N_HEADS {
            let q_h = &q_vals[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let mut scores = [0.0f32; CTX];
            for (j, score) in scores.iter_mut().enumerate() {
                let slot = gather_vals[j] as usize;
                let k_row = &k_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                *score = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                let slot = gather_vals[j] as usize;
                let v_row = &v_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                for (d, v) in v_row.iter().enumerate() {
                    attn_ref[h * HEAD_DIM + d] += e / denom * v;
                }
            }
        }

        let rt = luminal::test_support::run_ssa(
            &cx,
            &[
                (q.id, q_vals),
                (k_new.id, k_new_vals),
                (v_new.id, v_new_vals),
                (k_cache.id, k_cache_vals),
                (v_cache.id, v_cache_vals),
                (gather_idx.id, gather_vals),
                (scatter_idx.id, scatter_vals),
            ],
        );
        assert_close(rt.get_f32(attn.id).expect("attn out"), &attn_ref);
        assert_close(rt.get_f32(k_cache_new.id).expect("k cache"), &k_cache_ref);
        assert_close(rt.get_f32(v_cache_new.id).expect("v cache"), &v_cache_ref);
    }
}
