//! MiniDit scalar-reference fidelity (moved from luminal_nn's mini.rs
//! test module, 2026-08-13).

use luminal::prelude::*;
use scalar_refs::*;
use mini_flux::*;

/// MiniDit vs a full scalar reference: 1 double + 1 single block,
/// d=16, 2 heads (head_dim 8 = the 4-axis rope table width), 4 image
/// tokens (2×2 grid) + 2 text tokens, adaLN conditioning from
/// t/guidance scalars. The two ordering conventions the recon flagged
/// as silent-mismatch bait — (shift, scale, gate) in block triples vs
/// (scale, shift) at norm_out, and txt-before-img in every
/// concat/split — are exercised by construction.
#[test]
#[ignore = "BLOCKED on the rejoin-divergence ruling: three concat/view spellings \
            fixed (matmul rope, flat V concat, split out-projection, scatter-\
            assembled joint sequence) and the graph still finds a slice-through-\
            elementwise-distribution road into a view stack (stage-8 probe). The \
            adaLN broadcast-modulation architecture generates these roads \
            structurally; unblock = stratified composition or structural map \
            entries. Probes: probe_dit_stages / probe_dit_round_driver."]
fn mini_dit_matches_scalar_reference() {
    const IN_CH: usize = 4;
    const TXT_DIM: usize = 6;
    const D: usize = 16;
    const NH: usize = 2;
    const HD: usize = 8;
    const MLP: usize = 6;
    const T_HALF: usize = 2;
    const T_CH: usize = 2 * T_HALF;
    const S_TXT: usize = 2;
    const GRID: usize = 2;
    const S_IMG: usize = GRID * GRID;
    const S: usize = S_TXT + S_IMG;

    let mut cx = Graph::new();
    let model = MiniDit::new(IN_CH, TXT_DIM, D, NH, MLP, T_HALF, S_TXT, &mut cx);
    let latent = cx.tensor((S_IMG, IN_CH));
    let text = cx.tensor((S_TXT, TXT_DIM));
    let t = cx.tensor(1);
    let guidance = cx.tensor(1);
    let rope_cos = cx.tensor((S, HD));
    let rope_sin = cx.tensor((S, HD));
    let rope_rot = cx.tensor((HD, HD));
    let joint_base = cx.tensor((S, D));
    let velocity = model
        .forward(latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base)
        .output();

    let (cos_table, sin_table) = mini_dit_rope_tables(S_TXT, GRID, GRID);
    let rot_matrix = luminal_nn::rope_pairing_matrix(HD, true);
    let latent_vals = weights(S_IMG * IN_CH, 540);
    let text_vals = weights(S_TXT * TXT_DIM, 541);
    let (t_val, g_val) = (0.35f32, 0.8f32);
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (latent.id, latent_vals.clone().into()),
        (text.id, text_vals.clone().into()),
        (t.id, vec![t_val].into()),
        (guidance.id, vec![g_val].into()),
        (rope_cos.id, cos_table.clone().into()),
        (rope_sin.id, sin_table.clone().into()),
        (rope_rot.id, rot_matrix.clone().into()),
        (joint_base.id, vec![0.0; S * D].into()),
        (model.x_embed.weight.id, weights(IN_CH * D, 500).into()),
        (model.ctx_embed.weight.id, weights(TXT_DIM * D, 501).into()),
        (model.t_mlp1.weight.id, weights(T_CH * D, 502).into()),
        (model.t_mlp2.weight.id, weights(D * D, 503).into()),
        (model.g_mlp1.weight.id, weights(T_CH * D, 504).into()),
        (model.g_mlp2.weight.id, weights(D * D, 505).into()),
        (model.mod_img.weight.id, weights(D * 6 * D, 506).into()),
        (model.mod_txt.weight.id, weights(D * 6 * D, 507).into()),
        (model.mod_single.weight.id, weights(D * 3 * D, 508).into()),
        (model.norm_out.weight.id, weights(D * 2 * D, 509).into()),
        (model.proj_out.weight.id, weights(D * IN_CH, 510).into()),
        (model.img_q.weight.id, weights(D * D, 511).into()),
        (model.img_k.weight.id, weights(D * D, 512).into()),
        (model.img_v.weight.id, weights(D * D, 513).into()),
        (model.img_out.weight.id, weights(D * D, 514).into()),
        (model.txt_q.weight.id, weights(D * D, 515).into()),
        (model.txt_k.weight.id, weights(D * D, 516).into()),
        (model.txt_v.weight.id, weights(D * D, 517).into()),
        (model.txt_out.weight.id, weights(D * D, 518).into()),
        (model.img_qnorm.id, weights(HD, 519).into()),
        (model.img_knorm.id, weights(HD, 520).into()),
        (model.txt_qnorm.id, weights(HD, 521).into()),
        (model.txt_knorm.id, weights(HD, 522).into()),
        (model.ff_in.weight.id, weights(D * 2 * MLP, 523).into()),
        (model.ff_out.weight.id, weights(MLP * D, 524).into()),
        (model.ctx_ff_in.weight.id, weights(D * 2 * MLP, 525).into()),
        (model.ctx_ff_out.weight.id, weights(MLP * D, 526).into()),
        (model.single_proj.weight.id, weights(D * (3 * D + 2 * MLP), 527).into()),
        (model.single_out_attn.weight.id, weights(D * D, 531).into()),
        (model.single_out_mlp.weight.id, weights(MLP * D, 532).into()),
        (model.single_qnorm.id, weights(HD, 529).into()),
        (model.single_knorm.id, weights(HD, 530).into()),
    ];

    // ---- scalar reference ----
    // Row-wise helpers (test_refs' are single-row).
    let matmul_rows = |x: &[f32], w: &[f32], rows: usize, in_w: usize, out_w: usize| {
        let mut out = Vec::with_capacity(rows * out_w);
        for row in 0..rows {
            out.extend(ref_matmul(&x[row * in_w..(row + 1) * in_w], w, in_w, out_w));
        }
        out
    };
    let ln_rows = |x: &[f32], rows: usize| {
        let width = x.len() / rows;
        let mut out = Vec::with_capacity(x.len());
        for row in 0..rows {
            out.extend(ref_layer_norm(&x[row * width..(row + 1) * width], 1e-6));
        }
        out
    };
    // adaLN: x·(1+scale)+shift, modulation rows broadcast over rows.
    let ada_rows = |x: &[f32], scale: &[f32], shift: &[f32], rows: usize| {
        let width = x.len() / rows;
        let mut out = Vec::with_capacity(x.len());
        for row in 0..rows {
            for col in 0..width {
                out.push(x[row * width + col] * (1.0 + scale[col]) + shift[col]);
            }
        }
        out
    };
    let gate_rows = |x: &[f32], g: &[f32], rows: usize| {
        let width = x.len() / rows;
        (0..rows)
            .flat_map(|row| (0..width).map(move |col| x[row * width + col] * g[col]))
            .collect::<Vec<f32>>()
    };
    // Per-head QK-norm over (rows, D) with heads side by side.
    let head_norm_rows = |x: &[f32], w: &[f32], rows: usize| {
        let mut out = Vec::with_capacity(x.len());
        for row in 0..rows {
            out.extend(ref_rms_head_norm(&x[row * D..(row + 1) * D], HD, w));
        }
        out
    };
    // Interleaved-pair rope over (rows, D): per head, per pair:
    // x'[2m] = x[2m]·cos − x[2m+1]·sin; x'[2m+1] = x[2m+1]·cos + x[2m]·sin.
    let rope_rows = |x: &[f32], rows: usize| {
        let mut out = x.to_vec();
        for row in 0..rows {
            for head in 0..NH {
                for pair in 0..HD / 2 {
                    let base = row * D + head * HD + 2 * pair;
                    let (c0, s0) = (cos_table[row * HD + 2 * pair], sin_table[row * HD + 2 * pair]);
                    let (c1, s1) = (
                        cos_table[row * HD + 2 * pair + 1],
                        sin_table[row * HD + 2 * pair + 1],
                    );
                    let (even, odd) = (x[base], x[base + 1]);
                    out[base] = even * c0 - odd * s0;
                    out[base + 1] = odd * c1 + even * s1;
                }
            }
        }
        out
    };
    let swiglu_rows = |u: &[f32], rows: usize| {
        let mut out = Vec::with_capacity(rows * MLP);
        for row in 0..rows {
            let row = &u[row * 2 * MLP..(row + 1) * 2 * MLP];
            out.extend(
                ref_silu(&row[..MLP])
                    .iter()
                    .zip(&row[MLP..])
                    .map(|(a, b)| a * b),
            );
        }
        out
    };
    let add = |a: &[f32], b: &[f32]| -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    };

    // Conditioning.
    let sinusoid = |x: f32| -> Vec<f32> {
        let args: Vec<f32> = (0..T_HALF)
            .map(|i| 1000.0 * x * (-(i as f32) * (10000f32).ln() / T_HALF as f32).exp())
            .collect();
        args.iter()
            .map(|a| a.cos())
            .chain(args.iter().map(|a| a.sin()))
            .collect()
    };
    let temb = add(
        &ref_matmul(
            &ref_silu(&ref_matmul(&sinusoid(t_val), &weights(T_CH * D, 502), T_CH, D)),
            &weights(D * D, 503),
            D,
            D,
        ),
        &ref_matmul(
            &ref_silu(&ref_matmul(&sinusoid(g_val), &weights(T_CH * D, 504), T_CH, D)),
            &weights(D * D, 505),
            D,
            D,
        ),
    );
    let cond = ref_silu(&temb);
    let m_img = ref_matmul(&cond, &weights(D * 6 * D, 506), D, 6 * D);
    let m_txt = ref_matmul(&cond, &weights(D * 6 * D, 507), D, 6 * D);
    let m_single = ref_matmul(&cond, &weights(D * 3 * D, 508), D, 3 * D);
    let triple = |m: &[f32], set: usize| {
        let base = set * 3 * D;
        (
            m[base..base + D].to_vec(),           // shift
            m[base + D..base + 2 * D].to_vec(),   // scale
            m[base + 2 * D..base + 3 * D].to_vec(), // gate
        )
    };

    // Double-stream block.
    let (shift0, scale0, gate0) = triple(&m_img, 0);
    let (shift1, scale1, gate1) = triple(&m_img, 1);
    let (c_shift0, c_scale0, c_gate0) = triple(&m_txt, 0);
    let (c_shift1, c_scale1, c_gate1) = triple(&m_txt, 1);
    let mut img = matmul_rows(&latent_vals, &weights(IN_CH * D, 500), S_IMG, IN_CH, D);
    let mut txt = matmul_rows(&text_vals, &weights(TXT_DIM * D, 501), S_TXT, TXT_DIM, D);
    let img_n = ada_rows(&ln_rows(&img, S_IMG), &scale0, &shift0, S_IMG);
    let txt_n = ada_rows(&ln_rows(&txt, S_TXT), &c_scale0, &c_shift0, S_TXT);
    let q_img = head_norm_rows(
        &matmul_rows(&img_n, &weights(D * D, 511), S_IMG, D, D),
        &weights(HD, 519),
        S_IMG,
    );
    let k_img = head_norm_rows(
        &matmul_rows(&img_n, &weights(D * D, 512), S_IMG, D, D),
        &weights(HD, 520),
        S_IMG,
    );
    let v_img = matmul_rows(&img_n, &weights(D * D, 513), S_IMG, D, D);
    let q_txt = head_norm_rows(
        &matmul_rows(&txt_n, &weights(D * D, 515), S_TXT, D, D),
        &weights(HD, 521),
        S_TXT,
    );
    let k_txt = head_norm_rows(
        &matmul_rows(&txt_n, &weights(D * D, 516), S_TXT, D, D),
        &weights(HD, 522),
        S_TXT,
    );
    let v_txt = matmul_rows(&txt_n, &weights(D * D, 517), S_TXT, D, D);
    // txt first, then rope, then joint non-causal attention.
    let concat_rows = |a: &[f32], b: &[f32]| {
        let mut joined = a.to_vec();
        joined.extend_from_slice(b);
        joined
    };
    let q = rope_rows(&concat_rows(&q_txt, &q_img), S);
    let k = rope_rows(&concat_rows(&k_txt, &k_img), S);
    let v = concat_rows(&v_txt, &v_img);
    let attn = ref_attention(&q, &k, &v, S, S, NH, HD);
    let attn_txt = &attn[..S_TXT * D];
    let attn_img = &attn[S_TXT * D..];
    img = add(
        &img,
        &gate_rows(&matmul_rows(attn_img, &weights(D * D, 514), S_IMG, D, D), &gate0, S_IMG),
    );
    txt = add(
        &txt,
        &gate_rows(&matmul_rows(attn_txt, &weights(D * D, 518), S_TXT, D, D), &c_gate0, S_TXT),
    );
    let ff = swiglu_rows(
        &matmul_rows(
            &ada_rows(&ln_rows(&img, S_IMG), &scale1, &shift1, S_IMG),
            &weights(D * 2 * MLP, 523),
            S_IMG,
            D,
            2 * MLP,
        ),
        S_IMG,
    );
    img = add(
        &img,
        &gate_rows(&matmul_rows(&ff, &weights(MLP * D, 524), S_IMG, MLP, D), &gate1, S_IMG),
    );
    let c_ff = swiglu_rows(
        &matmul_rows(
            &ada_rows(&ln_rows(&txt, S_TXT), &c_scale1, &c_shift1, S_TXT),
            &weights(D * 2 * MLP, 525),
            S_TXT,
            D,
            2 * MLP,
        ),
        S_TXT,
    );
    txt = add(
        &txt,
        &gate_rows(&matmul_rows(&c_ff, &weights(MLP * D, 526), S_TXT, MLP, D), &c_gate1, S_TXT),
    );

    // Single-stream block over [txt ‖ img].
    let mut hidden = concat_rows(&txt, &img);
    let (s_shift, s_scale, s_gate) = triple(&m_single, 0);
    let normed = ada_rows(&ln_rows(&hidden, S), &s_scale, &s_shift, S);
    let proj = matmul_rows(&normed, &weights(D * (3 * D + 2 * MLP), 527), S, D, 3 * D + 2 * MLP);
    let width = 3 * D + 2 * MLP;
    let slice_cols = |x: &[f32], from: usize, to: usize| {
        let mut out = Vec::with_capacity(S * (to - from));
        for row in 0..S {
            out.extend_from_slice(&x[row * width + from..row * width + to]);
        }
        out
    };
    let q = rope_rows(
        &head_norm_rows(&slice_cols(&proj, 0, D), &weights(HD, 529), S),
        S,
    );
    let k = rope_rows(
        &head_norm_rows(&slice_cols(&proj, D, 2 * D), &weights(HD, 530), S),
        S,
    );
    let v = slice_cols(&proj, 2 * D, 3 * D);
    let attn = ref_attention(&q, &k, &v, S, S, NH, HD);
    let mlp_out = swiglu_rows(&slice_cols(&proj, 3 * D, 3 * D + 2 * MLP), S);
    // Row-split fused out-projection (mirrors single_out_attn/_mlp).
    let out_sum = add(
        &matmul_rows(&attn, &weights(D * D, 531), S, D, D),
        &matmul_rows(&mlp_out, &weights(MLP * D, 532), S, MLP, D),
    );
    hidden = add(&hidden, &gate_rows(&out_sum, &s_gate, S));

    // AdaLayerNormContinuous head — (scale, shift), REVERSED order.
    let img_final = &hidden[S_TXT * D..];
    let head = ref_matmul(&cond, &weights(D * 2 * D, 509), D, 2 * D);
    let (scale, shift) = (&head[..D], &head[D..]);
    let expected = matmul_rows(
        &ada_rows(&ln_rows(img_final, S_IMG), scale, shift, S_IMG),
        &weights(D * IN_CH, 510),
        S_IMG,
        D,
        IN_CH,
    );

    let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
    let mut rt = luminal::reference::ReferenceRuntime::load(&cx).expect("native load");
    rt.search(
        &data,
        &luminal::implementation_search::ImplementationSearchOptions::default(),
    )
    .expect("search finds a plan");
    for (id, values) in &pairs {
        rt.set_data(*id, values.clone());
    }
    rt.execute().expect("winner executes");
    assert_close(rt.get_f32(velocity.id).expect("velocity"), &expected);
}
