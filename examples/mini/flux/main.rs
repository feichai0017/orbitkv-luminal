//! MiniDit (flux2 family) demo on the reference runtime: one denoising
//! velocity prediction — adaLN conditioning from (t, guidance), one
//! double-stream + one single-stream block over [txt ‖ img] tokens,
//! host-precomputed interleaved-pair RoPE tables.
//! Run: cargo run --release --example mini_flux

use luminal::prelude::*;
use luminal_nn::{mini_dit_rope_tables, rope_pairing_matrix, MiniDit};

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6).collect()
}

fn main() {
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
    let pairs = vec![
        (latent.id, weights(S_IMG * IN_CH, 540)),
        (text.id, weights(S_TXT * TXT_DIM, 541)),
        (t.id, vec![0.35]),
        (guidance.id, vec![0.8]),
        (rope_cos.id, cos_table),
        (rope_sin.id, sin_table),
        (rope_rot.id, rope_pairing_matrix(HD, true)),
        (joint_base.id, vec![0.0; S * D]),
        (model.x_embed.weight.id, weights(IN_CH * D, 500)),
        (model.ctx_embed.weight.id, weights(TXT_DIM * D, 501)),
        (model.t_mlp1.weight.id, weights(T_CH * D, 502)),
        (model.t_mlp2.weight.id, weights(D * D, 503)),
        (model.g_mlp1.weight.id, weights(T_CH * D, 504)),
        (model.g_mlp2.weight.id, weights(D * D, 505)),
        (model.mod_img.weight.id, weights(D * 6 * D, 506)),
        (model.mod_txt.weight.id, weights(D * 6 * D, 507)),
        (model.mod_single.weight.id, weights(D * 3 * D, 508)),
        (model.norm_out.weight.id, weights(D * 2 * D, 509)),
        (model.proj_out.weight.id, weights(D * IN_CH, 510)),
        (model.img_q.weight.id, weights(D * D, 511)),
        (model.img_k.weight.id, weights(D * D, 512)),
        (model.img_v.weight.id, weights(D * D, 513)),
        (model.img_out.weight.id, weights(D * D, 514)),
        (model.txt_q.weight.id, weights(D * D, 515)),
        (model.txt_k.weight.id, weights(D * D, 516)),
        (model.txt_v.weight.id, weights(D * D, 517)),
        (model.txt_out.weight.id, weights(D * D, 518)),
        (model.img_qnorm.id, weights(HD, 519)),
        (model.img_knorm.id, weights(HD, 520)),
        (model.txt_qnorm.id, weights(HD, 521)),
        (model.txt_knorm.id, weights(HD, 522)),
        (model.ff_in.weight.id, weights(D * 2 * MLP, 523)),
        (model.ff_out.weight.id, weights(MLP * D, 524)),
        (model.ctx_ff_in.weight.id, weights(D * 2 * MLP, 525)),
        (model.ctx_ff_out.weight.id, weights(MLP * D, 526)),
        (model.single_proj.weight.id, weights(D * (3 * D + 2 * MLP), 527)),
        (model.single_out_attn.weight.id, weights(D * D, 531)),
        (model.single_out_mlp.weight.id, weights(MLP * D, 532)),
        (model.single_qnorm.id, weights(HD, 529)),
        (model.single_knorm.id, weights(HD, 530)),
    ];
    let rt = luminal::test_support::run_ssa(&cx, &pairs);
    println!("velocity: {:?}", rt.get_f32(velocity.id).unwrap());
}
