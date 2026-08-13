//! MiniDit (flux2 family) demo on the reference runtime: one denoising
//! velocity prediction — adaLN conditioning from (t, guidance), one
//! double-stream + one single-stream block over [txt ‖ img] tokens,
//! host-precomputed interleaved-pair RoPE tables.
//! Run: cargo run --release -p mini_flux

use luminal::prelude::*;
use luminal_nn::rope_pairing_matrix;
use mini_flux::{mini_dit_rope_tables, MiniDit};

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
    let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
        (latent.id, weights(S_IMG * IN_CH, 540).into()),
        (text.id, weights(S_TXT * TXT_DIM, 541).into()),
        (t.id, vec![0.35].into()),
        (guidance.id, vec![0.8].into()),
        (rope_cos.id, cos_table.into()),
        (rope_sin.id, sin_table.into()),
        (rope_rot.id, rope_pairing_matrix(HD, true).into()),
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
    let rt = luminal::test_support::run_ssa(&cx, &pairs);
    println!("velocity: {:?}", rt.get_f32(velocity.id).unwrap());
}
