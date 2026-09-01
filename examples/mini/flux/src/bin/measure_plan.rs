//! MiniDit (flux2 family) demo on the reference runtime: one denoising
//! velocity prediction — adaLN conditioning from (t, guidance), one
//! double-stream + one single-stream block over [txt ‖ img] tokens,
//! host-precomputed interleaved-pair RoPE tables.
//! Run: cargo run --release -p mini_flux

use luminal::prelude::*;
use luminal_nn::rope_pairing_matrix;
use mini_flux::{mini_dit_rope_tables, MiniDit};

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

fn main() {
    let __t_rec = std::time::Instant::now();
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
        .forward(
            latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base,
        )
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
        (
            model.single_proj.weight.id,
            weights(D * (3 * D + 2 * MLP), 527).into(),
        ),
        (model.single_out_attn.weight.id, weights(D * D, 531).into()),
        (model.single_out_mlp.weight.id, weights(MLP * D, 532).into()),
        (model.single_qnorm.id, weights(HD, 529).into()),
        (model.single_knorm.id, weights(HD, 530).into()),
    ];
    measure_plan(&cx, &pairs, __t_rec);
}

#[allow(dead_code)]
fn measure_plan(
    cx: &Graph,
    pairs: &[(petgraph::graph::NodeIndex, TypedBuffer)],
    t0: std::time::Instant,
) {
    let rec_us = t0.elapsed().as_micros();
    let model = match cx.logical.model_text() {
        Ok(m) => m,
        Err(e) => {
            println!("RECORD-POISONED: {e}");
            return;
        }
    };
    let rows = model.lines().filter(|l| !l.trim().is_empty()).count();
    let applies = model.matches("(LogicalIndexMapApply").count();
    let mut depth = std::collections::HashMap::new();
    let mut max_chain = 0usize;
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for line in model.lines() {
        if let Some(rest) = line.strip_prefix("(let v") {
            let id: usize = rest.split_whitespace().next().unwrap().parse().unwrap();
            let d = if rest.contains("(LogicalIndexMapApply v") {
                let op: usize = rest
                    .split("(LogicalIndexMapApply v")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                depth.get(&op).copied().unwrap_or(0) + 1
            } else {
                0
            };
            if d > 0 {
                *hist.entry(d).or_insert(0) += 1;
            }
            max_chain = max_chain.max(d);
            depth.insert(id, d);
        }
    }
    println!("MODEL rows={rows} applies={applies} max_apply_chain={max_chain} record_us={rec_us}");
    println!("CHAIN_DEPTH_HIST {hist:?}");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rt = luminal_reference::ReferenceRuntime::load(cx).expect("native load");
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            rt.bind_dyn_range(*var, *value as u64, *value as u64)
                .expect("dyn pin binds");
        }
        let data = pairs.iter().cloned().collect();
        let t = std::time::Instant::now();
        let outcome = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .expect("search finds a plan");
        let search_ms = t.elapsed().as_millis();
        println!("SEARCH wall_ms={search_ms} [{}]", outcome.timings.summary());
        println!("REFUSALS {}", outcome.refusal_breakdown.summary());
        let summary = outcome.best_plan.summary();
        let mut label_counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut in_ops = false;
        let mut total = 0usize;
        for line in summary.lines() {
            if line.starts_with("ops (") {
                in_ops = true;
                continue;
            }
            if line.starts_with("anti (") {
                in_ops = false;
            }
            if in_ops {
                if let Some(rest) = line.strip_prefix("  ") {
                    let label = rest.split(':').next().unwrap_or("").to_string();
                    *label_counts.entry(label).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
        let buffers_line = summary
            .lines()
            .find(|l| l.starts_with("buffers ("))
            .unwrap_or("buffers (?)")
            .to_string();
        let allocs = summary
            .lines()
            .filter(|l| l.trim_start().starts_with("alloc#"))
            .count();
        println!("PLAN total_ops={total} {buffers_line} allocated_buffers={allocs}");
        for (label, n) in &label_counts {
            println!("PLANOP {label} {n}");
        }
        for (id, v) in pairs {
            rt.set_data(*id, v.clone());
        }
        let t = std::time::Instant::now();
        rt.execute().expect("winner executes");
        println!("EXECUTE wall_ms={}", t.elapsed().as_millis());
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        println!("LADDER PANICKED: {msg}");
    }
}
