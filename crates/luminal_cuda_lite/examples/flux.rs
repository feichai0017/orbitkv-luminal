//! MiniDit (flux2 family) on CUDA-lite: BLOCKED — this example does not
//! pretend to run.
//!
//! mini_flux is blocked by a known pre-existing SEARCH divergence (the
//! rejoin-divergence family), upstream of any device concern; the
//! blockage is identical with or without the `device` feature. The
//! tree's own record of it is the ignore attribute on
//! `examples/mini/flux/tests/fidelity.rs::mini_dit_matches_scalar_reference`:
//!
//! > BLOCKED on the rejoin-divergence ruling: three concat/view
//! > spellings fixed (matmul rope, flat V concat, split out-projection,
//! > scatter-assembled joint sequence) and the graph still finds a
//! > slice-through-elementwise-distribution road into a view stack
//! > (stage-8 probe). The adaLN broadcast-modulation architecture
//! > generates these roads structurally; unblock = stratified
//! > composition or structural map entries. Probes: probe_dit_stages /
//! > probe_dit_round_driver.
//!
//! What this example DOES do: build the MiniDit graph with the mini
//! crate's own builder at the canonical dims
//! (`examples/mini/flux/src/bin/measure_plan.rs`) and prove the RECORD
//! is clean — the blockage begins at search saturation, so attempting
//! the search here would diverge, not fail fast. Then it prints the
//! blockage note and exits nonzero.
//!
//! Run: cargo run -p luminal_cuda_lite --example flux [--features device]

use luminal::prelude::*;
use mini_flux::MiniDit;

fn main() {
    // Canonical dims, verbatim from the mini's measure harness.
    const IN_CH: usize = 4;
    const TXT_DIM: usize = 6;
    const D: usize = 16;
    const NH: usize = 2;
    const HD: usize = 8;
    const MLP: usize = 6;
    const T_HALF: usize = 2;
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
    let _velocity = model
        .forward(latent, text, t, guidance, rope_cos, rope_sin, rope_rot, joint_base)
        .output();

    match cx.logical.model_text() {
        Ok(model_text) => {
            let rows = model_text.lines().filter(|l| !l.trim().is_empty()).count();
            println!("flux: MiniDit records cleanly ({rows} model rows) — the graph itself is fine.");
        }
        Err(e) => {
            // Louder than the known blockage: the record itself broke.
            eprintln!("flux: RECORD-POISONED (this is NEW, not the known blockage): {e}");
            std::process::exit(1);
        }
    }

    println!(
        "flux: BLOCKED — mini_flux (MiniDit) is blocked by a known pre-existing search \
         divergence, so this example does not run the model on either runtime.\n\
         The tree's record (examples/mini/flux/tests/fidelity.rs, ignore attribute):\n\
         \"BLOCKED on the rejoin-divergence ruling: three concat/view spellings fixed \
         (matmul rope, flat V concat, split out-projection, scatter-assembled joint \
         sequence) and the graph still finds a slice-through-elementwise-distribution \
         road into a view stack (stage-8 probe). The adaLN broadcast-modulation \
         architecture generates these roads structurally; unblock = stratified \
         composition or structural map entries. Probes: probe_dit_stages / \
         probe_dit_round_driver.\"\n\
         When the divergence ruling lands, promote this example to the shared \
         differential (support::device::run_differential) like its siblings."
    );
    std::process::exit(1);
}
