//! Probe: the TRUE frontend spelling of x.matmul(w.permute((1,0))) — the
//! Linear::new_permuted folded-permute case A[m,k],B[n,k]. Run by name with --nocapture.

use luminal::graph::Graph;

#[test]
fn print_amk_bnk_spelling() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize));
    let w = cx.tensor((3usize, 4usize)); // stored [n, k]
    let _out = x.matmul(w.permute((1usize, 0usize))).output();
    println!("=== native_program for x.matmul(w.permute((1,0))) ===");
    println!(
        "{}",
        cx.logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("recorder clean")
            .text
    );
}
