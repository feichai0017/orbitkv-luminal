use luminal::prelude::*;
fn main() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize));
    let w = cx.tensor((3usize, 4usize));
    let _ = x.matmul(w.permute((1, 0))).output();
    println!("{}", cx.logical.model_text().unwrap());
}
