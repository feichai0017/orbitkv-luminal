use luminal::prelude::*;
fn main() {
    let mut cx = Graph::new();
    let x = cx.tensor((8, 16));
    let w = cx.tensor((16, 12));
    let _ = x.matmul(w).output();
    println!("{}", cx.logical.model_text().unwrap());
}
