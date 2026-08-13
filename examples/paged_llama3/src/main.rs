use llama3::weights;
use paged_llama3::{BatchStep, Llama3Dims, Ticker};

const USAGE: &str = "\
paged_llama3 — Meta-Llama-3-8B-Instruct over the page-table cache driver
(two sequences share one pool; batched ticks; isolation by mask)

USAGE: cargo run --release -p paged_llama3 [-- OPTIONS]
  --layers <n>        transformer layers to instantiate (default 1)
  --slots <n>         shared pool capacity (default 64)
  --ticks <n>         batched decode ticks to run (default 6)
  --random-weights    (default; real-weight loading as in llama3)
  --help              this text";

fn main() {
    let mut layers = 1usize;
    let mut slots = 64usize;
    let mut ticks = 6usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--layers" => layers = value("--layers").parse().expect("--layers: integer"),
            "--slots" => slots = value("--slots").parse().expect("--slots: integer"),
            "--ticks" => ticks = value("--ticks").parse().expect("--ticks: integer"),
            "--random-weights" => {}
            "--help" | "-h" => {
                println!("{USAGE}");
                return;
            }
            other => {
                eprintln!("unknown argument '{other}'\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    let mut dims = Llama3Dims::llama3_8b();
    assert!(layers >= 1 && layers <= dims.layers);
    dims.layers = layers;
    println!("Recording the batched tick graph ({layers} layers, {slots} slots)...");
    let step = BatchStep::build(&dims, slots);
    let pairs = weights::random_weights(&step.cx);
    let mut ticker = Ticker::start(step, &pairs, &Default::default()).expect("search");

    let a = ticker.table.new_sequence();
    let b = ticker.table.new_sequence();
    let mut next = (11u32, 42u32);
    for tick in 0..ticks {
        let logits = ticker.tick(&[(a, next.0), (b, next.1)]).expect("tick");
        let vocab = logits.len() / paged_llama3::ROWS;
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap()
                .0 as u32
        };
        next = (argmax(&logits[..vocab]), argmax(&logits[vocab..]));
        println!("tick {tick}: A -> {}, B -> {}", next.0, next.1);
    }
}
