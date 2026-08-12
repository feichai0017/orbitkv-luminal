use llama3_1_fp8::{Fp8RunConfig, run_llama3_1_fp8};

const USAGE: &str = "\
llama3_1_fp8 — nvidia/Llama-3.1-8B-Instruct-FP8 as pure logical ops (E4M3FN weights)

USAGE: cargo run --release -p llama3_1_fp8 [-- OPTIONS]
  --prompt <text>     user prompt (llama3 chat template applied)
  --layers <n>        transformer layers to instantiate, 1..=32 (default 1)
  --tokens <n>        tokens to generate (default 8)
  --max-seq <n>       cache slots / maximum sequence length (default 64)
  --repo <id>         HuggingFace repo (default nvidia/Llama-3.1-8B-Instruct-FP8)
  --random-weights    skip the download; deterministic fake parameters
  --help              this text";

fn main() {
    let mut config = Fp8RunConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--prompt" => config.prompt = value("--prompt"),
            "--layers" => config.layers = value("--layers").parse().expect("--layers: integer"),
            "--tokens" => config.gen_tokens = value("--tokens").parse().expect("--tokens: integer"),
            "--max-seq" => config.max_seq = value("--max-seq").parse().expect("--max-seq: integer"),
            "--repo" => config.repo_id = value("--repo"),
            "--random-weights" => config.random_weights = true,
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
    if let Err(err) = run_llama3_1_fp8(config) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
