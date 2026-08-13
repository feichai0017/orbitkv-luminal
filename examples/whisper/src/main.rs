use luminal::implementation_search::ImplementationSearchOptions;
use whisper::model::WhisperDims;
use whisper::{TOKEN_EOT, TOKEN_NO_TIMESTAMPS, TOKEN_SOT, TranscribeStep, Transcriber, audio, greedy_pick, hf, weights};

const USAGE: &str = "\
whisper — openai/whisper-tiny.en as pure logical ops (encoder + decoder, one graph)

USAGE: cargo run --release -p whisper [-- OPTIONS]
  --wav <path>        16 kHz mono WAV (default: the bundled JFK sample)
  --tokens <n>        max tokens to generate (default 32)
  --random-weights    skip the download; deterministic fake parameters
  --help              this text";

fn main() {
    let mut wav: Option<String> = None;
    let mut gen_tokens = 32usize;
    let mut random_weights = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| panic!("{name} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--wav" => wav = Some(value("--wav")),
            "--tokens" => gen_tokens = value("--tokens").parse().expect("--tokens: integer"),
            "--random-weights" => random_weights = true,
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

    let dims = WhisperDims::whisper_tiny_en();
    let mel = if random_weights {
        vec![0.1f32; dims.n_mels * dims.mel_frames()]
    } else {
        let samples = match &wav {
            Some(path) => audio::load_wav(path).expect("wav loads"),
            None => audio::load_wav_bytes(include_bytes!("../assets/jfk.wav"))
                .expect("bundled sample decodes"),
        };
        audio::log_mel_spectrogram(&samples, dims.n_mels)
    };

    println!("Recording the transcribe-step graph...");
    let step = TranscribeStep::build(&dims);
    let (pairs, tokenizer) = if random_weights {
        (weights::random_weights(&step.cx), None)
    } else {
        let dir = hf::prepare_hf_model("openai/whisper-tiny.en").expect("download");
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        (
            weights::load_safetensors_weights(&step.cx, &dir).expect("weights load"),
            Some(tokenizer),
        )
    };
    let mut transcriber = Transcriber::start(
        step,
        &mel,
        &pairs,
        &ImplementationSearchOptions {
            generations: 2,
            generation_size: 4,
            mutations: 2,
            trials: 1,
            seed: 0,
        },
    )
    .expect("search");
    drop(pairs);

    // Forced prompt: <|startoftranscript|> <|notimestamps|>.
    let mut logits = Vec::new();
    for forced in [TOKEN_SOT, TOKEN_NO_TIMESTAMPS] {
        logits = transcriber.step(forced).expect("forced step");
    }
    let mut text_tokens: Vec<u32> = Vec::new();
    for step_index in 0..gen_tokens {
        let next = greedy_pick(&logits, step_index == 0, TOKEN_SOT);
        if next == TOKEN_EOT {
            break;
        }
        text_tokens.push(next);
        logits = transcriber.step(next).expect("step");
    }
    match &tokenizer {
        Some(tokenizer) => {
            let text = tokenizer.decode(&text_tokens, true).expect("decode");
            println!("{text}");
        }
        None => println!("{text_tokens:?}"),
    }
}
