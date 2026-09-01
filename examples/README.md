# Logical model examples

Everything under this directory is backend-neutral model definition. These
crates may depend on `luminal` and reusable `luminal_nn` building blocks, but
not on a runtime crate. They do not search, compile, execute, download weights,
or perform application I/O.

The full-size definitions correspond to the mini model families:

| Mini definition | Full logical definition | Checkpoint/model |
| --- | --- | --- |
| `mini/llama3` | `llama3` | Meta-Llama-3-8B-Instruct |
| `mini/qwen3` | `qwen3` | Qwen3-4B |
| `mini/gemma3` | `gemma3` | Gemma-3-4B-IT text tower |
| `mini/qwen3_moe` | `qwen3_moe` | Qwen3-30B-A3B |
| `mini/gemma4_moe` | `gemma4_moe` | Gemma-4-26B-A4B text tower |
| `mini/whisper` | `whisper` | Whisper tiny.en |
| `mini/conv` | `yolo_v11` | YOLO11n |
| `mini/flux` | `flux2` | FLUX.2-dev transformer |

Runnable, full-size CUDA Lite applications live in
`crates/luminal_cuda_lite/examples`. The mini definitions remain the small
execution-smoke fixtures used by runtime test suites.
