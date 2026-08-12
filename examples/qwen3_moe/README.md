# qwen3_moe — Qwen/Qwen3-30B-A3B

100% of this checkpoint: 48 layers, GQA 32/4 at head_dim 128, learned
per-head QK-norm before rope (eps 1e-6), rope theta 1e6 unscaled,
UNTIED lm_head, and the 128-expert top-8 MoE on every layer with the
Qwen3 scoring order — softmax over ALL experts first, then top-k
(stable ranking), then the picked probabilities renormalize to sum 1
(`norm_topk_prob`) — via `luminal_nn::MoETopK`, whose expert weights
are host-stacked at combine ([E, 2·I, H] gate;up and [E, H, I] down)
and fetched by coordinate-form gathers. The router computes in F32.
Cache: position-slots driver.

```bash
cargo run --release -p qwen3_moe -- --layers 1 --tokens 8
cargo test -p qwen3_moe
```
