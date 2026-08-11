# Qwen3 — the pure-logical conversion exemplar

Qwen3-4B authored entirely in logical ops on the native ladder: no
residency markers, no HLIR, no backend dependencies. The model is built
from `luminal_nn` constructs (`LlamaBlock` with QK-norm and the
concat-free pairing-matrix RoPE, paged scatter/gather KV cache with a
data-driven causal mask) and runs on the SSA reference runtime. This
crate is the pattern for re-seating the rest of the model zoo.

```bash
# Real weights (downloads Qwen/Qwen3-4B, repacks to a combined file):
cargo run --release -p qwen -- --layers 1 --tokens 8

# Offline, deterministic fake parameters, same anatomy:
cargo run --release -p qwen -- --random-weights
```

## How it runs

The decode graph is **step-invariant**: one token per step, a fixed
`(max_seq, kv_dim)` cache pair per layer, and every per-step quantity —
token id, query position, rope-table row, scatter slot, cache state —
arrives as data. The genetic implementation search therefore runs
**once**; generation is one `execute` per token, with cache state
flowing runtime-out → runtime-in between steps. Prefill feeds the
prompt through the same graph token by token.

## Honest limits (reference runtime)

- Everything stages as **f32** on the host, so all 36 layers (~16 GB of
  parameters) do not fit a small machine. `--layers` truncates the
  stack: real weights, real pipeline, demonstrative text. Full-depth,
  bf16-faithful execution returns when the GPU backends re-seat on the
  native ladder and bind this same logical graph.
- The checkpoint's bf16 is an on-disk staging format here; the in-graph
  dtype policy (bf16 matmuls, f32 norms) is runtime-binding business,
  not model text.

`cargo test -p qwen` runs the offline proofs: a tiny-dims decode loop
(determinism + cache write frontier) and a vocab-scale index-arithmetic
exactness probe for the embedding gather.
