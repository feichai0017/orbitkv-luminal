# Llama-3-8B-Instruct — pure-logical zoo conversion

Exactly `NousResearch/Meta-Llama-3-8B-Instruct` (per-HF-model fidelity
ruling): 32 pre-RMSNorm layers with learned norm weights at eps 1e-5,
GQA 32/8 at head_dim 128, split-half RoPE at theta 500 000 (this
checkpoint's `rope_scaling` is null — the 3.1 frequency ramp lives in
the fp8 example), SwiGLU, UNTIED lm_head. Projections are unfused (the
old [v;q;k]/[gate;up] byte-fusion was GEMV batching, not architecture)
so the combined checkpoint keeps original HF tensor names.

Cache form: the position-slots driver over `luminal_nn::KvCachePool`
(cache structure is model definition; other examples exercise other
drivers — see paged_llama3). Decode is step-invariant: one search,
one execute per token; sampling (greedy + repetition penalty 1.05) is
host-side.

```bash
cargo run --release -p llama3 -- --layers 1 --tokens 8
cargo run --release -p llama3 -- --random-weights   # offline
cargo test -p llama3                                 # offline proofs
```

The f32 reference runtime cannot hold all 32 layers (~26 GB f32);
`--layers` truncates honestly. Full depth returns with the backend
re-seat.
