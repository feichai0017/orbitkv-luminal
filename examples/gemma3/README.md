# gemma3 — unsloth/gemma-3-4b-it (text tower)

100% of this checkpoint's features: 34 layers in the 5-local:1-global
alternation, sliding window 1024 on locals (mask-enforced), dual rope
thetas (10k local / 1M global with /8 linear position scaling), learned
per-head QK-norm before rope (eps 1e-6), attention scale 1/16 folded
into Q, sandwich norms with the Gemma (1+w) pattern pre-baked at
combine, GeGLU, tied embeddings with the sqrt(hidden) normalizer
in-graph over the unscaled table (the parked example pre-scaled the
table and duplicated an unscaled head — same math, one table here).
No logit softcaps (Gemma 3 dropped them). Cache: position-slots driver.

```bash
cargo run --release -p gemma3 -- --layers 1 --tokens 8
cargo test -p gemma3
```
