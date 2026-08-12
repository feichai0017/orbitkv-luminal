# paged_llama3 — the page-table cache driver

Same checkpoint and model definition as `examples/llama3` (a library
dependency — one model, two cache drivers, per the cache-builder
ruling). What differs is the cache FORM: one slot pool shared by
multiple sequences through `luminal_nn::PageTable` (bump-allocated
per-sequence slot lists), batched fixed-width ticks, and visibility as
DATA — a (rows, slots) additive mask encoding causality,
cross-sequence isolation, and unwritten-slot exclusion in one
predicate (`luminal_nn::paged_attention_masked`).

The smoke test is the point: sequence A decoded alone through llama3's
position-slots driver produces exactly the same logits as A batched
with an unrelated sequence B in shared-pool ticks.

```bash
cargo run --release -p paged_llama3 -- --ticks 6
cargo test -p paged_llama3
```
