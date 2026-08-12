# gemma4_moe — google/gemma-4-26B-A4B (text tower)

The zoo's most heterogeneous anatomy, at 100% fidelity: 30 layers in
the 5-sliding:1-full alternation where the roles differ STRUCTURALLY —
sliding layers run head_dim 256 / 8 KV heads / theta 10k / full rotary
with their own v_proj; full layers run head_dim 512 / 2 KV heads /
theta 1M / PARTIAL rotary 0.25 (zero-angle lanes pass through the
pairing form) with V taken from the K projection. V gets a weightless
per-head RMS norm everywhere; attention scale is 1.0 (none). Seven
learned norms per layer wrap a PARALLEL dense+MoE FF stage (router on
the raw residual — std-normed × router.scale × 1/√hidden — experts on
the pre_ff_2-normed stream, top-8 renormalized × per_expert_scale,
gemma-gelu gating), and the whole residual stream multiplies a learned
per-layer scalar. Logits softcap at 30. Tied embeddings with the
√hidden normalizer in-graph. Cache: the HETEROGENEOUS slot pool
(per-layer kv widths — `KvCachePool::new_heterogeneous`).

```bash
cargo run --release -p gemma4_moe -- --layers 1 --tokens 8
cargo test -p gemma4_moe
```
