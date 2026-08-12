# llama3_1_fp8 — nvidia/Llama-3.1-8B-Instruct-FP8

The fp8 zoo example. Quantization is MODEL DEFINITION: every layer
linear is an E4M3FN tensor with two static F32 scales
(`luminal_nn::Fp8Linear`), the weights stage and store as native fp8
code buffers, and the quantize step (`x/input_scale → cast F8E4M3`,
round-to-nearest-even saturating at ±448) is spelled in model text.
The widening casts are the explicit dequant reads; the f32 matmul
equals the fp8 GEMM with f32 accumulation.

Fidelity deltas over the parked example: the llama3-type ROPE
frequency ramp (factor 8.0, low_freq 1.0, high_freq 4.0,
original_max 8192) that Llama-3.1 specifies and the old code omitted;
UNFUSED projections with original HF tensor names (and per-projection
scales — no shared-max requantization).

The fp8 conversion semantics are pinned exhaustively against the
checkpoint codec (all 256 codes + a dense encode sweep) in luminal's
`f8e4m3_semantics` tests, and `Fp8Linear` against a scalar reference
in luminal_nn.

```bash
cargo run --release -p llama3_1_fp8 -- --layers 1 --tokens 8
cargo test -p llama3_1_fp8
```
