# whisper — openai/whisper-tiny.en

100% of this checkpoint: the conv frontend (k3s1 + k3s2, exact erf
GELU), the stored sinusoidal encoder position table and the LEARNED
decoder position table (looked up by position-as-data — the old
legacy-tracker slice hack is dead), torch-style LayerNorm with bias
(eps 1e-5), the HF whisper projection-bias asymmetry (q/v/out biased,
k unbiased), plain 6×64 MHA with 1/8 on the scores, causal cached
decoder self-attention over the slot pool, UNCACHED cross-attention
recomputing K/V from the encoder output every step — encoder and
decoder in ONE graph, faithful to the original — and the tied output
head. Generation: forced [<|startoftranscript|>, <|notimestamps|>]
then greedy with whisper's suppression rule (special ids suppressed;
EOT allowed except on the first generated token). The host audio
pipeline (WAV → log-mel, librosa-matched) salvages verbatim.

```bash
cargo run --release -p whisper                    # bundled JFK sample
cargo run --release -p whisper -- --wav my.wav
cargo test -p whisper
```
