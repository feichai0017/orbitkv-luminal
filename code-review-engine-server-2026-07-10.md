# Code review: continuous-batching engine + HTTP server + MoE single-path
**Scope**: commits `6c1438dc` (engine), `d60b991d` (HTTP server), `46c4aa45` (tiled-path deletion)
**Method**: 8 finder angles → 24 deduped candidates → independent verification (CONFIRMED / PLAUSIBLE / REFUTED)

## Top findings (ranked by severity)

### 1. `serve --tokens 0` bricks the server, and /health lies about it — CONFIRMED
`engine.rs:465` — `queue_request` does `.clamp(1, self.default_max_tokens)`; Rust's clamp
**panics when min > max**, i.e. whenever `--tokens 0`. The panic kills the engine thread on the
first request, every later request gets 500 "engine gone", and the readiness flag stays true so
`/health` keeps returning 200 on a dead server. Related: the same clamp **silently truncates** a
client's `max_tokens: 500` to the cap with no error (the adjacent prompt-length check correctly
returns an error instead).

### 2. Non-ASCII output reaches clients as mojibake — CONFIRMED
`server.rs:101` — every token is decoded alone (`decode_text(&[tok])`) and the non-stream path
concatenates those fragments while **discarding `Completion.text`**, the correct whole-sequence
decode the engine already computed. Multi-byte UTF-8 characters (emoji, CJK, accents) split
across o200k tokens decode to `U+FFFD` in **both** streaming and non-streaming responses.

### 3. The literal string `<|return|>` is delivered as content — CONFIRMED
`engine.rs:587` — tick() pushes the stop token into `TickSummary::tokens`/`predicted_tokens`
like any token; the server forwards it (decode with `skip_special_tokens=false`), so every
stop-terminated request ends with visible `<|return|>` text and `usage.completion_tokens`
over-counts by one.

### 4. Queued requests send zero HTTP bytes until first token — CONFIRMED
`server.rs:213` — the handler awaits the first engine event **before constructing any response**;
no headers, and SSE KeepAlive only starts after the response exists. A request queued behind
others is a silent connection for tens of seconds → client/proxy read timeouts → retries →
amplified load. Matches the anti-scaling shape in the c=8 benchmark.

### 5. Disconnected-but-queued requests are undetectable — CONFIRMED
`server.rs:57` — cancellation only triggers when a post-admission Token send fails. A client that
times out while pending still gets a full ~300 ms prefill + `n+max_tokens` slot reservation, and a
dead request at the FIFO head blocks live requests behind it. Under overload the server sheds no
load; it compounds.

### 6. `set_var` UB race at the new threaded call site — CONFIRMED
`engine.rs:267` — `unsafe { std::env::set_var(...) }` was sound at its old home (top of
single-threaded main) but `serve.rs` now runs `build()` on a spawned thread **while the main
thread constructs the tokio runtime** — setenv racing libc getenv is UB on glibc; symptom would
be a rare unexplainable startup crash. Fix: hoist the env default into the binaries' main()
before any threads, or make it a real `CudaRuntime` option.

### 7. Reclaimed KV slots are never scrubbed — PLAUSIBLE (defense-in-depth)
`engine.rs:77` — a reused slot holds the **previous request's K/V** until overwritten. Safe today
(the graph gathers from the scatter output, so data dependency guarantees overwrite-before-read,
and masks are exactly right) — but the old design read benign zeros on any mask/ordering bug;
the new one reads another client's cache. A future mask off-by-one becomes silent cross-request
KV leakage instead of harmless attention-over-zeros.

### 8. `page_table.tables` grows forever — CONFIRMED
`engine.rs:50` — seq ids are never recycled; `free_sequence` empties the inner Vec (retaining
capacity) but the entry stays. One leaked Vec + retained capacity per completed/cancelled
request, unbounded over a server's lifetime. The three request maps are cleaned correctly; only
this one leaks.

### 9. Harmony template forced on a raw completions endpoint — CONFIRMED (design)
`engine.rs:417` — `encode()` wraps **every** prompt in the harmony chat template with no bypass,
but `/v1/completions` is a raw endpoint by OpenAI convention. Already-formatted prompts get
double-wrapped with no error; `usage.prompt_tokens` reports the wrapped length (~+30 tokens),
skewing cross-server comparisons. Chat formatting belongs in the server layer (or a
`/v1/chat/completions` handler), not the scheduler.

### 10. Solo-vs-batched equivalence check was lost — CONFIRMED (coverage)
`main.rs:73` — the old demo printed the same sequence's ids solo (s=1) and batched
(A_DECODE_IDS / A_SUPER_IDS), a diffable isolation check. The new demo runs everything batched;
no test anywhere covers batched-vs-solo equivalence, so a mask/isolation regression that perturbs
tokens without crashing ships undetected.

## Confirmed but below the cap (cleanup tier)

- `serve --port 70000` binds port 4464 (`as u16` wraps); `flag()` silently ignores unparseable values
- Prompts tokenized 2–3+ times: once at queue (length only), again at admission, **again every tick** the queue-head waits at the gate — on the serial engine thread
- Four parallel seq-keyed collections (`active`/`requests`/`prompts`/`predicted_tokens`) with copy-pasted teardown in retire vs cancel; `prompts` only ever read for its length after admission → one `SeqState` map + `remove_seq()`
- `step()` clones all five Batch vectors into `set_data` (they could be moves); plus a redundant `to_vec()` of the logits (~0.8 MB/tick)
- `Event::Rejected` arms in both response builders are unreachable dead code (rejection is always the intercepted first event)
- `flag()` duplicated across the two binaries; usage-JSON built inline twice in server.rs
- `test_ref::assert_close` name-collides with `tests/utilities::assert_close` (different tolerance semantics); `from_bf16_bytes`/`bf16_bits_to_f32` are dead
- PageTable/build_batch forked from paged_llama (which keeps the weaker copy: no reclamation, -1e10 mask); harmony_prompt/argmax/EOS consts forked from gpt_oss
- `FusedMoE` allocs scratch per execute (36 alloc/free per tick); would block CUDA-graph capture of the tick later
- No guard/warning on `--max-prefill` past the deleted tiled crossover (silent TTFT cliff, decision lives only in a comment)
- EOS ids hard-coded vs derived from tokenizer config (low risk while the model repo is pinned)

## Refuted during verification

- "Dual reply-map handoff can drop tokens" — provably safe: single-threaded loop, insert-before-tick, handoff-before-forward
- "get_f32 copies the bucket-sized (~51 MB) allocation" — false: the DtoH copy is logical-size (`s×vocab×4`), recomputed per execute
