# main-rejoin ledger

One row per `main` post-split commit walked, in chronological order. The walk
lands each main commit onto `logical-ssa-project` at whatever fidelity the
branch's own decisions allow, and records — here — the ones that cannot be
carried as code, so nothing is silently dropped.

Dispositions:

- **FILE-LEVEL** — main's diff applied to the same paths, unchanged. Used for
  areas the branch parks rather than builds (`crates/luminal_python`,
  `crates/luminal_metal`, `ci/`, `spec.md`) and for the
  `crates/luminal_cuda_lite_hlir/` park, which TRACKS main's
  `crates/luminal_cuda_lite/` path-rewritten so the target CL must reach keeps
  moving.
- **RE-EXPRESSED** — the intent landed, spelled in the branch's vocabulary
  (`IntExpr`, `legacy_tracker_ref()`/`legacy_tracker_mut()`/`dims()`,
  coordinate-form gather/scatter, the pad family). A branch rename or decision
  is never reverted to make a main hunk apply.
- **INTENT-ONLY** — no code landed because the code main patched does not exist
  on the branch. The requirement is written out in the last column so it can be
  satisfied later against the branch's own machinery.
- **DROPPED** — deliberately not carried at all.

| main sha | PR | title | disposition | where it landed | intent to carry |
| --- | --- | --- | --- | --- | --- |
| `cd0aa58f` | #384 | translate_sdpa: close the SDPA surface — precision, masks, GQA, dynamic shapes | FILE-LEVEL | PR #445 (branch `e2f5cd0a`) | — |
| `aa5664bb` | #385 | luminal_python: honor the in-place mutation contract (write-back outputs) | FILE-LEVEL | PR #448 (branch `16fbb5bd`) | — |
| `be3e2fe5` | #387 | translator: robustness fixes — dtype promotion, rank-extending expand, norm opmath | RE-EXPRESSED (movement / unary) | PR #448 (branch `5817a012`) | — |
| `7423ca37` | #391 | compile search progress UI | RE-EXPRESSED (`search_log` + Start/Faster/Slower) | branch `merge/main-391-search-ui` (this commit) | see **#391 progress UI** below |
| `7d2817fa` | — | luminal_python: search_iterations pass through more places | FILE-LEVEL | branch `7b25c63d` (`merge/main-7d2817fa-search-iterations`) | parked crate: re-point `search_iterations` at `ImplementationSearchOptions` when luminal_python is re-attached to the recorder |
| `bea18ecf` | #389 | Sdpa gqa fixes | FILE-LEVEL | branch `201aa15b` (`merge/main-389-sdpa-gqa`) | parked crate + non-gating `ci/`: the loosened gemma / gemma4_moe TPOT numbers are main's HLIR cuda_lite numbers and must be re-baselined against CL A100 draws before they gate anything here |
| `499d0779` | #386 | Search: early-stop candidate profiling against the best-so-far metric | MIXED — FILE-LEVEL (park + metal) / INTENT-ONLY (core) | this commit, on `merge/main-386-early-stop` | see **#386 early-stop profiling** below |

## #391 progress UI — re-expressed in `src/implementation_search.rs`

Main's diff patches the LLIR compile-search loop in `src/graph.rs` (main's
`Graph::search`, ~lines 2380–2660). That region does not exist on this branch:
the old HLIR search was deleted with `src/hlir.rs` / `src/op.rs`, and search now
lives in `src/implementation_search.rs` (a genetic search over the e-graph) plus
`src/extractor.rs`. Neither prints any live progress today, so there is nothing
to patch and nothing to re-spell — only a behaviour to record.

**What main prints, and when.** All of it is gated on one option:
`CompileOptions::search_log: bool` (default `true`), set by the builder
`.search_log(enabled)` and read through `log_channel_enabled(self.search_log,
"SEARCH_LOG")`, so the env var can override the programmatic setting. With it
off, the search prints nothing.

1. **`Start`** — once, before the loop, on the initial (baseline) genome:
   `   {:>6} {display}` with `Start` in bold cyan, followed by the progress bars
   (`render_bars(n_graphs, search_limit, bucket_progress)`) and an explicit
   stdout flush. This commit is what renamed that label from `Search` to
   `Start`: the first line reports the *baseline*, not a search result.
2. **`Faster`** — after any profiled candidate that beats the best-so-far:
   `   {:>6} {display_metric}` with `Faster` in bold green, carrying the new
   best metric. A `Faster` line is *permanent*: it is appended and the bars are
   redrawn beneath it, so a run leaves behind one line per improvement — the
   improvement history is the scrollback.
3. **`Slower x{n}`** — after any profiled candidate that does not beat the best:
   `   {:>6} x{n}` with `Slower` in bold yellow, where `n` is
   `slower_since_faster`, the count of consecutive non-improving candidates
   since the last improvement (reset to 0 on every `Faster`). A `Slower` line is
   *transient*: exactly one is ever on screen, replaced in place by the next
   `Slower`, and left to be overwritten/pushed by the next `Faster`.

**The cursor bookkeeping that makes 2 and 3 work.** Before printing, the cursor
walks up from the last progress bar to the first (`for _ in 1..n_bar_lines {
print!("\x1b[1A") }`); if a transient `Slower` line is currently visible *and*
this result is also slower, it walks up one more line so the new `Slower`
overwrites the old one; then `\r\x1b[2K` clears the line, the message is
printed, and `slower_line_visible = !new_best` records whether a transient line
now sits above the bars. The bars are re-rendered afterwards. Two bits of state
carry all of it: `slower_since_faster: usize` and `slower_line_visible: bool`.

**What landed here (ruling 5, 2026-09-02: match main).**
`ImplementationSearchOptions` gains `search_log: bool`, default `true` — main's
default, not the quieter one this row originally proposed — with the builder
`.search_log(enabled)` and the same env override, through a local
`log_channel_enabled(self.search_log, "SEARCH_LOG")` copied from main's
`src/egglog_utils/mod.rs` (this branch had no log-channel helper at all, so the
`LUMINAL_LOG=1` force-on and the `1/true/yes/on` flag parsing come across with
it). `search_implementations_with_runtime` builds a `SearchProgress` writer when
the channel is on, and reports on each PROFILED candidate (fingerprint-cache
hits are not candidates that ran, and can never improve the best): the first one
→ `Start` with the baseline metric; afterwards `nanos < *best_nanos` → a
permanent `Faster` line, otherwise the transient `Slower x{n}` counter, reset by
every improvement. Output goes to **stderr**, not main's stdout, so it never
contaminates a caller's data stream — and through a `CaptureAwareStderr`
adapter whose `Write::write` routes the bytes through `eprint!` rather than a
raw `Stderr` handle, because libtest's output capture intercepts the macro and
not the handle. Real runs print exactly as before; test runs are silent unless
`--nocapture`.

Two deliberate divergences from main, both console-only:

- **No cursor arithmetic.** Main walks the cursor up over its progress bars
  (`\x1b[1A` per bar row) before printing. This branch draws no bars, so that is
  dropped; the transient `Slower` line is written WITHOUT a newline and every
  later line begins by clearing it in place (`\r\x1b[2K`). A `Faster` line
  therefore replaces the pending `Slower` line instead of being appended below
  it, and `finish()` clears a still-pending one at the end of the search.
- **The harness stays quiet.** The DEFAULT matches main (`true`), and the suites
  are quiet anyway because the writer goes through the capture-aware macro; on
  top of that, `harness_search_options()` (`src/test_support.rs`) sets
  `search_log: false`, and so do the ten other struct-literal call sites the new
  field made exhaustive-literal-incomplete (all under `#[cfg(test)]`), so those
  searches do not even build a reporter. Nothing here rests on main's tests being
  noisy — main printed through `println!`, which libtest captures, so main's
  tests were silent too.

Unit test: `implementation_search::progress_tests::
progress_prints_start_once_faster_per_improvement_and_a_resetting_slower_counter`
drives the reporter over an in-memory writer and pins `Start` exactly once (with
the baseline metric), one `Faster` carrying the new best, the `x1 → x2` climb,
the reset back to `x1` after an improvement, and the five `\r\x1b[2K` in-place
rewrites. It strips ANSI so it passes whether or not `colored` colorizes.

## #386 early-stop profiling — what landed, and what is owed

Main's commit is one idea spread over six files: an opt-in
`CompileOptions::early_stop_factor(f64)` threads `Option<(best_metric, factor)>`
through `Runtime::profile` / `Runtime::profile_with_bucket_context`; each device
runtime, after every *timed* trial, compares the candidate's running MEAN trial
time against `best * factor` (the shared predicate `luminal::op::
early_stop_exceeded`) and breaks out, returning the partial mean. Selection is
explicitly unchanged: the truncated metric is still ranked, so early stop only
shortens the timing of candidates already out of contention. The initial genome
passes `None` because it *is* the baseline, and CUDA's warmup bail is left
untouched so a slow-warmup / fast-steady candidate is not disqualified.

**Landed FILE-LEVEL (parked, does not build):**

- `crates/luminal_cuda_lite_hlir/src/runtime.rs` — main's
  `crates/luminal_cuda_lite/` hunks with the path rewritten, per the ruling that
  the hlir park TRACKS main so the target CL must reach keeps moving. Applied
  cleanly against the park's existing branch drift (`IntExpr`, `alias_state` →
  `alloc_state_buffer` + `bind_*_buffer`, no `mask_events`); only hunk offsets
  moved.
- `crates/luminal_metal/src/runtime.rs` — file-level, per the ruling that metal
  becomes a runtime like the others and is ported later.

Both now reference `luminal::op::early_stop_exceeded`, which does not exist on
this branch (`src/op.rs` is deleted). Neither crate is a workspace member, so
nothing fails to build; the dangling reference is the standing cost of parking
these files at main's spelling, and it resolves when each crate is ported.

**Not landed (no counterpart on this branch):**

- `src/op.rs` (+41: the `Runtime::profile` / `profile_with_bucket_context`
  signature change, the `early_stop_exceeded` predicate, and its
  `#[cfg(test)] mod early_stop_tests`) and `src/hlir.rs` (+1: the
  `ReferenceRuntime` impl) — both files are deleted on this branch.
- `examples/llama/src/main.rs` (opts in at `.early_stop_factor(2.0)`) — this
  branch has no `examples/llama`; the zoo is `examples/llama3`,
  `examples/paged_llama3`, … and none of them use `CompileOptions`.
- `src/graph.rs` (+109: the `CompileOptions::early_stop_factor` builder, passing
  `None` for the initial genome and `Some((best, factor))` thereafter, and the
  regression test `search_passes_best_so_far_to_profile_early_stop`) — main's
  `src/graph.rs` is the HLIR `CompileOptions` / `Graph::search` file; this
  branch's `src/graph.rs` is the LogicalGraph recorder, with no
  `CompileOptions`, no search loop, no `trials` and no `timeout`.

**Why the re-expression was NOT done here.** The brief allows re-expressing the
idea against `src/implementation_search.rs` only if it is small and mechanical.
It is not — it needs two decisions that are not the implementer's to make:

1. **Which metric.** Main's guarantee ("remaining trials cannot change the
   outcome") is an argument about a running *mean*, which only rises as trials
   accumulate. This branch's `ReferenceProfiler` (`crates/luminal_reference/src/
   search.rs`) ranks by the running *minimum* — `best_nanos.min(...)` over
   `trials` timed executes after one warmup — and a running minimum can still
   fall on a later trial. Early-stopping on a minimum is therefore a heuristic
   truncation of a metric bounded only from below: it can promote a candidate
   whose truncated min is worse than its true min. Adopting a mean instead
   changes what `best_nanos` means everywhere that reads it
   (`SearchOutcome::best_nanos`, search logs, anything baselined on it).
2. **Where the cutoff hooks in.** The trial loop lives inside the profiler, not
   inside the selection loop, so the cutoff has to cross `PlanProfiler::profile`
   — the deliberately thin runtime-owned-execution seam (ruling 2026-08-17,
   "every runtime owns its execution", including how its candidates are timed).
   That is either a fifth positional argument on a public trait method that
   already has four, or an options/context struct — a seam decision, not a
   mechanical edit.

**The requirement, for whenever it is taken up.** (a) `early_stop_factor:
Option<f64>` on `ImplementationSearchOptions`, default `None` (off), with main's
`>= 1.0` assertion; behaviour identical to today when unset. (b) The selection
loop in `src/implementation_search.rs` passes `None` while `best.is_none()` and
`Some((best_nanos, factor))` afterwards — the loop already holds exactly that
`best: Option<(nanos, genome, plan)>`. (c) `ReferenceProfiler::profile` checks
after each timed execute and returns the partial metric; `StaticProfiler`
ignores it (it never runs trials). (d) The predicate re-expresses as a free
function over `u128` nanos rather than main's `Duration`, and main's
`test_early_stop_exceeded` moves essentially verbatim once retyped. (e) Main's
`search_passes_best_so_far_to_profile_early_stop` regression test cannot move —
it is built on `Runtime` / `LLIRGraph` / `compile_with_rng` / `CompileOptions` —
but its intent re-expresses as a recording `PlanProfiler` that asserts `None`
for the first candidate and `Some((best, factor))` for every later one under a
fixed seed.

**And the precondition that decides whether it is worth anything.** The only
profiler on this branch that actually executes candidates is the host
`ReferenceProfiler`, at `trials: 3` — a maximum saving of two executes per
losing candidate — and the search already suppresses duplicate work with the
plan-fingerprint cache. CL does not time candidates at all: `crates/
luminal_cuda_lite/src/runtime.rs` searches with `StaticProfiler`, ranking by the
heuristic bytes-moved cost with no execution. So the CUDA half of main's commit
— the half where this feature pays what the PR claims — has no landing target
until CL grows a real device `PlanProfiler`. That is the follow-on this row is
waiting on, and it is a much larger question than the cutoff itself.
