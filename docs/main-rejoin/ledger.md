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
