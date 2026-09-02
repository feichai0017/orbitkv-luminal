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
| `7423ca37` | #391 | compile search progress UI | INTENT-ONLY | nothing landed — this row | see **#391 progress UI** below |

## #391 progress UI — the requirement, for `src/implementation_search.rs`

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

**What satisfying this on the branch would mean.** `search_implementations` /
`search_implementations_with_runtime` would need (a) a log-enable option on
`ImplementationSearchOptions` (default off is the safer branch default — the
branch's search is called from tests and examples that expect quiet output), and
(b) the same three-state reporting driven by the selection loop's existing
bookkeeping: the initial baseline candidate → `Start`; `nanos < *best_nanos`
(the loop already computes exactly this against `best: Option<(nanos, genome,
plan)>`) → `Faster` plus a permanent line; otherwise → a transient
`Slower x{n}`. There are no progress *bars* on the branch to move the cursor
relative to, so a port must either add a bar row (generations × generation_size
is a known total, so a bar is well defined) or drop the cursor arithmetic and
print plain lines. Nothing about correctness depends on this: it is console UX
only, with no tests or goldens attached on main.
