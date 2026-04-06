# Luminal

## Package Management

- Use `uv add`, `uv add --dev`, `uv remove` for Python dependencies (pyproject.toml is in `crates/luminal_python/`)
- Use `uv sync` to sync the Python environment
- Never use pip, pip-tools, poetry, or conda
- Never manually create or activate virtual environments — uv manages `.venv/` automatically
- Never generate requirements.txt

## Code Execution

- Always use `uv run` to execute Python tools: `uv run pytest`, `uv run pre-commit`, `uv run python`
- Use `cargo` directly for Rust: `cargo build`, `cargo test`, `cargo check`, `cargo clippy`
- Python project root is `crates/luminal_python/` — run `uv run` commands from there

## Building the Python Package (Maturin)

- After modifying `.rs` files that affect the Python bridge, rebuild with: `maturin develop --release`
- Maturin config is in `crates/luminal_python/pyproject.toml` under `[tool.maturin]`

## Pre-commit

- Run with: `uv run pre-commit run --all-files`
- Hooks configured: ruff-check, ruff-format (Python), cargo-fmt, cargo-clippy (Rust)
- Manual-stage hooks (cargo-clippy-metal, cargo-clippy-cuda-lite) run with `--hook-stage manual`

## Testing

- **Rust tests**: `cargo test -p <crate_name>`
- **Python tests**: `cd crates/luminal_python && uv run pytest`
  - `./run_test.sh` — native backend
  - `./run_tests_cuda.sh` — CUDA backend
- See `crates/luminal_python/CLAUDE.md` for Python test patterns and conventions
