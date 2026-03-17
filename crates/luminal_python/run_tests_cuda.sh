#!/bin/bash
set -e

echo "=== Luminal Python Test Runner (CUDA Backend) ==="
echo ""

# Force clean rebuild of Rust extension
echo "Step 1: Cleaning previous builds..."
rm -rf rust/target/wheels rust/target/debug rust/target/release

# Rebuild in development mode (faster compilation)
echo "Step 2: Building Rust extension..."
uv run maturin develop --manifest-path rust/Cargo.toml --features cuda -r

# Run pytest with CUDA backend
echo "Step 3: Running pytest with CUDA backend..."
RUST_BACKTRACE=1 LUMINAL_BACKEND=cuda uv run pytest tests/test_qwen3.py::test_hf_qwen3_8b_full  -v -s --log-cli-level=INFO

echo ""
echo "=== Tests Complete ==="
