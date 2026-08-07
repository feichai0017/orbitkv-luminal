# luminal_python

PyTorch `torch.compile` integration for Luminal.

## Hugging Face paged KV cache

`luminal.transformers_cache.LuminalPagedCache` is a Transformers `Cache`
implementation for Luminal's page-size-one FlashInfer path. It keeps each
layer's K/V allocation in physical `[max_tokens, kv_heads * head_dim]` NHD
storage while returning the normal `[batch, kv_heads, context, head_dim]`
logical tensors to Hugging Face attention code. PT2 therefore sees ordinary,
semantic `index_put` and `index_select` operations; the CUDA egglog rules can
replace the resulting attention island with `FlashInferAttention`.

The current contract is deliberately narrow: batch size one, append-only full
attention, no beam reordering, and no sliding-window layers. Construct it
before calling the compiled model:

```python
from luminal.transformers_cache import LuminalPagedCache

cache = LuminalPagedCache(
    model.config,
    max_cache_len=prompt_tokens + max_new_tokens,
    dtype=torch.bfloat16,
    device="cuda",
)
outputs = compiled_model(
    input_ids=input_ids,
    past_key_values=cache,
    use_cache=True,
)
```

The cache's compact slot state is `int32`, matching FlashInfer's page-table
ABI. Explicit `int64` casts exist only around PyTorch indexed operations.

## CUDA Tests

The Python CUDA CI job builds the Rust extension with the CUDA feature and runs
the non-slow pytest suite:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s -m "not slow"
```

The slow tests are explicit opt-in. They include large/pretrained model tests,
full-width architecture compiles, Whisper end-to-end cases, and other cases that
can take a long time or need a large GPU / Hugging Face cache.

Run the full Python CUDA suite, including slow tests:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s
```

Run only the slow Python CUDA tests:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s -m slow
```

The helper script follows the same convention:

```bash
cd crates/luminal_python
./run_tests_cuda.sh              # non-slow CUDA suite
./run_tests_cuda.sh --slow-only  # only slow CUDA tests
./run_tests_cuda.sh --include-slow
```

The GitHub/Modal entrypoint uses the same marker split:

```bash
cd crates/luminal_python
modal run modal_pytest_runner.py --gpu A100 --timeout 7200 tests/ -v -s -m "not slow"
modal run modal_pytest_runner.py --gpu A100 --timeout 7200 tests/ -v -s
```
