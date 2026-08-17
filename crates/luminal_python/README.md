# luminal_python

PyTorch `torch.compile` integration for Luminal.

## Frontends

Luminal exposes two compilation entry points over the same compiler pipeline.
Use the generic backend when normal PyTorch replacement and mutation semantics
must be preserved for every runtime tensor:

```python
import luminal
import torch

compiled = torch.compile(model, backend=luminal.backend)
```

Use Luminal's direct, inference-oriented frontend when preparing a model for
Luminal-managed execution:

```python
compiled = luminal.compile(model, example_input)
```

The returned artifact still supports generic calls. CUDA inference servers can
instead bind fixed storage explicitly and replay it without rescanning tensor
bindings or allocating outputs:

```python
bound = compiled.bind(input_buffer)

input_buffer.copy_(next_input)
outputs = bound.run()
```

Bound execution currently requires contiguous CUDA inputs and float32,
float16, or bfloat16 ordinary outputs. Binding is exclusive: once an artifact
has produced a bound executable, it no longer accepts generic calls.

Custom native backend factories can be configured with
`luminal.make_backend(factory)`. The older names `luminal_backend` and
`register_backend` remain compatibility aliases.

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
