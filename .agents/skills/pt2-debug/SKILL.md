---
name: pt2-debug
description: Debug torch.compile failures, graph breaks, recompilation issues, accuracy mismatches, and Triton kernel errors. Use when encountering BackendCompilerFailed exceptions, torch.compile errors, recompilation warnings, or numerical accuracy issues with compiled PyTorch models.
---

# PyTorch 2 Compile Debugging

Debug `torch.compile`, Dynamo, Inductor, and AOTAutograd failures when using PyTorch as a library.

## Diagnostic Environment Variables

Pick the right diagnostic based on the error:

| Command | When to use |
|---|---|
| `TORCH_LOGS="+dynamo,graph_breaks,recompiles" python script.py` | Quick overview of what's going wrong |
| `TORCH_COMPILE_DEBUG=1 python script.py` | Full debug artifacts (FX graphs, Inductor IR, generated code) in `torch_compile_debug/` |
| `TORCH_LOGS="output_code" python script.py` | See the generated Triton/C++ kernel code |
| `TORCH_TRACE=/path/to/trace python script.py` | Structured trace (parse with `tlparse`) |
| `TORCHINDUCTOR_COMPILE_THREADS=1 python script.py` | Single-threaded compilation for pdb debugging |

## Error Triage

Classify the failure and jump to the right section:

| Error Pattern | Category |
|---|---|
| `Unsupported: ...` or `graph break` in logs | [Graph Breaks](#graph-breaks) |
| `BackendCompilerFailed` | [Backend Failures](#backend-compiler-failures) |
| `RecompileError` or `cache_size_limit` | [Recompilation](#recompilation-issues) |
| Accuracy mismatch / wrong numerical output | [Accuracy](#accuracy-issues) |
| `InternalTorchDynamoError` | [Internal Errors](#internal-dynamo-errors) |
| Segfault or CUDA IMA | [Runtime Crashes](#runtime-crashes) |
| Triton assertion / index out of bounds | [Triton Failures](#triton-kernel-failures) |

## Graph Breaks

Graph breaks split the compiled graph into smaller subgraphs, causing performance regressions.

**Diagnose:**
```bash
TORCH_LOGS="graph_breaks" python script.py
```

**Common causes:**
- Data-dependent control flow
- Unsupported Python builtins
- In-place ops on inputs, unsupported dtypes
- Calls to non-traceable functions

**Fix approaches:**
1. Read the graph break message to identify the unsupported operation
2. Check for a decomposition or supported alternative
3. Consider `torch._dynamo.allow_in_graph` or restructure user code

## Backend Compiler Failures

`BackendCompilerFailed` means Inductor crashed during compilation.

**Diagnose with the minifier:**
```bash
# Generate minifier launcher
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=2 python script.py

# Run the minifier to get minimal failing graph
python minifier_launcher.py minify

# Run the minimized reproduction
python minifier_launcher.py run
```

**Then inspect:**
```bash
TORCH_COMPILE_DEBUG=1 python script.py  # FX graphs in torch_compile_debug/
```

## Recompilation Issues

Excessive recompilation from guards that are too specific, causing cache misses.

**Diagnose:**
```bash
TORCH_LOGS="recompiles,recompiles_verbose,guards" python script.py
```

**Key config:**
```python
torch._dynamo.config.recompile_limit  # default: 8
torch._dynamo.config.fail_on_recompile_limit_hit = True  # hard error on limit
```

**Common causes:**
- Changing tensor shapes without marking them dynamic
- Python scalar values that change between calls
- Global state mutations between calls

**Fix:** Read the recompilation reason from logs, identify the failing guard, then either:
- Mark dimensions as dynamic: `torch._dynamo.mark_dynamic(tensor, dim)`
- Fix the source of guard instability

## Accuracy Issues

Compiled model produces different numerical results than eager mode.

**Diagnose:**
```bash
# Compares compiled vs eager with fp64 reference, dumps repro on failure
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4 python script.py
```

**Fix approach:**
1. Get minimal failing graph from the minifier
2. Compare eager vs compiled output at fp64 precision
3. Binary search through ops to find the diverging operation
4. Check for known issues: reduction order, fused kernels, dtype promotions

## Internal Dynamo Errors

`InternalTorchDynamoError` indicates a bug in Dynamo.

**Diagnose:**
```bash
TORCHDYNAMO_VERBOSE=1 python script.py
# or equivalently:
TORCH_LOGS="+dynamo" python script.py
```

**Debug interactively:**
```bash
TORCHINDUCTOR_COMPILE_THREADS=1 python script.py  # then attach pdb
```

## Runtime Crashes

Segfaults and CUDA illegal memory access during execution of compiled code.

**Make crash deterministic:**
```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 python script.py
```

**Add NaN checks to find the first bad kernel:**
```bash
TORCHINDUCTOR_NAN_ASSERTS=1 python script.py
```

**Inductor sync debugging:**
```python
torch._inductor.config.triton.debug_sync_kernel = True  # sync after every kernel
torch._inductor.config.triton.debug_sync_graph = True   # sync before/after graph
```

**Fix approach:**
1. Make deterministic with `PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1`
2. Check input shapes, devices, dtypes
3. Inspect generated kernel code with `TORCH_LOGS="output_code"`
4. Use `TORCHINDUCTOR_NAN_ASSERTS=1` to find the first kernel producing bad values
5. Dynamic shapes are historically a common source of IMA

## Triton Kernel Failures

Triton assertion failures or index-out-of-bounds in generated kernels.

**Diagnose:**
```bash
TORCH_LOGS="output_code,schedule" python script.py
```

**Fix approach:**
1. Get the generated Triton kernel from `output_code` logs
2. Check index computations for off-by-one or wrong stride calculations
3. Check IR with `TORCH_COMPILE_DEBUG=1` to trace back to the FX op
4. Check if fusion decisions created invalid index combinations

## Distinguish Trace-Time vs Runtime

Many bugs come from confusing these:
- **Trace-time**: Inside Dynamo's symbolic interpreter. Function calls may be constant-folded.
- **Runtime**: Real tensors, real Python calls.

When debugging, add `print()` directly in source files rather than monkey-patching -- dispatch chains make monkey-patching unreliable.

## Using the Minifier

The minifier reduces a failing graph to the smallest reproduction:

```bash
# For compilation failures (level 2)
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=2 python script.py
python minifier_launcher.py minify
python minifier_launcher.py run

# For accuracy failures (level 4)
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4 python script.py
```
