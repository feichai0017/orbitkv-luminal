---
name: aoti-debug
description: Debug AOTInductor (AOTI) errors including device mismatches, CUDA illegal memory access, segfaults, and wrong outputs when deploying compiled PyTorch models. Use when encountering errors with aoti_compile_and_package, aoti_load_package, or the deprecated aot_compile/aot_load APIs.
---

# AOTInductor Debugging

Debug errors when compiling and deploying PyTorch models with AOTInductor.

## First Step: Always Check Device and Shape Matching

**For ANY AOTI error (segfault, exception, crash, wrong output), check these first:**

1. **Compile device == Load device**: The model must be loaded on the same device type it was compiled on
2. **Input devices match**: Runtime inputs must be on the same device as the compiled model
3. **Input shapes match**: Runtime input shapes must match compilation shapes (or satisfy dynamic shape constraints)

```python
# Compilation -- note the device and shapes
model = MyModel().eval().cuda()
inp = torch.randn(2, 10, device="cuda")
pkg = torch._inductor.aoti_compile_and_package(model, (inp,))

# Loading -- device type MUST match compilation
loaded = torch._inductor.aoti_load_package(pkg)  # auto-detects device from package

# Inference -- device and shapes MUST match
out = loaded(torch.randn(2, 10, device="cuda"))  # same device, same shape
```

**AOTI requires compile and load to use the same device type.** Cross-device loading (compile on GPU, load on CPU) is NOT supported. Device index can differ (cuda:0 vs cuda:1).

## Current vs Deprecated API

### Current API (use this)
```python
torch._inductor.aoti_compile_and_package()  # compile
torch._inductor.aoti_load_package()          # load (auto-detects device)
```

### Deprecated API (migrate away)
```python
torch._export.aot_compile()  # deprecated
torch._export.aot_load()     # deprecated
```

The new API stores device metadata in the package, so `aoti_load_package()` automatically uses the correct device type.

## Common Error Patterns

### Device Mismatch Segfault

**Symptom**: Segfault, exception, or crash during load or execution.

**Example errors**:
- `The specified pointer resides on host memory and is not registered with any CUDA device`
- Crash during constant loading
- `Expected out tensor to have device cuda:0, but got cpu instead`

**Solution**: Ensure compile and load use the same device type.

### Input Device Mismatch at Runtime

**Symptom**: RuntimeError during model execution.

**Better debugging**: Run with `AOTI_RUNTIME_CHECK_INPUTS=1` for clear errors:
```bash
AOTI_RUNTIME_CHECK_INPUTS=1 python script.py
```

Produces actionable messages like:
```
Error: input_handles[0]: unmatched device type, expected: 0(cpu), but got: 1(cuda)
```

## Debugging CUDA Illegal Memory Access (IMA)

### Step 1: Sanity Checks

```bash
AOTI_RUNTIME_CHECK_INPUTS=1 python script.py          # validate inputs match compilation guards
TORCHINDUCTOR_NAN_ASSERTS=1 python script.py           # check for NaN before/after each kernel
```

Both flags take effect at **compile time** (codegen time).

### Step 2: Make IMA Deterministic

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 python script.py
```

- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` -- disables caching allocator (which allocates bigger buffers, masking IMA)
- `CUDA_LAUNCH_BLOCKING=1` -- forces synchronous kernel launches (pinpoints which kernel crashed)

Both take effect at **runtime**.

### Step 3: Identify the Problematic Kernel

```bash
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 python script.py
```

Prints kernels one by one at runtime. Combined with Step 2 flags, shows which kernel launched right before the error.

To inspect inputs to specific kernels:
```bash
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="kernel_name_1,kernel_name_2" \
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 python script.py
```

If inputs to a kernel are unexpected, trace back to the kernel that produced the bad input.

## Environment Variables Reference

| Variable | When | Purpose |
|---|---|---|
| `AOTI_RUNTIME_CHECK_INPUTS=1` | Compile time | Validate inputs match compilation guards |
| `TORCHINDUCTOR_NAN_ASSERTS=1` | Compile time | Check for NaN before/after kernels |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Runtime | Make IMA errors deterministic |
| `CUDA_LAUNCH_BLOCKING=1` | Runtime | Force synchronous kernel launches |
| `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3` | Compile time | Print kernels at runtime |
| `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="..."` | Compile time | Filter which kernels to print |
| `TORCH_LOGS="+inductor,output_code"` | Runtime | See PT2 internal logs |
| `TORCH_SHOW_CPP_STACKTRACES=1` | Runtime | Show C++ stack traces |

## Common Sources of Issues

- **Dynamic shapes**: Historically a common source of IMA errors. Pay special attention when using dynamic shape constraints.
- **Custom ops**: Especially C++ custom ops with dynamic shapes. The meta function may need to handle SymInt properly.
