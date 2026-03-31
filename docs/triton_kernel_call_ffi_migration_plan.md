# Plan: Register `triton_kernel_call` as FFI with `kCmdBufferCompatible`

## Problem Statement

TE's MoE permutation uses ~40 Triton custom calls per training step (4 decoder
layers). Each call is dispatched individually because `triton_kernel_call` is
registered as a **legacy custom call** (not FFI), and XLA's command buffer
conversion pass only captures custom calls that have an **FFI handler with
`kCmdBufferCompatible` trait**. This host dispatch overhead (~150 µs/call)
accounts for ~6 ms/step — negating the GPU kernel speedup from Triton fusion.

## Current Architecture

### Registration Chain

```
jaxlib/gpu_triton.py
  └── xla_client.register_custom_call_target(
          "triton_kernel_call",
          _cuda_triton.get_custom_call(),    # PyCapsule from nanobind
          platform='CUDA')

jaxlib/gpu/triton.cc
  └── m.def("get_custom_call",
            [] { return EncapsulateFunction(&TritonKernelCall); });

jaxlib/gpu/triton_kernels.h
  └── void TritonKernelCall(gpuStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len,
                            XlaCustomCallStatus* status);
```

### Key Files

| File | Role |
|------|------|
| `jaxlib/gpu_triton.py` | Python registration (legacy, no traits) |
| `jaxlib/gpu/triton.cc` | nanobind module exposing `TritonKernelCall` capsule |
| `jaxlib/gpu/triton_kernels.h` | `TritonKernelCall` declaration |
| `jaxlib/gpu/triton_kernels.cc` | `TritonKernelCall` implementation (proto decode → kernel launch) |
| `jaxlib/gpu/gpu_kernels.cc` | `XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM` (for external HLO runners) |

### Why It Fails Command Buffer Capture

XLA's `command_buffer_conversion_pass.cc` uses this check for custom calls:

```cpp
bool IsConvertible(const CustomCallThunk& custom_call_thunk, ...) {
  auto registration = ffi::FindHandler(target_name, "gpu");
  return registration.ok()
             ? ffi::IsCommandBufferCompatible(registration->metadata)
             : false;
}
```

Since `triton_kernel_call` has NO FFI handler registered (only a legacy capsule),
`FindHandler` returns an error → `false` → not capturable.

## `triton_kernel_call` vs `__gpu$xla.gpu.triton`

These are **different custom call targets** with different pipelines:

| Aspect | `triton_kernel_call` | `__gpu$xla.gpu.triton` |
|--------|---------------------|------------------------|
| **Used by** | TE (via `jax.ffi.ffi_lowering`), legacy Pallas path | Current Pallas GPU lowering |
| **API version** | `API_VERSION_STATUS_RETURNING` (legacy) | `API_VERSION_TYPED_FFI` (v4) |
| **Backend config** | zlib-compressed protobuf (opaque) | MLIR-structured (Triton IR, grid, etc.) |
| **Kernel compilation** | Pre-compiled PTX/HSACO packed in proto | XLA compiler compiles inline |
| **Registration** | `register_custom_call_target` (PyCapsule) | XLA built-in (`thunk_emitter.cc`) |
| **Command buffer** | Not compatible | Potentially compatible (XLA-internal) |

The JAX issue [#27988](https://github.com/jax-ml/jax/issues/27988) targets
`__gpu$xla.gpu.triton` (Pallas), **not** `triton_kernel_call`. But the
underlying problem — Triton custom calls not being command-buffer-compatible —
is the same for both paths.

JAX already has TODOs to migrate:
- `jax/_src/pallas/triton/pallas_call_registration.py` line 191:
  `# TODO(b/392558289): Migrate to jax.ffi.`

## Proposed Solutions (Ordered by Effort)

### Option A: FFI Shim Handler (Low-Medium Effort)

Register a **parallel FFI handler** under the same name `triton_kernel_call`
that wraps the existing `TritonKernelCall` implementation. This adds command
buffer compatibility without changing the lowering or serialization format.

**Changes required:**

1. **`jaxlib/gpu/triton_kernels.cc`** — Add FFI handler:

```cpp
#include "xla/ffi/api/ffi.h"

static xla::ffi::Error TritonKernelCallFFI(
    xla::ffi::PlatformStream<gpuStream_t> stream,
    xla::ffi::RemainingArgs args,
    xla::ffi::RemainingRets rets,
    xla::ffi::Span<const uint8_t> backend_config) {
  // Decode the same zlib-compressed protobuf from backend_config
  // Reuse existing kernel launch logic from TritonKernelCall
  XlaCustomCallStatus status;
  // ... adapt buffers from args/rets to void** ...
  TritonKernelCall(stream.value(), buffers,
                   reinterpret_cast<const char*>(backend_config.data()),
                   backend_config.size(), &status);
  if (!status.ok()) return xla::ffi::Error(...);
  return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(kTritonKernelCallHandler, TritonKernelCallFFI,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<gpuStream_t>>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<xla::ffi::Span<const uint8_t>>("backend_config"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(),
    "triton_kernel_call", "CUDA", kTritonKernelCallHandler);
```

2. **`jaxlib/gpu_triton.py`** — Keep the legacy registration (backward compat)
   but add the FFI handler import to ensure it's linked.

3. **TE lowering** — Update `triton_call_lowering` in
   `transformer_engine/jax/triton_extensions/utils.py` to use
   `api_version=4` (FFI) instead of `api_version=2` (legacy).

**Risks:**
- `RemainingArgs` / `RemainingRets` FFI binding must handle variable buffer
  counts (different Triton kernels have different arities). This is supported
  by XLA FFI but needs careful implementation.
- The `backend_config` encoding must be compatible: FFI uses typed attributes
  vs legacy's raw opaque bytes. May need to pass as a byte-span attribute.
- Autotuning during `TritonKernelCall` execution involves trial kernel launches
  that may not be CUDA-graph-safe on first invocation. After autotuning
  completes (warmup steps), subsequent launches are deterministic and safe.

### Option B: Dual Registration (Legacy + FFI Metadata) (Low Effort, Partial)

Register a **no-op FFI handler** with `kCmdBufferCompatible` under
`triton_kernel_call`, alongside the existing legacy registration. XLA's
`IsConvertible` would find the FFI handler and return `true`, while the actual
execution still uses the legacy `CustomCallThunk`.

**Caveat:** This is a **hack**. The command buffer conversion pass assumes that
if `IsConvertible` returns true, the thunk can be traced via CUDA graph capture.
The legacy `CustomCallThunk` would then be captured inside a CUDA graph. Whether
this works depends on whether `TritonKernelCall` is graph-safe (no host sync,
no dynamic allocation). After autotuning, it likely is — but this needs testing.

### Option C: Full FFI Migration (High Effort, Clean)

Migrate `TritonKernelCall` entirely to the FFI API:

1. Redesign the function signature to use `ffi::Ffi::Bind()` with typed args
2. Use `API_VERSION_TYPED_FFI` in all lowerings
3. Pass kernel metadata as FFI attributes instead of opaque bytes
4. Remove the legacy registration

This aligns with JAX's stated direction (TODO `b/392558289`) but is a
significant refactor across jaxlib's Triton stack.

## Feasibility Assessment

### Is `TritonKernelCall` CUDA-graph-safe?

After autotuning (which happens during warmup), `TritonKernelCall`:
1. Decodes the protobuf (CPU-side, done before graph capture)
2. Launches a pre-compiled PTX kernel via `cuLaunchKernel`

`cuLaunchKernel` **IS** graph-safe. The main concern is:
- **First invocation** per kernel config triggers autotuning (multiple trial
  launches) — NOT graph-safe
- **Subsequent invocations** — deterministic single launch — graph-safe

Since profiling/production always has warmup steps, this is acceptable. The
autotuning happens during warmup (outside profiled steps), and command buffers
are built from the compiled HLO which reflects post-autotuning state.

### Effort Estimate

| Option | Files Changed | Lines of Code | Risk | Timeline |
|--------|--------------|---------------|------|----------|
| A (FFI Shim) | 3–4 in jaxlib + 1 in TE | ~100–200 | Medium (buffer adaptation) | 1–2 weeks |
| B (Dual Registration) | 1–2 in jaxlib | ~20 | High (relies on undocumented behavior) | 1–2 days |
| C (Full Migration) | 10+ across jaxlib | ~500+ | Low (clean design) | 4–8 weeks |

**Recommendation:** Start with **Option A** (FFI Shim). It's the cleanest
short-term path, provides immediate command-buffer benefits, and is forward-
compatible with an eventual full migration (Option C).

## Verification

### How to confirm command buffer capture in profiles

1. **Thunk sequence**: With HLO dump enabled, check
   `*.thunk_sequence.txt` — Triton calls should appear inside a
   `CommandBuffer { ... }` block rather than as standalone `kCustomCall` entries.

2. **Trace viewer**: In xprof, look for `CommandBuffer` spans in the GPU
   timeline that group multiple kernel launches. Without command buffers,
   each Triton kernel appears as a separate span with host-side gaps.

3. **XLA logging**: Set `TF_CPP_VMODULE=command_buffer_conversion_pass=2` to
   see which thunks are being captured and which are rejected.

## Files to Modify

### Option A (FFI Shim)

**jaxlib (JAX repo):**
- `jaxlib/gpu/triton_kernels.cc` — Add FFI handler + registration
- `jaxlib/gpu/triton_kernels.h` — Add FFI handler declaration
- `jaxlib/gpu/BUILD` — Add `xla/ffi` dependency
- `jaxlib/gpu_triton.py` — Ensure FFI module is linked

**TransformerEngine:**
- `transformer_engine/jax/triton_extensions/utils.py` — Update `api_version`
  in `triton_call_lowering` from 2 to 4 (FFI)

## Relationship to JAX Issue #27988

Issue [#27988](https://github.com/jax-ml/jax/issues/27988) targets
`__gpu$xla.gpu.triton` (Pallas path). A JAX maintainer responded:

> "The solution here is likely to switch the Pallas triton internals to use
> XLA's FFI interface so that this support happens automatically."

Our `triton_kernel_call` is a **different target** but has the **same root
cause**: legacy custom call registration without FFI `kCmdBufferCompatible`
trait. The fix approach is identical. Proposing this to the JAX team as a
companion change to #27988 makes sense — both targets need FFI migration.
