"""Register a no-op FFI handler for ``triton_kernel_call`` so that XLA's
command-buffer conversion pass treats Triton custom calls as CUDA-graph
compatible.

**How it works**

1. Execution path (unchanged): TE's Triton kernels are lowered with
   ``api_version=API_VERSION_STATUS_RETURNING`` (legacy), so XLA creates a
   ``CustomCallThunk`` that executes via the legacy custom-call registry.

2. Command-buffer eligibility (new): ``IsConvertible`` in
   ``command_buffer_conversion_pass.cc`` calls
   ``ffi::FindHandler("triton_kernel_call", "gpu")``. Without this patch,
   no FFI handler exists and the call returns false. This patch registers a
   shim FFI handler with ``kCmdBufferCompatible``, making ``FindHandler``
   succeed and ``IsConvertible`` return true.

3. The ``CommandBufferThunk`` then wraps the legacy thunk and captures its
   GPU work (``cuLaunchKernel``) via CUDA graph tracing. The shim FFI handler
   is never actually called.

**Usage** — import this module before JAX compiles any program::

    import maxtext.triton_cmd_buffer_patch  # noqa: F401

Or set the env-var in ``profile_moe_comparison.sh`` before the training command.

**Safety**: Post-warmup (after Triton autotuning), ``TritonKernelCall`` only
calls ``cuLaunchKernel`` which is CUDA-graph safe. During warmup, autotuning
involves trial launches that are NOT graph-safe, but command buffers are built
from the compiled HLO which reflects post-autotuning state.
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XLA FFI C ABI constants (from xla/ffi/api/c_api.h)
#
# When XLA registers an FFI handler it *calls* the handler with a special
# call frame carrying an XLA_FFI_Metadata_Extension.  The handler must
# detect this extension, populate the XLA_FFI_Metadata struct with the
# API version and traits, then return NULL.
#
# Struct layouts on LP64 (x86-64 / aarch64):
#
#   XLA_FFI_CallFrame            offset
#     size_t   struct_size         0
#     void*    extension_start     8   ← pointer to first extension
#     ...
#
#   XLA_FFI_Extension_Base       offset
#     size_t   struct_size         0
#     int      type                8   (XLA_FFI_Extension_Metadata = 1)
#     // 4 bytes padding
#     void*    next               16
#                          total: 24
#
#   XLA_FFI_Metadata_Extension   offset
#     Extension_Base base          0   (24 bytes)
#     Metadata*      metadata     24
#
#   XLA_FFI_Api_Version          offset
#     size_t   struct_size         0
#     void*    extension_start     8
#     int      major_version      16
#     int      minor_version      20
#                          total: 24
#
#   XLA_FFI_Metadata             offset
#     size_t         struct_size   0
#     Api_Version    api_version   8   (24 bytes)
#     uint32_t       traits       32
#     TypeId         state_type   36
# ---------------------------------------------------------------------------
_XLA_FFI_Extension_Metadata = 1
_XLA_FFI_Api_Version_STRUCT_SIZE = 24
_XLA_FFI_HANDLER_TYPE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)


def _ffi_handler(call_frame_ptr):
    """XLA FFI handler that responds to metadata queries with the correct
    API version and kCmdBufferCompatible trait.  For any other call frame
    (which should never happen) it returns success as a no-op."""
    if not call_frame_ptr:
        return None

    # call_frame->extension_start at offset 8
    ext_ptr = ctypes.c_void_p.from_address(call_frame_ptr + 8).value
    if not ext_ptr:
        return None

    # extension_base.type at offset 8
    ext_type = ctypes.c_int.from_address(ext_ptr + 8).value
    if ext_type != _XLA_FFI_Extension_Metadata:
        return None

    # XLA_FFI_Metadata_Extension.metadata at offset 24
    metadata_ptr = ctypes.c_void_p.from_address(ext_ptr + 24).value
    if not metadata_ptr:
        return None

    # Populate metadata->api_version  (starts at metadata + 8)
    ctypes.c_uint64.from_address(metadata_ptr + 8).value = (
        _XLA_FFI_Api_Version_STRUCT_SIZE
    )
    ctypes.c_uint64.from_address(metadata_ptr + 16).value = 0     # extension_start = NULL
    ctypes.c_int.from_address(metadata_ptr + 24).value = 0        # major = 0
    ctypes.c_int.from_address(metadata_ptr + 28).value = 1        # minor = 1 (min supported)

    # Populate metadata->traits at offset 32
    ctypes.c_uint32.from_address(metadata_ptr + 32).value = 1     # kCmdBufferCompatible

    return None  # NULL → no error


_ffi_func = _XLA_FFI_HANDLER_TYPE(_ffi_handler)


def _make_pycapsule(ffi_func):
    """Wrap a ctypes CFUNCTYPE callback in a PyCapsule."""
    _PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    _PyCapsule_New.restype = ctypes.py_object
    _PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.CFUNCTYPE(None, ctypes.py_object),
    )
    return _PyCapsule_New(ffi_func, None, ctypes.CFUNCTYPE(None, ctypes.py_object)(0))


_handler_capsule = _make_pycapsule(_ffi_func)

# ---------------------------------------------------------------------------
# Register the shim in the *global* FFI handler registry, bypassing the GPU
# plugin handler (which rejects traits).
#
# jaxlib._jax.register_custom_call_target → PyRegisterCustomCallTarget (ffi.cc)
#   → ffi::Ffi::RegisterStaticHandler  (process-global FFI registry)
#
# Platform "cuda" is canonicalized to match FindHandler("triton_kernel_call", "gpu")
# since CanonicalPlatformName("gpu") == "cuda" on CUDA builds.
# ---------------------------------------------------------------------------
_COMMAND_BUFFER_COMPATIBLE = 1  # XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE

def _register():
    try:
        from jaxlib import _jax  # core XLA extension (always available)
    except ImportError:
        logger.warning(
            "triton_cmd_buffer_patch: could not import jaxlib._jax; "
            "command-buffer capture for Triton calls will NOT be enabled."
        )
        return

    try:
        _jax.register_custom_call_target(
            "triton_kernel_call",
            {"execute": _handler_capsule},
            "cuda",                         # canonical GPU platform
            1,                              # api_version = FFI
            _COMMAND_BUFFER_COMPATIBLE,     # traits
        )
        logger.info(
            "triton_cmd_buffer_patch: registered FFI shim for "
            "'triton_kernel_call' with kCmdBufferCompatible"
        )
    except Exception as e:
        logger.warning(
            "triton_cmd_buffer_patch: FFI shim registration failed: %s", e
        )

_register()
