"""Minimal ctypes bindings for the CUDA Driver API and NVRTC.

This module deliberately avoids third-party GPU wrappers (CuPy, PyCUDA, ...)
and talks to the NVIDIA libraries directly:

* ``nvcuda.dll`` / ``libcuda.so.1`` - the CUDA Driver API that ships with the
  NVIDIA display driver (context, memory, modules, kernel launch, events).
* ``nvrtc64_*.dll`` / ``libnvrtc.so`` - the NVIDIA runtime compiler, which
  turns CUDA C++ source into PTX at run time. It comes from the
  ``nvidia-cuda-nvrtc-cu12`` pip wheel or from a local CUDA Toolkit.

Only the handful of entry points the pipeline needs are wrapped.
"""

import ctypes
import ctypes.util
import glob
import os
import site
import sys

import numpy as np

# Opaque driver handles are pointers; device pointers are 64-bit integers.
CUdeviceptr = ctypes.c_uint64
_HANDLE = ctypes.c_void_p

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16


class CudaError(RuntimeError):
    """Raised when a Driver API or NVRTC call returns a non-zero status."""


def _load_driver():
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")


def _candidate_nvrtc_paths():
    """Yields likely NVRTC library locations, most specific first."""
    site_dirs = list(site.getsitepackages()) + [site.getusersitepackages()]
    if sys.platform == "win32":
        patterns = [
            os.path.join("nvidia", "cuda_nvrtc", "bin", "nvrtc64_*.dll")
        ]
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            yield from sorted(
                glob.glob(os.path.join(cuda_path, "bin", "nvrtc64_*.dll")))
    else:
        patterns = [os.path.join("nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*")]
    for site_dir in site_dirs:
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(site_dir, pattern))):
                if "builtins" not in path and not path.endswith(".alt.dll"):
                    yield path
    found = ctypes.util.find_library("nvrtc")
    if found:
        yield found
    if sys.platform != "win32":
        yield from sorted(glob.glob("/usr/local/cuda*/lib64/libnvrtc.so*"))


def _load_nvrtc():
    errors = []
    for path in _candidate_nvrtc_paths():
        try:
            if sys.platform == "win32":
                os.add_dll_directory(os.path.dirname(path))
                return ctypes.WinDLL(path)
            return ctypes.CDLL(path)
        except OSError as exc:  # Try the next candidate.
            errors.append(f"{path}: {exc}")
    raise CudaError("Could not load NVRTC. Install it with "
                    "'pip install nvidia-cuda-nvrtc-cu12' or install the CUDA "
                    "Toolkit.\n" + "\n".join(errors))


class CudaDevice:
    """Owns a CUDA context on one device and exposes the calls we need."""

    def __init__(self, device_ordinal=0):
        self._cu = _load_driver()
        self._nvrtc = _load_nvrtc()
        self._check(self._cu.cuInit(0), "cuInit")

        self._device = ctypes.c_int()
        self._check(
            self._cu.cuDeviceGet(ctypes.byref(self._device), device_ordinal),
            "cuDeviceGet")
        self._context = _HANDLE()
        self._check(
            self._cu.cuCtxCreate_v2(ctypes.byref(self._context), 0,
                                    self._device), "cuCtxCreate")

        name = ctypes.create_string_buffer(256)
        self._cu.cuDeviceGetName(name, 256, self._device)
        self.name = name.value.decode()
        self.compute_capability = (
            self._attribute(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            self._attribute(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
        self.multiprocessor_count = self._attribute(
            _CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        total = ctypes.c_size_t()
        self._cu.cuDeviceTotalMem_v2(ctypes.byref(total), self._device)
        self.total_memory_bytes = total.value
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        self.nvrtc_version = (major.value, minor.value)
        driver_version = ctypes.c_int()
        self._cu.cuDriverGetVersion(ctypes.byref(driver_version))
        self.driver_cuda_version = (driver_version.value // 1000,
                                    (driver_version.value % 1000) // 10)

    # -- Error handling -------------------------------------------------------

    def _check(self, status, what):
        if status != 0:
            message = ctypes.c_char_p()
            self._cu.cuGetErrorString(status, ctypes.byref(message))
            text = message.value.decode() if message.value else "unknown error"
            raise CudaError(f"{what} failed with CUDA error {status}: {text}")

    def _check_nvrtc(self, status, what):
        if status != 0:
            self._nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
            text = self._nvrtc.nvrtcGetErrorString(status).decode()
            raise CudaError(f"{what} failed with NVRTC error {status}: {text}")

    def _attribute(self, attribute):
        value = ctypes.c_int()
        self._check(
            self._cu.cuDeviceGetAttribute(ctypes.byref(value), attribute,
                                          self._device), "cuDeviceGetAttribute")
        return value.value

    # -- Compilation ----------------------------------------------------------

    def compile_kernels(self, source, name="kernels.cu", extra_options=()):
        """Compiles CUDA C++ source for this device with NVRTC.

    Prefers a CUBIN (native SASS for sm_XY): CUDA minor-version compatibility
    lets a CUBIN from a newer 12.x NVRTC run on an older 12.x driver, whereas
    PTX from a newer NVRTC is rejected (CUDA_ERROR_UNSUPPORTED_PTX_VERSION).
    Falls back to PTX if NVRTC does not know the real architecture.

    Returns:
      (image bytes, "cubin" | "ptx"), ready for load_module().
    """
        major, minor = self.compute_capability
        try:
            return self._compile(source, name, f"sm_{major}{minor}",
                                 extra_options), "cubin"
        except CudaError:
            return self._compile(source, name, f"compute_{major}{minor}",
                                 extra_options), "ptx"

    def _compile(self, source, name, arch, extra_options):
        options = [
            f"--gpu-architecture={arch}", "--use_fast_math", "-default-device",
            *extra_options
        ]
        c_options = (ctypes.c_char_p *
                     len(options))(*[opt.encode() for opt in options])
        want_cubin = arch.startswith("sm_")

        program = _HANDLE()
        self._check_nvrtc(
            self._nvrtc.nvrtcCreateProgram(ctypes.byref(program),
                                           source.encode(), name.encode(), 0,
                                           None, None), "nvrtcCreateProgram")
        try:
            status = self._nvrtc.nvrtcCompileProgram(program, len(options),
                                                     c_options)
            log_size = ctypes.c_size_t()
            self._nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            self._nvrtc.nvrtcGetProgramLog(program, log)
            if status != 0:
                raise CudaError(f"NVRTC compilation of {name} failed:\n"
                                f"{log.value.decode(errors='replace')}")
            kind = "CUBIN" if want_cubin else "PTX"
            get_size = getattr(self._nvrtc, f"nvrtcGet{kind}Size")
            get_image = getattr(self._nvrtc, f"nvrtcGet{kind}")
            image_size = ctypes.c_size_t()
            self._check_nvrtc(get_size(program, ctypes.byref(image_size)),
                              f"nvrtcGet{kind}Size")
            image = ctypes.create_string_buffer(image_size.value)
            self._check_nvrtc(get_image(program, image), f"nvrtcGet{kind}")
            return image.raw
        finally:
            self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def load_module(self, image):
        """Loads a CUBIN (or JIT-compiles PTX) and returns a CUmodule."""
        module = _HANDLE()
        self._check(self._cu.cuModuleLoadData(ctypes.byref(module), image),
                    "cuModuleLoadData")
        return module

    def get_function(self, module, name):
        function = _HANDLE()
        self._check(
            self._cu.cuModuleGetFunction(ctypes.byref(function), module,
                                         name.encode()),
            f"cuModuleGetFunction({name})")
        return function

    def get_global(self, module, name):
        """Returns (device pointer, size) of a __device__/__constant__ symbol."""
        pointer = CUdeviceptr()
        size = ctypes.c_size_t()
        self._check(
            self._cu.cuModuleGetGlobal_v2(ctypes.byref(pointer),
                                          ctypes.byref(size), module,
                                          name.encode()),
            f"cuModuleGetGlobal({name})")
        return pointer.value, size.value

    # -- Memory ---------------------------------------------------------------

    def malloc(self, num_bytes):
        pointer = CUdeviceptr()
        self._check(
            self._cu.cuMemAlloc_v2(ctypes.byref(pointer),
                                   ctypes.c_size_t(max(num_bytes, 1))),
            "cuMemAlloc")
        return pointer.value

    def free(self, pointer):
        self._check(self._cu.cuMemFree_v2(CUdeviceptr(pointer)), "cuMemFree")

    def memcpy_htod(self, dst_pointer, host_array):
        host_array = np.ascontiguousarray(host_array)
        self._check(
            self._cu.cuMemcpyHtoD_v2(CUdeviceptr(dst_pointer),
                                     host_array.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_size_t(host_array.nbytes)),
            "cuMemcpyHtoD")

    def memcpy_dtoh(self, host_array, src_pointer):
        if not host_array.flags["C_CONTIGUOUS"]:
            raise ValueError("Destination host array must be C-contiguous.")
        self._check(
            self._cu.cuMemcpyDtoH_v2(host_array.ctypes.data_as(ctypes.c_void_p),
                                     CUdeviceptr(src_pointer),
                                     ctypes.c_size_t(host_array.nbytes)),
            "cuMemcpyDtoH")

    def memset_d32(self, pointer, value, count):
        self._check(
            self._cu.cuMemsetD32_v2(CUdeviceptr(pointer), ctypes.c_uint(value),
                                    ctypes.c_size_t(count)), "cuMemsetD32")

    # -- Execution ------------------------------------------------------------

    def launch(self, function, grid, block, args, shared_bytes=0):
        """Launches a kernel.

    Args:
      function: CUfunction handle from get_function().
      grid: (x, y, z) number of blocks.
      block: (x, y, z) threads per block.
      args: sequence of ctypes scalars (c_uint64 for device pointers,
        c_int, c_float, ...) in kernel-parameter order.
      shared_bytes: dynamic shared memory per block.
    """
        arg_pointers = (ctypes.c_void_p * len(args))(
            *
            [ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in args])
        self._check(
            self._cu.cuLaunchKernel(function, grid[0], grid[1], grid[2],
                                    block[0], block[1], block[2], shared_bytes,
                                    None, arg_pointers, None), "cuLaunchKernel")

    def synchronize(self):
        self._check(self._cu.cuCtxSynchronize(), "cuCtxSynchronize")

    def create_event(self):
        event = _HANDLE()
        self._check(self._cu.cuEventCreate(ctypes.byref(event), 0),
                    "cuEventCreate")
        return event

    def record_event(self, event):
        self._check(self._cu.cuEventRecord(event, None), "cuEventRecord")

    def elapsed_ms(self, start_event, end_event):
        """Milliseconds between two recorded events (waits for end_event)."""
        self._check(self._cu.cuEventSynchronize(end_event),
                    "cuEventSynchronize")
        elapsed = ctypes.c_float()
        self._check(
            self._cu.cuEventElapsedTime(ctypes.byref(elapsed), start_event,
                                        end_event), "cuEventElapsedTime")
        return elapsed.value

    def destroy_event(self, event):
        self._cu.cuEventDestroy_v2(event)

    def close(self):
        if self._context:
            self._cu.cuCtxDestroy_v2(self._context)
            self._context = _HANDLE()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
