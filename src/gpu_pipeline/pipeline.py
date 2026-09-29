"""GPU image pipeline and a NumPy CPU reference implementation.

Both implementations compute the same four outputs for an RGB image:

* ``gray``      - BT.601 luminance
* ``equalized`` - histogram-equalized luminance
* ``blurred``   - Gaussian-smoothed equalized image
* ``edges``     - Sobel gradient magnitude (or binary edge map)

The CPU version exists to (1) validate GPU results pixel-by-pixel and
(2) give a fair, vectorized baseline for the speed-up measurements.
"""

import ctypes
import dataclasses
import os
import time

import numpy as np

from gpu_pipeline import cuda_driver

KERNEL_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "kernels",
                           "image_kernels.cu")
MAX_BLUR_RADIUS = 16  # Must match MAX_BLUR_RADIUS in image_kernels.cu.
BLUR_TILE = 128  # Must match BLUR_TILE in image_kernels.cu.
COL_TILE_W = 32  # Must match COL_TILE_W in image_kernels.cu.
COL_TILE_H = 64  # Must match COL_TILE_H in image_kernels.cu.
COL_THREADS_Y = 8  # Must match COL_THREADS_Y in image_kernels.cu.
SOBEL_TILE = 16  # Must match SOBEL_TILE in image_kernels.cu.
LINEAR_BLOCK = 256
HIST_BINS = 256
# Max allowed |GPU - CPU| per output, in gray levels (see README, "Lessons").
VERIFY_TOLERANCE = {"gray": 0, "equalized": 0, "blurred": 1, "edges": 4}


@dataclasses.dataclass
class PipelineParams:
    """User-tunable algorithm parameters."""
    sigma: float = 1.5
    edge_threshold: float = -1.0  # < 0 keeps the raw magnitude.

    @property
    def radius(self):
        return min(MAX_BLUR_RADIUS, max(1, int(np.ceil(3.0 * self.sigma))))


@dataclasses.dataclass
class PipelineResult:
    """Outputs and timings for one image."""
    gray: np.ndarray
    equalized: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    kernel_ms: float = 0.0  # GPU time spent in kernels only.
    total_ms: float = 0.0  # Wall time incl. host<->device copies.
    stage_ms: dict = dataclasses.field(default_factory=dict)


def gaussian_weights(sigma, radius):
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return (weights / weights.sum()).astype(np.float32)


def _to_byte(values):
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def cpu_pipeline(rgb, params):
    """Vectorized NumPy implementation of the full pipeline."""
    start = time.perf_counter()
    rgb_i = rgb.astype(np.uint32)
    gray = ((77 * rgb_i[..., 0] + 150 * rgb_i[..., 1] + 29 * rgb_i[..., 2] +
             128) >> 8).astype(np.uint8)

    hist = np.bincount(gray.ravel(), minlength=HIST_BINS).astype(np.int64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    denom = gray.size - cdf_min
    if denom == 0:
        lut = np.arange(HIST_BINS, dtype=np.uint8)
    else:
        scaled = ((cdf - cdf_min) * 255 + denom // 2) // denom
        lut = np.where(cdf >= cdf_min, scaled, 0).astype(np.uint8)
    equalized = lut[gray]

    radius = params.radius
    weights = gaussian_weights(params.sigma, radius)
    padded = np.pad(equalized.astype(np.float32), ((0, 0), (radius, radius)),
                    mode="edge")
    rows = np.zeros(equalized.shape, dtype=np.float32)
    width = equalized.shape[1]
    for k, weight in enumerate(weights):
        rows += weight * padded[:, k:k + width]
    padded = np.pad(rows, ((radius, radius), (0, 0)), mode="edge")
    cols = np.zeros(equalized.shape, dtype=np.float32)
    height = equalized.shape[0]
    for k, weight in enumerate(weights):
        cols += weight * padded[k:k + height, :]
    blurred = _to_byte(cols)

    p = np.pad(blurred.astype(np.float32), 1, mode="edge")
    gx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) -
          (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]))
    gy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) -
          (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]))
    magnitude = np.sqrt(gx * gx + gy * gy)
    if params.edge_threshold >= 0:
        edges = np.where(magnitude >= params.edge_threshold, 255,
                         0).astype(np.uint8)
    else:
        edges = _to_byte(magnitude)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PipelineResult(gray, equalized, blurred, edges, 0.0, elapsed_ms)


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------


def _ceil_div(a, b):
    return (a + b - 1) // b


class GpuPipeline:
    """Runs the pipeline on the GPU, reusing device buffers across images.

    Device buffers are grown on demand and reused, so a batch of images only
    pays for cudaMalloc when a larger image than any seen so far arrives.
    """

    _KERNELS = ("RgbToGray", "Histogram256", "BuildEqualizationLut", "ApplyLut",
                "GaussianBlurRows", "GaussianBlurCols", "SobelMagnitude")
    _STAGES = ("h2d", "gray", "histogram", "lut", "equalize", "blur_rows",
               "blur_cols", "sobel", "d2h")

    def __init__(self, device, params):
        self._dev = device
        self._params = params
        with open(KERNEL_FILE, encoding="utf-8") as source_file:
            source = source_file.read()
        compile_start = time.perf_counter()
        self.binary, self.binary_kind = self._dev.compile_kernels(
            source, "image_kernels.cu")
        self._module = self._dev.load_module(self.binary)
        self.compile_ms = (time.perf_counter() - compile_start) * 1000.0
        self._fn = {
            name: self._dev.get_function(self._module, name)
            for name in self._KERNELS
        }
        self._upload_gaussian_weights()
        self._capacity = 0
        self._buffers = {}
        self._events = [
            self._dev.create_event() for _ in range(len(self._STAGES) + 1)
        ]
        # Enough blocks to fill every SM several times for grid-stride loops.
        self._linear_grid = device.multiprocessor_count * 8

    def _upload_gaussian_weights(self):
        weights = gaussian_weights(self._params.sigma, self._params.radius)
        pointer, size = self._dev.get_global(self._module, "c_gauss_weights")
        padded = np.zeros(size // 4, dtype=np.float32)
        padded[:weights.size] = weights
        self._dev.memcpy_htod(pointer, padded)

    def _ensure_capacity(self, num_pixels):
        if num_pixels <= self._capacity:
            return
        self._free_buffers()
        sizes = {
            "rgb": 3 * num_pixels,
            "gray": num_pixels,
            "equalized": num_pixels,
            "blur_tmp": 4 * num_pixels,  # float32 intermediate
            "blurred": num_pixels,
            "edges": num_pixels,
            "hist": 4 * HIST_BINS,
            "lut": HIST_BINS,
        }
        self._buffers = {name: self._dev.malloc(n) for name, n in sizes.items()}
        self._capacity = num_pixels

    def _free_buffers(self):
        for pointer in self._buffers.values():
            self._dev.free(pointer)
        self._buffers = {}
        self._capacity = 0

    def close(self):
        self._free_buffers()
        for event in self._events:
            self._dev.destroy_event(event)
        self._events = []

    def process(self, rgb):
        """Runs all stages on one HxWx3 uint8 image."""
        height, width = rgb.shape[:2]
        num_pixels = height * width
        self._ensure_capacity(num_pixels)
        buf = {k: ctypes.c_uint64(v) for k, v in self._buffers.items()}
        n = ctypes.c_int(num_pixels)
        w = ctypes.c_int(width)
        h = ctypes.c_int(height)
        radius = ctypes.c_int(self._params.radius)
        threshold = ctypes.c_float(self._params.edge_threshold)
        linear_grid = (min(self._linear_grid, _ceil_div(num_pixels,
                                                        LINEAR_BLOCK)), 1, 1)
        linear_block = (LINEAR_BLOCK, 1, 1)
        events = self._events
        dev = self._dev

        wall_start = time.perf_counter()
        dev.record_event(events[0])
        dev.memcpy_htod(self._buffers["rgb"], rgb)
        dev.record_event(events[1])
        dev.launch(self._fn["RgbToGray"], linear_grid, linear_block,
                   (buf["rgb"], buf["gray"], n))
        dev.record_event(events[2])
        dev.memset_d32(self._buffers["hist"], 0, HIST_BINS)
        dev.launch(self._fn["Histogram256"], linear_grid, linear_block,
                   (buf["gray"], buf["hist"], n))
        dev.record_event(events[3])
        dev.launch(self._fn["BuildEqualizationLut"], (1, 1, 1),
                   (HIST_BINS, 1, 1), (buf["hist"], buf["lut"], n))
        dev.record_event(events[4])
        dev.launch(self._fn["ApplyLut"], linear_grid, linear_block,
                   (buf["gray"], buf["equalized"], buf["lut"], n))
        dev.record_event(events[5])
        dev.launch(self._fn["GaussianBlurRows"],
                   (_ceil_div(width, BLUR_TILE), height, 1), (BLUR_TILE, 1, 1),
                   (buf["equalized"], buf["blur_tmp"], w, h, radius))
        dev.record_event(events[6])
        dev.launch(self._fn["GaussianBlurCols"],
                   (_ceil_div(width, COL_TILE_W), _ceil_div(
                       height, COL_TILE_H), 1), (COL_TILE_W, COL_THREADS_Y, 1),
                   (buf["blur_tmp"], buf["blurred"], w, h, radius))
        dev.record_event(events[7])
        dev.launch(
            self._fn["SobelMagnitude"],
            (_ceil_div(width, SOBEL_TILE), _ceil_div(height, SOBEL_TILE), 1),
            (SOBEL_TILE, SOBEL_TILE, 1),
            (buf["blurred"], buf["edges"], w, h, threshold))
        dev.record_event(events[8])

        outputs = {
            name: np.empty((height, width), dtype=np.uint8)
            for name in ("gray", "equalized", "blurred", "edges")
        }
        for name, host in outputs.items():
            dev.memcpy_dtoh(host, self._buffers[name])
        dev.record_event(events[9])
        dev.synchronize()
        total_ms = (time.perf_counter() - wall_start) * 1000.0

        stage_ms = {
            stage: dev.elapsed_ms(events[i], events[i + 1])
            for i, stage in enumerate(self._STAGES)
        }
        kernel_ms = sum(
            v for k, v in stage_ms.items() if k not in ("h2d", "d2h"))
        return PipelineResult(outputs["gray"], outputs["equalized"],
                              outputs["blurred"], outputs["edges"], kernel_ms,
                              total_ms, stage_ms)
