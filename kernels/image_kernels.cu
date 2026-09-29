// Copyright 2026 Harshul
//
// CUDA kernels for the batch image enhancement / edge detection pipeline.
//
// Pipeline per image:
//   RGB -> grayscale -> 256-bin histogram -> equalization LUT (GPU scan)
//       -> apply LUT -> separable Gaussian blur (row + column pass)
//       -> Sobel gradient magnitude (optionally thresholded to a binary map)
//
// The kernels are compiled at runtime with NVRTC and launched through the
// CUDA Driver API (see src/gpu_pipeline/cuda_driver.py). Every kernel is
// declared extern "C" so its symbol name is not mangled.

#define HIST_BINS 256
#define MAX_BLUR_RADIUS 16
#define BLUR_TILE 128   // threads per block for the row blur pass
#define COL_TILE_W 32   // column blur tile width (= one warp)
#define COL_TILE_H 64   // column blur tile height
#define COL_THREADS_Y 8 // column blur block is COL_TILE_W x COL_THREADS_Y
#define SOBEL_TILE 16   // SOBEL_TILE x SOBEL_TILE threads per block

// Gaussian weights live in constant memory: every thread in a warp reads the
// same weight at the same time, which is the broadcast case constant memory
// is designed for. The host writes it via cuModuleGetGlobal + cuMemcpyHtoD.
__constant__ float c_gauss_weights[2 * MAX_BLUR_RADIUS + 1];

__device__ __forceinline__ int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ unsigned char ToByte(float v) {
  return static_cast<unsigned char>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
}

// ---------------------------------------------------------------------------
// 1. Interleaved RGB (HxWx3, uint8) -> luminance (HxW, uint8), ITU-R BT.601
//    in 8.8 fixed point: (77 R + 150 G + 29 B + 128) >> 8. Integer math is
//    bit-exact with the CPU reference; float math with fast-math/FMA flips
//    pixels on .5 boundaries, which the equalization LUT then amplifies.
// ---------------------------------------------------------------------------
extern "C" __global__ void RgbToGray(const unsigned char* __restrict__ rgb,
                                     unsigned char* __restrict__ gray,
                                     int num_pixels) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    const unsigned int r = rgb[3 * i + 0];
    const unsigned int g = rgb[3 * i + 1];
    const unsigned int b = rgb[3 * i + 2];
    gray[i] = static_cast<unsigned char>((77u * r + 150u * g + 29u * b +
                                          128u) >> 8);
  }
}

// ---------------------------------------------------------------------------
// 2. 256-bin histogram using a per-block privatized histogram in shared
//    memory. Shared-memory atomics are far cheaper than global atomics and
//    each block only issues 256 global atomics at the end.
// ---------------------------------------------------------------------------
extern "C" __global__ void Histogram256(const unsigned char* __restrict__ img,
                                        unsigned int* __restrict__ hist,
                                        int num_pixels) {
  __shared__ unsigned int s_hist[HIST_BINS];
  for (int b = threadIdx.x; b < HIST_BINS; b += blockDim.x) s_hist[b] = 0;
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    atomicAdd(&s_hist[img[i]], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < HIST_BINS; b += blockDim.x) {
    if (s_hist[b] != 0) atomicAdd(&hist[b], s_hist[b]);
  }
}

// ---------------------------------------------------------------------------
// 3. Build the histogram-equalization look-up table on the GPU.
//    Launched with exactly one block of 256 threads. A Hillis-Steele inclusive
//    scan in shared memory turns the histogram into a CDF, then each thread
//    produces one LUT entry:
//        lut[v] = round((cdf[v] - cdf_min) * 255 / (N - cdf_min))
//    computed in 64-bit integers so it is exact.
// ---------------------------------------------------------------------------
extern "C" __global__ void BuildEqualizationLut(
    const unsigned int* __restrict__ hist, unsigned char* __restrict__ lut,
    int num_pixels) {
  __shared__ unsigned int s_cdf[HIST_BINS];
  __shared__ unsigned int s_cdf_min;
  const int t = threadIdx.x;
  s_cdf[t] = hist[t];
  __syncthreads();

  for (int offset = 1; offset < HIST_BINS; offset <<= 1) {
    const unsigned int add = (t >= offset) ? s_cdf[t - offset] : 0u;
    __syncthreads();
    s_cdf[t] += add;
    __syncthreads();
  }

  // cdf_min = CDF value at the first non-empty bin. Bins before it have
  // cdf == 0, so the smallest non-zero CDF value is what we want.
  if (t == 0) s_cdf_min = 0xFFFFFFFFu;
  __syncthreads();
  if (s_cdf[t] != 0) atomicMin(&s_cdf_min, s_cdf[t]);
  __syncthreads();

  const unsigned int cdf_min = s_cdf_min;
  const unsigned int denom = static_cast<unsigned int>(num_pixels) - cdf_min;
  if (denom == 0 || s_cdf[t] < cdf_min) {
    // Constant image, or a bin below the first occupied one.
    lut[t] = (denom == 0) ? static_cast<unsigned char>(t) : 0;
  } else {
    const unsigned long long numer =
        static_cast<unsigned long long>(s_cdf[t] - cdf_min) * 255ull;
    lut[t] = static_cast<unsigned char>((numer + denom / 2) / denom);
  }
}

// ---------------------------------------------------------------------------
// 4. Apply a 256-entry LUT. The LUT is staged in shared memory once per block.
// ---------------------------------------------------------------------------
extern "C" __global__ void ApplyLut(const unsigned char* __restrict__ in,
                                    unsigned char* __restrict__ out,
                                    const unsigned char* __restrict__ lut,
                                    int num_pixels) {
  __shared__ unsigned char s_lut[HIST_BINS];
  for (int b = threadIdx.x; b < HIST_BINS; b += blockDim.x) s_lut[b] = lut[b];
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    out[i] = s_lut[in[i]];
  }
}

// ---------------------------------------------------------------------------
// 5a. Horizontal Gaussian pass (uint8 -> float). One block handles BLUR_TILE
//     consecutive pixels of one row; the tile plus a halo of `radius` pixels
//     on each side is loaded into shared memory so each input pixel is read
//     from global memory once instead of (2 * radius + 1) times.
//     Borders use clamp-to-edge.
// ---------------------------------------------------------------------------
extern "C" __global__ void GaussianBlurRows(const unsigned char* __restrict__ in,
                                            float* __restrict__ out, int width,
                                            int height, int radius) {
  __shared__ float s_row[BLUR_TILE + 2 * MAX_BLUR_RADIUS];
  const int y = blockIdx.y;
  const int x0 = blockIdx.x * BLUR_TILE;
  const int tile_len = BLUR_TILE + 2 * radius;
  const unsigned char* row = in + static_cast<size_t>(y) * width;

  for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
    const int gx = ClampInt(x0 + i - radius, 0, width - 1);
    s_row[i] = row[gx];
  }
  __syncthreads();

  const int x = x0 + threadIdx.x;
  if (x >= width || y >= height) return;
  float acc = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    acc += c_gauss_weights[k + radius] * s_row[threadIdx.x + radius + k];
  }
  out[static_cast<size_t>(y) * width + x] = acc;
}

// ---------------------------------------------------------------------------
// 5b. Vertical Gaussian pass (float -> uint8).
//     A first version mirrored 5a with one block per column, so the 32
//     threads of a warp read 32 vertically adjacent pixels - 32 separate
//     memory transactions per warp load - and it ran ~3.8x slower than the
//     row pass. This version uses 2-D tiles COL_TILE_W columns wide and
//     COL_TILE_H rows tall (plus halo): threadIdx.x walks along a row, so every
//     warp load is one coalesced 128-byte transaction. Each thread computes
//     COL_TILE_H / COL_THREADS_Y outputs. The +1 padding column avoids
//     shared-memory bank conflicts.
// ---------------------------------------------------------------------------
extern "C" __global__ void GaussianBlurCols(const float* __restrict__ in,
                                            unsigned char* __restrict__ out,
                                            int width, int height,
                                            int radius) {
  __shared__ float s_tile[COL_TILE_H + 2 * MAX_BLUR_RADIUS][COL_TILE_W + 1];
  const int tx = threadIdx.x;
  const int x = blockIdx.x * COL_TILE_W + tx;
  const int y0 = blockIdx.y * COL_TILE_H;
  const int tile_len = COL_TILE_H + 2 * radius;
  const int cx = min(x, width - 1);

  for (int i = threadIdx.y; i < tile_len; i += COL_THREADS_Y) {
    const int gy = ClampInt(y0 + i - radius, 0, height - 1);
    s_tile[i][tx] = in[static_cast<size_t>(gy) * width + cx];
  }
  __syncthreads();

  if (x >= width) return;
  for (int i = threadIdx.y; i < COL_TILE_H; i += COL_THREADS_Y) {
    const int y = y0 + i;
    if (y >= height) break;
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
      acc += c_gauss_weights[k + radius] * s_tile[i + radius + k][tx];
    }
    out[static_cast<size_t>(y) * width + x] = ToByte(acc);
  }
}

// ---------------------------------------------------------------------------
// 6. Sobel gradient magnitude with a 2-D shared-memory tile (+1 pixel halo).
//    If threshold >= 0 the output is a binary edge map (0 / 255); otherwise it
//    is the clamped magnitude.
// ---------------------------------------------------------------------------
extern "C" __global__ void SobelMagnitude(const unsigned char* __restrict__ in,
                                          unsigned char* __restrict__ out,
                                          int width, int height,
                                          float threshold) {
  __shared__ float s_tile[SOBEL_TILE + 2][SOBEL_TILE + 2];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x * SOBEL_TILE;
  const int by = blockIdx.y * SOBEL_TILE;

  // Cooperative load of the (TILE+2)^2 region, clamped at image borders.
  for (int j = ty; j < SOBEL_TILE + 2; j += SOBEL_TILE) {
    for (int i = tx; i < SOBEL_TILE + 2; i += SOBEL_TILE) {
      const int gx = ClampInt(bx + i - 1, 0, width - 1);
      const int gy = ClampInt(by + j - 1, 0, height - 1);
      s_tile[j][i] = in[static_cast<size_t>(gy) * width + gx];
    }
  }
  __syncthreads();

  const int x = bx + tx;
  const int y = by + ty;
  if (x >= width || y >= height) return;

  const int cx = tx + 1;
  const int cy = ty + 1;
  const float gx = (s_tile[cy - 1][cx + 1] + 2.0f * s_tile[cy][cx + 1] +
                    s_tile[cy + 1][cx + 1]) -
                   (s_tile[cy - 1][cx - 1] + 2.0f * s_tile[cy][cx - 1] +
                    s_tile[cy + 1][cx - 1]);
  const float gy = (s_tile[cy + 1][cx - 1] + 2.0f * s_tile[cy + 1][cx] +
                    s_tile[cy + 1][cx + 1]) -
                   (s_tile[cy - 1][cx - 1] + 2.0f * s_tile[cy - 1][cx] +
                    s_tile[cy - 1][cx + 1]);
  const float mag = sqrtf(gx * gx + gy * gy);
  const size_t idx = static_cast<size_t>(y) * width + x;
  if (threshold >= 0.0f) {
    out[idx] = mag >= threshold ? 255 : 0;
  } else {
    out[idx] = ToByte(mag);
  }
}
