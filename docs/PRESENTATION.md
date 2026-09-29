# Presentation outline and script (target: 7-8 minutes)

The rubric needs a 5-10 minute video. For the top grade it should include a
live demo and a discussion of next steps. Record your screen (OBS, Zoom or
the Windows Game Bar with Win+Alt+R), then upload to YouTube as *unlisted*
and paste the link in the submission.

| Time | Show on screen | Say |
|------|----------------|-----|
| 0:00-0:45 | README title + montage | Goal: batch-enhance images and extract edges on the GPU, verify against a CPU reference, and measure how speed-up scales with image size. Why it matters: this preprocessing runs in front of OCR, medical-imaging and computer-vision models over thousands of images. |
| 0:45-1:30 | Architecture diagram in README | Python host, ctypes to the CUDA Driver API and NVRTC. Kernels compiled at run time for the detected GPU (sm_86 here). Explain why CuPy was not usable here: Smart App Control blocked its unsigned DLL, and going lower-level turned into a learning opportunity. |
| 1:30-4:00 | `kernels/image_kernels.cu` | Walk through the kernels: grid-stride gray; privatized shared-memory histogram (why shared atomics beat global); the single-block Hillis-Steele scan for the CDF and the `__syncthreads` placement; separable Gaussian (2r+1 instead of (2r+1)^2 multiplies), shared tile + halo, weights in `__constant__` memory (broadcast); Sobel with a 2-D tile. |
| 4:00-4:40 | `cuda_driver.py` `launch()` and `compile_ptx()` | How a kernel launch looks at the Driver API level: void** argument array, grid/block dims, events for timing. |
| 4:40-6:00 | **Live demo**: run `./run.sh` in a terminal | Point out the GPU info line, NVRTC compile time, the per-image GPU vs CPU times, the PASS verification and the unit tests. Open a few output PNGs (the washed-out band recovered, the edge map). |
| 6:00-7:00 | `docs/figures/timing.png`, `stages.png` | Results: speed-up grows with image size; small images are launch- and transfer-bound. The stage chart shows PCIe copies dominate, and the column blur is slower than the row blur because of uncoalesced access. |
| 7:00-8:00 | README "Next steps" | Streams + pinned memory to overlap copies, kernel fusion, a coalesced column pass, nvJPEG decode, Canny. Wrap up with what you learned. |

Tips: keep the terminal font large, and do one practice run beforehand so the
NVRTC compile and PNG decode are warm in the file cache.
