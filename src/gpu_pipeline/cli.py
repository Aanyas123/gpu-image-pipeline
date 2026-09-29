"""Command-line entry point: batch-process a directory of images on the GPU.

Example:
    python -m gpu_pipeline.cli --input data/input --output data/output \
        --sigma 1.5 --benchmark --verify
"""

import argparse
import csv
import datetime
import logging
import os
import platform
import sys
import time

import numpy as np
from PIL import Image

from gpu_pipeline import cuda_driver
from gpu_pipeline import pipeline

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm",
                    ".pgm")
OUTPUT_KINDS = ("gray", "equalized", "blurred", "edges")
CSV_FIELDS = ("image", "width", "height", "megapixels", "gpu_kernel_ms",
              "gpu_total_ms", "cpu_ms", "speedup_kernel", "speedup_total",
              "gpu_mpix_per_s", "max_abs_diff", "mismatch_pct", "h2d_ms",
              "gray_ms", "histogram_ms", "lut_ms", "equalize_ms",
              "blur_rows_ms", "blur_cols_ms", "sobel_ms", "d2h_ms")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("GPU batch image enhancement and edge detection "
                     "(custom CUDA kernels via NVRTC + CUDA Driver API)."))
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        help="Directory of input images (or a single image).")
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Directory to write processed images to.")
    parser.add_argument("--sigma",
                        type=float,
                        default=1.5,
                        help="Gaussian blur sigma (default: 1.5). The kernel "
                        "radius is ceil(3*sigma), capped at 16.")
    parser.add_argument("--threshold",
                        type=float,
                        default=-1.0,
                        help="Sobel threshold; >= 0 emits a binary edge map, "
                        "< 0 emits gradient magnitude (default: -1).")
    parser.add_argument("--outputs",
                        default="equalized,edges",
                        help="Comma-separated subset of "
                        f"{','.join(OUTPUT_KINDS)} to save "
                        "(default: equalized,edges).")
    parser.add_argument("--verify",
                        action="store_true",
                        help="Also run the NumPy CPU reference and compare "
                        "outputs pixel-by-pixel.")
    parser.add_argument("--benchmark",
                        action="store_true",
                        help="Time the CPU reference too and report speed-ups "
                        "(implies --verify).")
    parser.add_argument("--repeat",
                        type=int,
                        default=3,
                        help="GPU runs per image; the fastest is reported "
                        "(default: 3). The first run of the batch is a "
                        "warm-up.")
    parser.add_argument("--csv",
                        default=None,
                        help="Path of the per-image metrics CSV "
                        "(default: <output>/metrics.csv).")
    parser.add_argument("--log",
                        default=None,
                        help="Path of the run log (default: <output>/run.log).")
    parser.add_argument("--device",
                        type=int,
                        default=0,
                        help="CUDA device ordinal (default: 0).")
    args = parser.parse_args(argv)
    args.outputs = [kind.strip() for kind in args.outputs.split(",") if kind]
    unknown = set(args.outputs) - set(OUTPUT_KINDS)
    if unknown:
        parser.error(f"Unknown --outputs value(s): {sorted(unknown)}")
    if not 0 < args.sigma <= 16 / 3:
        parser.error("--sigma must be in (0, 5.33] so the radius fits.")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1.")
    args.verify = args.verify or args.benchmark
    return args


def find_images(path):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(IMAGE_EXTENSIONS))


def setup_logging(log_path):
    logger = logging.getLogger("gpu_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def compare(gpu_image, cpu_image):
    diff = np.abs(gpu_image.astype(np.int16) - cpu_image.astype(np.int16))
    return int(diff.max()), float((diff > 1).mean() * 100.0)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output, "metrics.csv")
    log = setup_logging(args.log or os.path.join(args.output, "run.log"))

    images = find_images(args.input)
    if not images:
        log.error("No images found in %s", args.input)
        return 1
    params = pipeline.PipelineParams(sigma=args.sigma,
                                     edge_threshold=args.threshold)

    with cuda_driver.CudaDevice(args.device) as device:
        log.info("Run started %s on %s (Python %s)",
                 datetime.datetime.now().isoformat(timespec="seconds"),
                 platform.platform(), platform.python_version())
        log.info(
            "GPU: %s | compute capability %d.%d | %d SMs | %.1f GiB | "
            "driver CUDA %d.%d | NVRTC %d.%d", device.name,
            *device.compute_capability, device.multiprocessor_count,
            device.total_memory_bytes / 2**30, *device.driver_cuda_version,
            *device.nvrtc_version)
        gpu = pipeline.GpuPipeline(device, params)
        log.info(
            "Compiled kernels/image_kernels.cu with NVRTC in %.1f ms "
            "(%d-byte %s for sm_%d%d)", gpu.compile_ms, len(gpu.binary),
            gpu.binary_kind.upper(), *device.compute_capability)
        log.info("Parameters: sigma=%.2f radius=%d threshold=%.1f repeat=%d",
                 params.sigma, params.radius, params.edge_threshold,
                 args.repeat)
        log.info("Processing %d image(s) from %s", len(images), args.input)

        # Warm-up: the first launch of each kernel includes one-off JIT/driver
        # overhead that should not be attributed to any single image.
        first = np.asarray(Image.open(images[0]).convert("RGB"))
        gpu.process(first)

        rows = []
        all_passed = True
        batch_start = time.perf_counter()
        try:
            for index, path in enumerate(images, start=1):
                rgb = np.ascontiguousarray(
                    np.asarray(Image.open(path).convert("RGB")))
                height, width = rgb.shape[:2]
                result = min((gpu.process(rgb) for _ in range(args.repeat)),
                             key=lambda r: r.total_ms)
                stem = os.path.splitext(os.path.basename(path))[0]
                for kind in args.outputs:
                    Image.fromarray(getattr(result, kind)).save(
                        os.path.join(args.output, f"{stem}_{kind}.png"))

                megapixels = width * height / 1e6
                row = {
                    "image":
                        os.path.basename(path),
                    "width":
                        width,
                    "height":
                        height,
                    "megapixels":
                        round(megapixels, 3),
                    "gpu_kernel_ms":
                        round(result.kernel_ms, 3),
                    "gpu_total_ms":
                        round(result.total_ms, 3),
                    "gpu_mpix_per_s":
                        round(megapixels / (result.kernel_ms / 1000.0), 1),
                }
                row.update({
                    f"{stage}_ms": round(ms, 4)
                    for stage, ms in result.stage_ms.items()
                })
                message = (f"[{index}/{len(images)}] {row['image']} "
                           f"{width}x{height}: GPU kernels "
                           f"{result.kernel_ms:.2f} ms, GPU total "
                           f"{result.total_ms:.2f} ms")
                if args.verify:
                    reference = pipeline.cpu_pipeline(rgb, params)
                    worst_diff, worst_pct, passed = 0, 0.0, True
                    for kind in OUTPUT_KINDS:
                        diff, pct = compare(getattr(result, kind),
                                            getattr(reference, kind))
                        worst_diff = max(worst_diff, diff)
                        worst_pct = max(worst_pct, pct)
                        passed &= diff <= pipeline.VERIFY_TOLERANCE[kind]
                    all_passed &= passed
                    row.update({
                        "cpu_ms":
                            round(reference.total_ms, 3),
                        "speedup_kernel":
                            round(reference.total_ms / result.kernel_ms, 1),
                        "speedup_total":
                            round(reference.total_ms / result.total_ms, 1),
                        "max_abs_diff":
                            worst_diff,
                        "mismatch_pct":
                            round(worst_pct, 4),
                    })
                    message += (f", CPU {reference.total_ms:.2f} ms "
                                f"(x{row['speedup_total']} end-to-end, "
                                f"x{row['speedup_kernel']} kernels), "
                                f"max |diff| {worst_diff}, "
                                f"{worst_pct:.3f}% px off by >1 "
                                f"[{'PASS' if passed else 'FAIL'}]")
                log.info(message)
                rows.append(row)
        finally:
            gpu.close()

    batch_s = time.perf_counter() - batch_start
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_mpix = sum(row["megapixels"] for row in rows)
    kernel_s = sum(row["gpu_kernel_ms"] for row in rows) / 1000.0
    gpu_total_s = sum(row["gpu_total_ms"] for row in rows) / 1000.0
    log.info(
        "Batch done: %d images, %.1f megapixels, %.1f s wall "
        "(includes PNG decode/encode%s)", len(rows), total_mpix, batch_s,
        " and CPU reference" if args.verify else "")
    log.info(
        "GPU kernel throughput %.0f Mpix/s; end-to-end incl. PCIe copies "
        "%.0f Mpix/s", total_mpix / kernel_s, total_mpix / gpu_total_s)
    if args.verify:
        cpu_s = sum(row["cpu_ms"] for row in rows) / 1000.0
        log.info(
            "CPU reference %.0f Mpix/s -> overall speed-up x%.1f "
            "(end-to-end) / x%.1f (kernels only)", total_mpix / cpu_s,
            cpu_s / gpu_total_s, cpu_s / kernel_s)
        worst = max(row["max_abs_diff"] for row in rows)
        log.info(
            "Verification: worst per-pixel difference vs CPU = %d "
            "gray level(s); per-stage tolerances %s -> %s", worst,
            pipeline.VERIFY_TOLERANCE, "PASS" if all_passed else "FAIL")
    log.info("Metrics written to %s", csv_path)
    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
