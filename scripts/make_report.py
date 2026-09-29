"""Turns metrics.csv into charts and a before/after montage for the README.

Usage:
    python scripts/make_report.py --metrics data/output/metrics.csv \
        --input data/input --output data/output --figures docs/figures
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from PIL import Image  # pylint: disable=wrong-import-position

STAGES = ("h2d", "gray", "histogram", "lut", "equalize", "blur_rows",
          "blur_cols", "sobel", "d2h")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        for key, value in row.items():
            if key != "image" and value != "":
                row[key] = float(value)
    return sorted(rows, key=lambda row: row["megapixels"])


def plot_timing(rows, path):
    mpix = np.array([row["megapixels"] for row in rows])
    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_time.loglog(mpix, [r["cpu_ms"] for r in rows], "o-", label="CPU (NumPy)")
    ax_time.loglog(mpix, [r["gpu_total_ms"] for r in rows], "s-",
                   label="GPU end-to-end (incl. PCIe copies)")
    ax_time.loglog(mpix, [r["gpu_kernel_ms"] for r in rows], "^-",
                   label="GPU kernels only")
    ax_time.set_xlabel("Image size (megapixels)")
    ax_time.set_ylabel("Time per image (ms)")
    ax_time.set_title("Pipeline time vs image size")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()

    ax_speed.semilogx(mpix, [r["speedup_total"] for r in rows], "s-",
                      label="end-to-end")
    ax_speed.semilogx(mpix, [r["speedup_kernel"] for r in rows], "^-",
                      label="kernels only")
    ax_speed.set_xlabel("Image size (megapixels)")
    ax_speed.set_ylabel("Speed-up over CPU (x)")
    ax_speed.set_title("GPU speed-up vs image size")
    ax_speed.grid(True, which="both", alpha=0.3)
    ax_speed.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stages(rows, path):
    largest = rows[-1]
    values = [largest[f"{stage}_ms"] for stage in STAGES]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(STAGES[::-1], values[::-1], color="#4c72b0")
    ax.bar_label(bars, fmt="%.3f ms", padding=3, fontsize=8)
    ax.set_xlabel("GPU time (ms)")
    ax.set_title(f"Per-stage GPU time, {largest['image']} "
                 f"({largest['megapixels']:.1f} MP)")
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_montage(stems, input_dir, output_dir, path, tile=320):
    columns = ("input", "equalized", "edges")
    rows = []
    for stem in stems:
        inputs = [f for f in os.listdir(input_dir)
                  if os.path.splitext(f)[0] == stem]
        images = [Image.open(os.path.join(input_dir, inputs[0])).convert("RGB")]
        for kind in columns[1:]:
            images.append(Image.open(os.path.join(
                output_dir, f"{stem}_{kind}.png")).convert("RGB"))
        rows.append([image.resize((tile, int(tile * image.height /
                                              image.width)))
                     for image in images])
    height = sum(row[0].height for row in rows)
    sheet = Image.new("RGB", (tile * len(columns), height), "white")
    y = 0
    for row in rows:
        for x, image in enumerate(row):
            sheet.paste(image, (x * tile, y))
        y += row[0].height
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", default="data/output/metrics.csv")
    parser.add_argument("--input", default="data/input")
    parser.add_argument("--output", default="data/output")
    parser.add_argument("--figures", default="docs/figures")
    parser.add_argument("--montage-count", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.figures, exist_ok=True)
    rows = load_rows(args.metrics)
    if "cpu_ms" in rows[0] and rows[0]["cpu_ms"] != "":
        plot_timing(rows, os.path.join(args.figures, "timing.png"))
    plot_stages(rows, os.path.join(args.figures, "stages.png"))
    picks = np.linspace(0, len(rows) - 1, args.montage_count).astype(int)
    stems = [os.path.splitext(rows[i]["image"])[0] for i in picks]
    make_montage(stems, args.input, args.output,
                 os.path.join(args.figures, "montage.png"))
    print(f"figures written to {args.figures}")


if __name__ == "__main__":
    main()
