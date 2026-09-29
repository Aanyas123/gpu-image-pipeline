"""Generates a reproducible synthetic test set of RGB images.

The images mix smooth gradients, geometric shapes, a low-contrast region and
sensor-like noise, so every pipeline stage has something to do: histogram
equalization stretches the low-contrast band, the blur suppresses noise and
Sobel picks out the shape boundaries. Sizes range from 256x256 to 4096x4096
to show how GPU speed-up scales with problem size.

Usage:
    python scripts/generate_images.py --output data/input --count 24 --seed 7
"""

import argparse
import os

import numpy as np
from PIL import Image

# (width, height) pairs; the list is cycled when --count exceeds its length.
SIZES = ((256, 256), (512, 512), (640, 480), (800, 600), (1024, 768),
         (1280, 720), (1024, 1024), (1600, 1200), (1920, 1080), (2048, 1536),
         (2560, 1440), (3840, 2160), (4096, 4096))


def make_image(width, height, rng):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = xx / width, yy / height
    image = np.empty((height, width, 3), dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=3)
    for channel in range(3):
        image[..., channel] = 90 + 40 * np.sin(2 * np.pi * (
            u * rng.uniform(0.5, 2) + v * rng.uniform(0.5, 2)) + phase[channel])

    for _ in range(rng.integers(6, 14)):
        color = rng.uniform(20, 235, size=3)
        cx, cy = rng.uniform(0.1, 0.9, size=2)
        size = rng.uniform(0.05, 0.2)
        if rng.random() < 0.5:
            mask = (u - cx)**2 + ((v - cy) * height / width)**2 < size**2
        else:
            mask = (np.abs(u - cx) < size) & (np.abs(v - cy) < size * 0.6)
        image[mask] = color

    # A washed-out band that histogram equalization should recover.
    band = (v > 0.7) & (v < 0.85)
    image[band] = 110 + 0.15 * (image[band] - 110)

    image += rng.normal(0, 6, size=image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output",
                        default="data/input",
                        help="Directory to write PNGs to.")
    parser.add_argument("--count",
                        type=int,
                        default=len(SIZES),
                        help=f"Number of images (default: {len(SIZES)}).")
    parser.add_argument("--seed",
                        type=int,
                        default=7,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for index in range(args.count):
        width, height = SIZES[index % len(SIZES)]
        path = os.path.join(args.output,
                            f"synthetic_{index:02d}_{width}x{height}.png")
        Image.fromarray(make_image(width, height, rng)).save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
