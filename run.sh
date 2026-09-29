#!/usr/bin/env bash
# End-to-end demo: build the dataset, run the GPU pipeline with CPU
# verification + benchmarking, run the unit tests and render the figures.
# Works in Linux (Coursera lab), macOS-less CUDA hosts and Git Bash on Windows.
#
# Usage: ./run.sh [extra CLI args, e.g. --sigma 2.0 --threshold 60]
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# Keep NumPy's BLAS from grabbing one buffer per core on small-RAM machines.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

INPUT_DIR="${INPUT_DIR:-data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"

if [ -z "$(ls -A "$INPUT_DIR" 2>/dev/null)" ]; then
  echo "== Generating synthetic dataset in $INPUT_DIR"
  "$PYTHON" scripts/generate_images.py --output "$INPUT_DIR"
fi

echo "== Running GPU pipeline"
"$PYTHON" -m gpu_pipeline.cli --input "$INPUT_DIR" --output "$OUTPUT_DIR" \
  --benchmark --outputs gray,equalized,blurred,edges "$@"

echo "== Running unit tests"
"$PYTHON" -m unittest discover -s tests -v 2>&1 | tee "$OUTPUT_DIR/tests.log"

echo "== Rendering figures"
"$PYTHON" scripts/make_report.py --metrics "$OUTPUT_DIR/metrics.csv" \
  --input "$INPUT_DIR" --output "$OUTPUT_DIR" --figures docs/figures
