# Convenience targets. The CUDA kernels are compiled at run time by NVRTC,
# so "build" only checks the kernels compile for the local GPU.
PYTHON ?= python3
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS ?= 1
INPUT_DIR ?= data/input
OUTPUT_DIR ?= data/output
ARGS ?=

.PHONY: all install build data sipi run test report clean

all: build run

install:
	$(PYTHON) -m pip install -r requirements.txt

build:
	$(PYTHON) -c "from gpu_pipeline import cuda_driver, pipeline; \
	d = cuda_driver.CudaDevice(); \
	g = pipeline.GpuPipeline(d, pipeline.PipelineParams()); \
	print('Compiled for', d.name, 'in %.1f ms' % g.compile_ms); g.close(); d.close()"

data:
	$(PYTHON) scripts/generate_images.py --output $(INPUT_DIR)

sipi:
	$(PYTHON) scripts/fetch_sipi.py --output $(INPUT_DIR)

run:
	$(PYTHON) -m gpu_pipeline.cli --input $(INPUT_DIR) --output $(OUTPUT_DIR) \
		--benchmark --outputs gray,equalized,blurred,edges $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v

report:
	$(PYTHON) scripts/make_report.py --metrics $(OUTPUT_DIR)/metrics.csv \
		--input $(INPUT_DIR) --output $(OUTPUT_DIR) --figures docs/figures

clean:
	rm -rf $(OUTPUT_DIR) docs/figures
