PYTHON := /home/tomasruiz/code/FlashSampling/.venv/bin/python
NSYS := nsys profile -c cudaProfilerApi --capture-range-end=stop --force-overwrite=true
WARMUP_ITERS ?= 25
ACTIVE_ITERS ?= 10
BATCH_SIZE ?= 32
N_PARAMS ?= 4096
TRAIN_ARGS := --warmup-iters $(WARMUP_ITERS) --active-iters $(ACTIVE_ITERS) --batch-size $(BATCH_SIZE) --n-params $(N_PARAMS)

.PHONY: nsys nsys-sync-free rm-nsys

nsys:
	$(NSYS) -o sync_trace $(PYTHON) train.py $(TRAIN_ARGS)

nsys-sync-free:
	$(NSYS) -o sync_free_trace $(PYTHON) train.py --no-print $(TRAIN_ARGS)

rm-nsys:
	rm -f ./*.nsys-rep ./*.sqlite
