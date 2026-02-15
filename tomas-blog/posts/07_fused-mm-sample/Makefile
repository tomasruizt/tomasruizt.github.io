# Copy benchmark plots from fused-mm-sample repo into blog imgs/ directory.
#
# Usage:
#   make copy-imgs                        # copy all benchmark plots
#   FMMS_REPO=~/code/fused-mm-sample make copy-imgs   # override repo path

FMMS_REPO ?= $(HOME)/code/fused-mm-sample
BENCH      := $(FMMS_REPO)/benchmarking/modal-results
IMGS       := imgs

# GPUs included in the blog (strongest to weakest).
GPUS := b300 b200 h200 h100

# Triton-bench source directories per GPU.
# h100 results live in "triton-bench-h100!" (with exclamation mark).
triton_dir = $(BENCH)/triton-bench-$(if $(filter h100,$(1)),h100!,$(1))/custom-plots

# Per-GPU plot types and their source/target filenames.
#   source-name.png -> target-name-{gpu}.png
TRITON_PLOTS := \
	batch-scaling-large:batch-scaling-large \
	relative-performance-vs-pytorch-large:relative-perf-vs-pytorch \
	relative-performance-vs-flashinfer-large:relative-perf-vs-flashinfer \
	roofline-large:roofline-large \
	memory-throughput-large:memory-throughput-large

# vLLM plots (GPU-independent).
VLLM_DIR := $(BENCH)/vllm-bench-b200/imgs
VLLM_PLOTS := tpot_vs_concurrency.png speedup_vs_concurrency.png

.PHONY: copy-imgs
copy-imgs:
	@echo "Copying triton-bench plots..."
	@$(foreach gpu,$(GPUS),\
		$(foreach plot,$(TRITON_PLOTS),\
			cp "$(call triton_dir,$(gpu))/$(word 1,$(subst :, ,$(plot))).png" \
			   "$(IMGS)/$(word 2,$(subst :, ,$(plot)))-$(gpu).png" && \
		) \
	) true
	@echo "Copying vLLM plots..."
	@$(foreach plot,$(VLLM_PLOTS),\
		cp "$(VLLM_DIR)/$(plot)" "$(IMGS)/$(plot)" && \
	) true
	@echo "Done. Copied plots for GPUs: $(GPUS)"
