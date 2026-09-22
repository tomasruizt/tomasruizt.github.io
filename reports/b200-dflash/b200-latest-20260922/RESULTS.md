# B200 release comparison — DFlash block 16

[Open the tabbed HTML report](RESULTS.html).

Each table's Logs row links to Server logs and Bench logs, metric JSON exports, and configurations by concurrency. Bench also documents the source fields and formulas. Share the HTML, benchmark-logs.html, and model directories together.

- **Releases:** vLLM 0.30.0 (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); SGLang 0.5.20. Latest published releases checked on 2026-09-22.
- **PR2:** head `a8db1fe32ac19c2296bc0e9faedf9550ee56dd2d` merged with vLLM 0.30.0 as `63dc18cf5a93f69be959b2d2f3c26109ac693766`. Only PR2's four files differ; compiled kernels come from the release wheel.

- **Hardware and precision:** one NVIDIA B200 per run, TP=1, FP8 weights, BF16 compute and BF16 Mamba convolution/SSM states.
- **Server limits:** prefix caching disabled; memory fraction 0.92; 32 request slots; prefill chunk 2,048; maximum context 32,768.
- **CUDA graphs:** vLLM uses MRV2 and capture sizes through 512 tokens; SGLang uses its release-default graph policy.
- **DFlash:** 15 proposed tokens (SGLang block size 16); ReplaySSM is not enabled explicitly.
- **Acceptance length:** includes the bonus token. vLLM/PR use 1 + accepted draft tokens / draft iterations; SGLang uses the mean sampled acceptance-length gauge, with different weighting. Baselines are not applicable (—).
- **MoE comparison:** Qwen3.6-35B-A3B uses the same workload and 15 proposals as this matrix. The PR's original H200 experiment used ShareGPT and 8 proposals; this is not an exact reproduction.
- **MoE backend:** both SGLang MoE variants use Triton because release-default TRTLLM failed with missing FP8 input scales in the original block-16 sweep.
- **Draft backend:** vLLM and PR2 explicitly use `TRITON_ATTN`; SGLang also selects Triton. The original block-16 sweep's vLLM release-default FlashAttention draft backend failed on B200 with `No common block size for 336/480`.

- **Workload:** GSM8K, up to 256 output tokens, EOS and model sampling defaults enabled. Dense-model checkpoints match the pinned H100 experiments; all target/draft revisions are recorded per run.
- **Requests/warmups:** 100/10 at c=1, otherwise 20c/2c. Each variant reuses one server across ascending concurrency levels.
- **Execution:** variants run concurrently on separate reserved GPUs on a shared host, using up to seven benchmark GPUs.
- **Scope:** n=3 independently restarted runs per configuration; tables show means. Latencies average per-run percentiles: TPOT p90, ITL and TTFT p99, not pooled percentiles. Plot coordinates average each run's throughput and 1,000 / TPOT p90 separately; error bars show ±1 sample SD.
- **Concurrency caveat:** C=1 measures GSM8K prompt indices 10–109, whereas C=2 measures 4–43 (zero-based), because request and warmup counts differ. These curves retain the measured workloads; n=3 does not correct the prompt-mix confound. Small request counts limit p99 reliability.

[SGLang verification findings](SGLANG_VERIFICATION.md).

## 27B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120.00 | 463.29 | 132.90 | 589.90 | 513.04 |
| 2 | 227.68 | 769.27 | 261.40 | 1,071.07 | 856.87 |
| 4 | 450.12 | 1,274.29 | 508.82 | 1,691.91 | 1,391.58 |
| 8 | 880.27 | 1,870.76 | 976.46 | 2,523.63 | 2,148.68 |
| 16 | 1,695.01 | 2,599.91 | 1,869.05 | 3,240.05 | 3,181.93 |
| 32 | 3,080.62 | 3,207.61 | 3,326.50 | 4,632.60 | 3,986.62 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

## 27B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.44 | 20.47 | 11.00 | 15.08 | 13.97 |
| 2 | 9.86 | 83.84 | 10.97 | 39.71 | 69.83 |
| 4 | 9.57 | 81.57 | 11.65 | 46.56 | 70.24 |
| 8 | 9.68 | 87.78 | 11.60 | 50.96 | 72.83 |
| 16 | 10.26 | 90.16 | 12.36 | 89.95 | 74.99 |
| 32 | 11.94 | 118.15 | 14.46 | 77.09 | 97.66 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

## 27B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 136.04 | 88.52 | 39.79 | 51.00 | 76.72 |
| 2 | 127.41 | 152.35 | 71.83 | 82.35 | 134.78 |
| 4 | 94.66 | 178.94 | 76.48 | 103.06 | 171.49 |
| 8 | 165.41 | 228.33 | 76.49 | 105.84 | 198.23 |
| 16 | 117.40 | 254.47 | 90.14 | 124.66 | 211.56 |
| 32 | 211.14 | 339.66 | 146.90 | 447.87 | 275.07 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

## 27B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.17 | 2.41 | 7.41 | 1.92 | 2.09 |
| 2 | 8.49 | 2.59 | 7.46 | 2.01 | 2.39 |
| 4 | 8.61 | 3.21 | 7.64 | 2.55 | 3.03 |
| 8 | 8.79 | 4.80 | 7.98 | 3.60 | 4.17 |
| 16 | 9.13 | 6.98 | 8.28 | 5.92 | 5.78 |
| 32 | 9.95 | 11.49 | 8.80 | 7.95 | 9.19 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

## 27B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 7.61 | — | 7.46 | 7.59 |
| 2 | — | 7.78 | — | 7.72 | 7.76 |
| 4 | — | 7.66 | — | 7.57 | 7.60 |
| 8 | — | 7.45 | — | 7.63 | 7.48 |
| 16 | — | 7.66 | — | 7.73 | 7.62 |
| 32 | — | 7.70 | — | 7.64 | 7.70 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

## 27B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 2,134,931.00 | 1,139,046.67 | 2,140,480.00 | 1,071,151.00 | 1,134,165.00 |
| 128K seq equivalents (est.) | 16.29 | 8.69 | 16.33 | 8.17 | 8.65 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Server](27B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#27B-vllm_baseline) | [Server](27B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-vllm_dflash) | [Server](27B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#27B-sglang_baseline) | [Server](27B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#27B-sglang_dflash) | [Server](27B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#27B-pr2_dflash) |

128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run.

## 4B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 387.81 | 643.37 | 406.50 | 1,095.36 | 844.91 |
| 2 | 716.93 | 1,092.84 | 790.32 | 1,819.28 | 1,472.87 |
| 4 | 1,415.99 | 1,835.17 | 1,538.62 | 3,097.50 | 2,383.93 |
| 8 | 2,715.41 | 2,845.69 | 2,942.68 | 4,291.27 | 3,665.64 |
| 16 | 5,036.88 | 4,256.38 | 5,584.93 | 5,765.82 | 5,351.84 |
| 32 | 8,860.49 | 5,675.74 | 9,638.42 | 9,442.54 | 6,651.20 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

## 4B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.41 | 9.70 | 3.22 | 8.24 | 7.87 |
| 2 | 3.40 | 45.98 | 3.36 | 22.69 | 36.65 |
| 4 | 3.62 | 46.68 | 3.54 | 23.77 | 37.98 |
| 8 | 3.82 | 48.01 | 4.02 | 30.66 | 39.49 |
| 16 | 5.01 | 50.93 | 5.11 | 47.27 | 40.36 |
| 32 | 5.58 | 58.33 | 7.81 | 33.10 | 52.26 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

## 4B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.29 | 52.94 | 33.46 | 33.04 | 44.53 |
| 2 | 60.88 | 96.51 | 52.05 | 56.47 | 73.93 |
| 4 | 60.08 | 105.88 | 51.55 | 61.08 | 108.79 |
| 8 | 79.86 | 131.84 | 49.96 | 70.96 | 110.22 |
| 16 | 92.33 | 144.38 | 49.93 | 73.71 | 122.79 |
| 32 | 142.11 | 187.99 | 134.55 | 208.43 | 171.13 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

## 4B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.47 | 1.95 | 2.36 | 1.12 | 1.43 |
| 2 | 2.59 | 2.14 | 2.38 | 1.36 | 1.56 |
| 4 | 2.63 | 2.74 | 2.45 | 1.51 | 1.96 |
| 8 | 2.71 | 3.42 | 2.56 | 2.33 | 2.70 |
| 16 | 2.88 | 4.88 | 2.71 | 3.65 | 3.84 |
| 32 | 3.20 | 7.29 | 2.96 | 4.36 | 6.37 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

## 4B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 5.63 | — | 5.60 | 5.75 |
| 2 | — | 5.64 | — | 5.29 | 5.57 |
| 4 | — | 5.55 | — | 5.30 | 5.49 |
| 8 | — | 5.50 | — | 5.22 | 5.52 |
| 16 | — | 5.62 | — | 5.59 | 5.57 |
| 32 | — | 5.60 | — | 5.54 | 5.64 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

## 4B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 5,036,229.67 | 2,856,496.00 | 5,128,384.00 | 2,674,702.00 | 2,856,496.00 |
| 128K seq equivalents (est.) | 38.42 | 21.79 | 39.13 | 20.41 | 21.79 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Server](4B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#4B-vllm_baseline) | [Server](4B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-vllm_dflash) | [Server](4B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#4B-sglang_baseline) | [Server](4B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#4B-sglang_dflash) | [Server](4B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#4B-pr2_dflash) |

128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run.

## 35B-A3B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 240.47 | 602.46 | 305.11 | 791.12 | 703.50 |
| 2 | 428.87 | 1,035.72 | 552.21 | 1,120.56 | 1,152.14 |
| 4 | 806.22 | 1,569.57 | 1,010.17 | 1,921.25 | 1,797.60 |
| 8 | 1,371.03 | 2,398.98 | 1,623.04 | 2,823.40 | 2,685.02 |
| 16 | 2,312.08 | 3,448.68 | 2,728.21 | 3,924.49 | 3,805.00 |
| 32 | 3,933.29 | 3,125.55 | 4,127.09 | 6,673.55 | 3,448.48 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.07 | 12.79 | 5.57 | 12.01 | 10.35 |
| 2 | 5.30 | 54.83 | 8.08 | 31.55 | 44.31 |
| 4 | 5.70 | 58.92 | 8.08 | 34.50 | 46.08 |
| 8 | 6.38 | 58.32 | 9.29 | 40.26 | 52.65 |
| 16 | 7.71 | 63.92 | 9.71 | 70.85 | 62.08 |
| 32 | 9.69 | 120.23 | 12.37 | 45.20 | 116.30 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50.54 | 55.01 | 44.32 | 47.66 | 58.94 |
| 2 | 79.59 | 104.06 | 70.48 | 1,239.56 | 102.10 |
| 4 | 76.35 | 162.99 | 62.75 | 71.93 | 102.28 |
| 8 | 887.15 | 162.78 | 65.41 | 79.78 | 144.89 |
| 16 | 172.82 | 409.66 | 61.99 | 95.94 | 196.54 |
| 32 | 215.12 | 334.46 | 89.53 | 293.53 | 332.30 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.02 | 1.93 | 3.17 | 1.48 | 1.59 |
| 2 | 4.44 | 2.16 | 3.42 | 1.78 | 1.89 |
| 4 | 4.75 | 2.82 | 3.78 | 2.38 | 2.46 |
| 8 | 5.38 | 3.95 | 4.76 | 3.40 | 3.50 |
| 16 | 6.49 | 5.61 | 5.71 | 5.12 | 5.04 |
| 32 | 7.67 | 12.59 | 7.32 | 5.64 | 11.76 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 6.45 | — | 6.49 | 6.47 |
| 2 | — | 6.68 | — | 6.80 | 6.74 |
| 4 | — | 6.62 | — | 6.44 | 6.65 |
| 8 | — | 6.50 | — | 6.48 | 6.49 |
| 16 | — | 6.51 | — | 6.64 | 6.64 |
| 32 | — | 6.56 | — | 6.60 | 6.56 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 6,368,085.33 | 1,967,364.33 | 6,697,728.00 | 2,648,553.00 | 1,967,364.33 |
| 128K seq equivalents (est.) | 48.59 | 15.01 | 51.10 | 20.21 | 15.01 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Server](35B-A3B/vllm_baseline/vllm_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_baseline) | [Server](35B-A3B/vllm_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-vllm_dflash) | [Server](35B-A3B/sglang_baseline/sglang_baseline/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_baseline) | [Server](35B-A3B/sglang_dflash/sglang_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-sglang_dflash) | [Server](35B-A3B/pr2_dflash/vllm_dflash/server.log) · [Bench](benchmark-logs.html#35B-A3B-pr2_dflash) |

128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run.

- **Completed:** 270/270 benchmark points. A dash indicates an unavailable or inapplicable metric.
- **Artifacts:** per-run commands, versions, GPU identifiers, metrics and logs are in the model/variant directories.
- **Environment:** see `environment.json` for dependency versions and source revisions.

## 27B throughput vs interactivity

![27B throughput versus interactivity](plots/27b-throughput-interactivity.svg)

Coordinates: mean throughput and mean per-run 1,000 / TPOT p90 (ms); error bars: ±1 sample SD, n=3. Dashed curves show the baselines.

## 4B throughput vs interactivity

![4B throughput versus interactivity](plots/4b-throughput-interactivity.svg)

Coordinates: mean throughput and mean per-run 1,000 / TPOT p90 (ms); error bars: ±1 sample SD, n=3. Dashed curves show the baselines.

## 35B-A3B throughput vs interactivity

![35B-A3B throughput versus interactivity](plots/35b-a3b-throughput-interactivity.svg)

Coordinates: mean throughput and mean per-run 1,000 / TPOT p90 (ms); error bars: ±1 sample SD, n=3. Dashed curves show the baselines.
