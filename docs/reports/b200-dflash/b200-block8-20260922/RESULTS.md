# B200 release comparison — DFlash block 8

[Open the tabbed HTML report](RESULTS.html).

Each table's Logs link opens server and benchmark logs, metric exports, and configurations for every repetition.

- **Releases:** vLLM 0.30.0 (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); SGLang 0.5.20. Latest published releases checked on 2026-09-22.
- **PR2:** head `a8db1fe32ac19c2296bc0e9faedf9550ee56dd2d` merged with vLLM 0.30.0 as `63dc18cf5a93f69be959b2d2f3c26109ac693766`. Only PR2's four files differ; compiled kernels come from the release wheel.

- **Hardware and precision:** one NVIDIA B200 per run, TP=1, FP8 weights, BF16 compute and BF16 Mamba convolution/SSM states.
- **Server limits:** prefix caching disabled; memory fraction 0.92; 32 request slots; prefill chunk 2,048; maximum context 32,768.
- **CUDA graphs:** vLLM uses MRV2 and capture sizes through 512 tokens; SGLang uses its release-default graph policy.
- **DFlash:** 7 proposed tokens (SGLang block size 8); ReplaySSM is not enabled explicitly.
- **Acceptance length:** includes the bonus token. vLLM/PR use 1 + accepted draft tokens / draft iterations; SGLang uses the mean sampled acceptance-length gauge, with different weighting. Baselines are not applicable (—).
- **MoE comparison:** Qwen3.6-35B-A3B uses the same workload and 7 proposals as this matrix. The PR's original H200 experiment used ShareGPT and 8 proposals; this is not an exact reproduction.
- **MoE backend:** both SGLang MoE variants use Triton because release-default TRTLLM failed with missing FP8 input scales in the original block-16 sweep.
- **Draft backend:** vLLM and PR2 explicitly use `TRITON_ATTN`; SGLang also selects Triton. The original block-16 sweep's vLLM release-default FlashAttention draft backend failed on B200 with `No common block size for 336/480`.

- **Workload:** GSM8K, up to 256 output tokens, EOS and model sampling defaults enabled. Dense-model checkpoints match the pinned H100 experiments; all target/draft revisions are recorded per run.
- **Requests/warmups:** 100/10 at c=1, otherwise 20c/2c. Each variant reuses one server across ascending concurrency levels.
- **Execution:** variants run concurrently on separate reserved GPUs on a shared host, using up to seven benchmark GPUs.
- **Scope:** n=3 independently restarted runs per configuration; tables show means. Latencies average per-run percentiles: TPOT p90, ITL and TTFT p99, not pooled percentiles. Plot coordinates average each run's throughput and 1,000 / TPOT p90 separately; error bars show ±1 sample SD.
- **Concurrency caveat:** C=1 measures GSM8K prompt indices 10–109, whereas C=2 measures 4–43 (zero-based), because request and warmup counts differ. These curves retain the measured workloads; n=3 does not correct the prompt-mix confound. Small request counts limit p99 reliability.

## 27B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 120.16 | 379.54 | 134.66 | 506.41 | 446.95 |
| 2 | 228.38 | 621.90 | 262.26 | 896.40 | 794.93 |
| 4 | 450.02 | 1,073.89 | 509.12 | 1,552.34 | 1,339.55 |
| 8 | 879.27 | 1,766.14 | 985.00 | 2,440.88 | 2,115.31 |
| 16 | 1,692.83 | 2,563.07 | 1,837.63 | 3,541.97 | 3,127.64 |
| 32 | 3,087.34 | 3,467.10 | 3,332.15 | 5,715.89 | 4,308.52 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

## 27B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.16 | 17.73 | 10.92 | 14.45 | 12.87 |
| 2 | 9.33 | 83.27 | 11.68 | 39.55 | 61.84 |
| 4 | 9.31 | 86.34 | 11.78 | 42.22 | 62.97 |
| 8 | 9.60 | 84.52 | 11.81 | 44.64 | 64.26 |
| 16 | 10.42 | 88.10 | 12.84 | 77.36 | 69.11 |
| 32 | 11.27 | 99.58 | 14.04 | 54.83 | 77.87 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

## 27B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56.72 | 77.59 | 43.18 | 50.49 | 72.66 |
| 2 | 102.97 | 171.10 | 87.71 | 82.34 | 120.49 |
| 4 | 102.67 | 222.82 | 73.06 | 94.43 | 164.35 |
| 8 | 102.77 | 245.36 | 87.66 | 96.86 | 187.67 |
| 16 | 132.96 | 253.75 | 138.65 | 109.90 | 223.67 |
| 32 | 178.67 | 297.17 | 598.07 | 410.21 | 254.03 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

## 27B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.19 | 2.83 | 7.32 | 2.08 | 2.27 |
| 2 | 8.50 | 3.22 | 7.44 | 2.28 | 2.47 |
| 4 | 8.63 | 3.78 | 7.63 | 2.68 | 3.04 |
| 8 | 8.81 | 4.82 | 7.90 | 3.52 | 4.05 |
| 16 | 9.12 | 6.71 | 8.32 | 4.94 | 5.48 |
| 32 | 9.85 | 10.07 | 8.77 | 5.80 | 8.05 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

## 27B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 5.65 | — | 5.71 | 5.65 |
| 2 | — | 5.68 | — | 5.69 | 5.68 |
| 4 | — | 5.67 | — | 5.64 | 5.68 |
| 8 | — | 5.70 | — | 5.66 | 5.63 |
| 16 | — | 5.74 | — | 5.73 | 5.74 |
| 32 | — | 5.72 | — | 5.70 | 5.72 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

## 27B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 2,134,931.00 | 1,429,326.00 | 2,140,480.00 | 1,300,975.00 | 1,429,294.67 |
| 128K seq equivalents (est.) | 16.29 | 10.90 | 16.33 | 9.93 | 10.90 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Logs](benchmark-logs.html#27B-vllm_baseline) | [Logs](benchmark-logs.html#27B-vllm_dflash) | [Logs](benchmark-logs.html#27B-sglang_baseline) | [Logs](benchmark-logs.html#27B-sglang_dflash) | [Logs](benchmark-logs.html#27B-pr2_dflash) |

128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run.

## 4B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 390.87 | 516.45 | 401.05 | 982.28 | 707.92 |
| 2 | 723.29 | 857.74 | 795.30 | 1,705.97 | 1,116.03 |
| 4 | 1,428.60 | 1,555.46 | 1,542.89 | 2,923.88 | 1,868.45 |
| 8 | 2,745.92 | 2,466.20 | 2,960.17 | 4,450.07 | 2,935.14 |
| 16 | 5,104.60 | 3,801.90 | 5,594.52 | 6,265.98 | 4,393.27 |
| 32 | 8,993.86 | 5,452.43 | 9,942.55 | 11,828.37 | 6,876.03 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

## 4B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.69 | 10.47 | 4.55 | 8.03 | 8.65 |
| 2 | 3.63 | 50.26 | 3.31 | 21.96 | 41.24 |
| 4 | 3.66 | 48.08 | 4.13 | 23.27 | 45.07 |
| 8 | 4.03 | 49.33 | 4.10 | 29.37 | 46.50 |
| 16 | 4.30 | 51.19 | 5.33 | 32.86 | 52.76 |
| 32 | 5.16 | 54.31 | 7.79 | 27.13 | 45.74 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

## 4B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46.47 | 50.33 | 54.33 | 37.51 | 52.91 |
| 2 | 65.83 | 101.40 | 63.83 | 58.61 | 96.53 |
| 4 | 64.58 | 109.70 | 63.62 | 60.53 | 105.65 |
| 8 | 80.44 | 138.81 | 54.57 | 64.63 | 122.22 |
| 16 | 86.41 | 148.63 | 58.53 | 66.57 | 150.56 |
| 32 | 129.64 | 183.27 | 89.62 | 158.11 | 190.63 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

## 4B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.44 | 2.32 | 2.38 | 1.16 | 1.66 |
| 2 | 2.56 | 2.64 | 2.35 | 1.33 | 2.02 |
| 4 | 2.61 | 2.86 | 2.44 | 1.56 | 2.44 |
| 8 | 2.69 | 3.82 | 2.55 | 2.15 | 3.15 |
| 16 | 2.87 | 5.00 | 2.70 | 3.14 | 4.44 |
| 32 | 3.15 | 7.14 | 2.96 | 3.17 | 5.65 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

## 4B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 4.66 | — | 4.67 | 4.63 |
| 2 | — | 4.60 | — | 4.57 | 4.55 |
| 4 | — | 4.60 | — | 4.43 | 4.55 |
| 8 | — | 4.58 | — | 4.58 | 4.57 |
| 16 | — | 4.64 | — | 4.59 | 4.56 |
| 32 | — | 4.62 | — | 4.55 | 4.62 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

## 4B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 5,036,758.33 | 3,383,178.00 | 5,128,384.00 | 2,795,991.00 | 3,383,638.67 |
| 128K seq equivalents (est.) | 38.43 | 25.81 | 39.13 | 21.33 | 25.82 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Logs](benchmark-logs.html#4B-vllm_baseline) | [Logs](benchmark-logs.html#4B-vllm_dflash) | [Logs](benchmark-logs.html#4B-sglang_baseline) | [Logs](benchmark-logs.html#4B-sglang_dflash) | [Logs](benchmark-logs.html#4B-pr2_dflash) |

128K = 131,072 tokens. Sequence equivalents = reported cache tokens / 131,072, not measured concurrency. These allocations come from servers configured for 32,768-token contexts and 32 request slots; reconfiguring for 128K may change capacity. No long-context measurements were run.

## 35B-A3B output tok/s

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 248.24 | 493.84 | 309.84 | 783.26 | 677.20 |
| 2 | 441.38 | 824.80 | 560.31 | 1,256.95 | 1,070.71 |
| 4 | 818.13 | 1,414.35 | 1,023.49 | 1,877.24 | 1,706.85 |
| 8 | 1,422.34 | 2,203.22 | 1,626.00 | 2,817.21 | 2,647.70 |
| 16 | 2,331.68 | 3,160.92 | 2,718.99 | 4,009.21 | 3,940.81 |
| 32 | 3,951.02 | 4,220.69 | 4,156.35 | 7,426.49 | 5,720.60 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B ITL p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.74 | 13.34 | 4.71 | 10.31 | 9.94 |
| 2 | 5.13 | 57.00 | 5.30 | 28.38 | 47.33 |
| 4 | 5.79 | 57.69 | 7.04 | 32.55 | 53.39 |
| 8 | 6.46 | 60.70 | 8.07 | 37.24 | 47.49 |
| 16 | 8.09 | 64.01 | 9.57 | 52.36 | 50.80 |
| 32 | 9.48 | 87.29 | 12.32 | 40.53 | 58.22 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B TTFT p99 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 45.43 | 63.04 | 36.95 | 43.53 | 65.18 |
| 2 | 72.75 | 116.27 | 62.70 | 68.53 | 104.01 |
| 4 | 84.32 | 147.09 | 59.00 | 68.96 | 127.92 |
| 8 | 139.46 | 163.99 | 80.79 | 73.69 | 141.07 |
| 16 | 157.36 | 422.04 | 70.57 | 83.92 | 185.14 |
| 32 | 214.52 | 286.84 | 100.52 | 273.40 | 215.05 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B TPOT p90 (ms)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.90 | 2.22 | 3.12 | 1.38 | 1.57 |
| 2 | 4.31 | 2.47 | 3.41 | 1.68 | 1.84 |
| 4 | 4.66 | 3.03 | 3.74 | 2.25 | 2.54 |
| 8 | 5.30 | 3.97 | 4.75 | 3.19 | 3.26 |
| 16 | 6.43 | 5.67 | 5.77 | 4.63 | 4.54 |
| 32 | 7.54 | 8.93 | 7.30 | 4.67 | 6.30 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B Acceptance length (including bonus)

| c | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | 5.25 | — | 5.19 | 5.25 |
| 2 | — | 5.28 | — | 5.25 | 5.27 |
| 4 | — | 5.13 | — | 5.18 | 5.18 |
| 8 | — | 5.19 | — | 5.19 | 5.24 |
| 16 | — | 5.19 | — | 5.16 | 5.20 |
| 32 | — | 5.19 | — | 5.18 | 5.20 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

## 35B-A3B Memory & capacity

| Metric | vLLM | vLLM DFlash | SGLang | SGLang DFlash | vLLM + DFlash PR 52297 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported cache tokens | 6,380,032.00 | 2,850,782.00 | 6,697,728.00 | 2,841,513.00 | 2,850,865.67 |
| 128K seq equivalents (est.) | 48.68 | 21.75 | 51.10 | 21.68 | 21.75 |
| Configured request limit | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Logs | [Logs](benchmark-logs.html#35B-A3B-vllm_baseline) | [Logs](benchmark-logs.html#35B-A3B-vllm_dflash) | [Logs](benchmark-logs.html#35B-A3B-sglang_baseline) | [Logs](benchmark-logs.html#35B-A3B-sglang_dflash) | [Logs](benchmark-logs.html#35B-A3B-pr2_dflash) |

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
