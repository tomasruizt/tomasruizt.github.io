# Replication Plan: AJR (2001), "The Colonial Origins of Comparative Development"

**Goal:** Reproduce the paper's core plots and regressions in Python, approximately.

**Paper:** `acemoglu-2001-colonial-origins.pdf` (AER 91(5), pp. 1369–1401)

## Core idea
Institutions cause prosperity. Settler mortality → settlement strategy → institutions → income today.
Settler mortality is the **instrument** for institutions (expropriation risk).

## Variables (base sample, ~64 countries)
- `logpgp95` — log GDP per capita, 1995 (PPP)
- `avexpr` — avg. protection against expropriation risk, 1985–95 (institutions)
- `logem4` — log settler mortality (instrument)

## What we replicate
| # | Target | Method | Tool |
|---|--------|--------|------|
| 1 | **Figure 1** — log GDP vs. settler mortality (reduced form) | scatter + fit | matplotlib |
| 2 | **Figure 2** — log GDP vs. expropriation risk (OLS) | scatter + fit | matplotlib |
| 3 | **Figure 3** — expropriation risk vs. settler mortality (first stage) | scatter + fit | matplotlib |
| 4 | **Table 1** — summary statistics | descriptive | pandas |
| 5 | **Table 2** — OLS: `logpgp95 ~ avexpr` | OLS | statsmodels |
| 6 | **Table 4** — 2SLS: `logpgp95 ~ avexpr`, instrument `logem4` (9 cols, 3 panels) | IV / 2SLS | linearmodels.IV2SLS |

**Headline number to match:** 2SLS coefficient on `avexpr` ≈ 0.94 [paper: 0.94 (0.16)].

Note: the public `maketable4.dta` lacks the `other` continent dummy needed for
cols (7)–(8); it can be merged in from `maketable2.dta` on `shortnam`.

## Steps
1. Get data — AJR replication dataset (`maketable*.dta`, mirrored online).
2. Load + filter to base sample.
3. Build the 3 figures.
4. Run OLS (Table 2) and 2SLS (Table 4); compare coefficients to paper.

## Stack
`pandas`, `numpy`, `matplotlib`, `statsmodels`, `linearmodels`
