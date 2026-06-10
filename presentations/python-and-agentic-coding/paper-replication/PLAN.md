# Replication Plan: AJR (2001), "The Colonial Origins of Comparative Development"

**Goal:** Reproduce the paper's core plots and regressions in Python, approximately.

**Paper:** `acemoglu-2001-colonial-origins.pdf` (AER 91(5), pp. 1369–1401)

## Core idea
Institutions cause prosperity. Settler mortality → settlement strategy → institutions → income today.

## What we replicate
| # | Target | Method |
|---|--------|--------|
| 1 | **Figure 1** — log GDP vs. settler mortality (reduced form) | scatter + fit |
| 2 | **Figure 2** — log GDP vs. expropriation risk (OLS) | scatter + fit |
| 3 | **Figure 3** — expropriation risk vs. settler mortality (first stage) | scatter + fit |
| 4 | **Table 1** — summary statistics | descriptive |
| 5 | **Table 2** — OLS | OLS |
| 6 | **Table 4** — 2SLS (9 cols, 3 panels) | IV / 2SLS |

## Steps
1. Get data
2. Build the 3 figures.
3. Build summary (Table 1).
4. Run OLS (Table 2) and 2SLS (Table 4)

## Stack
`pandas`, `numpy`, `matplotlib`, `statsmodels`, `linearmodels`
