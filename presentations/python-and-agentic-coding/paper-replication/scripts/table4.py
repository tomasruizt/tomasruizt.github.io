"""
Replicate AJR (2001) Table 4: IV Regressions of Log GDP per Capita (p. 1385).

The headline of the paper. Protection against expropriation risk (`avexpr`) is
instrumented with log settler mortality (`logem4`); the 2SLS coefficient on
`avexpr` in column (1) is the famous ~0.94.

Nine columns, three panels:
  (1) base sample
  (2) base sample + latitude
  (3) base sample without Neo-Europes (rich4 == 0)
  (4) (3) + latitude
  (5) base sample without Africa (africa == 0)
  (6) (5) + latitude
  (7) base sample + continent dummies (asia, africa, "other")
  (8) (7) + latitude
  (9) dependent variable is log output per worker (loghjypl), base sample, no controls
  Panel A: 2SLS.  Panel B: first stage (avexpr on logem4).  Panel C: OLS.
  America is the omitted continent dummy.

Data note: the public maketable4.dta does not ship the `other` continent dummy
that columns (7)-(9) need, so we merge it in from maketable2.dta on `shortnam`
(same dataset, same coding the existing table2.py already relies on).
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).parent.parent  # paper-replication/
df = pd.read_stata(ROOT / "data" / "maketable4.dta")
other = pd.read_stata(ROOT / "data" / "maketable2.dta")[["shortnam", "other"]]
df = df.merge(other, on="shortnam", how="left")

# (label, outcome, sample filter, extra exogenous controls)
SPECS = [
    ("(1)", "logpgp95", df["baseco"] == 1, []),
    ("(2)", "logpgp95", df["baseco"] == 1, ["lat_abst"]),
    ("(3)", "logpgp95", (df["baseco"] == 1) & (df["rich4"] == 0), []),
    ("(4)", "logpgp95", (df["baseco"] == 1) & (df["rich4"] == 0), ["lat_abst"]),
    ("(5)", "logpgp95", (df["baseco"] == 1) & (df["africa"] == 0), []),
    ("(6)", "logpgp95", (df["baseco"] == 1) & (df["africa"] == 0), ["lat_abst"]),
    ("(7)", "logpgp95", df["baseco"] == 1, ["asia", "africa", "other"]),
    ("(8)", "logpgp95", df["baseco"] == 1, ["lat_abst", "asia", "africa", "other"]),
    ("(9)", "loghjypl", df["baseco"] == 1, []),
]

CONTROLS = ["lat_abst", "asia", "africa", "other"]


def fit_column(outcome, mask, controls):
    cols = [outcome, "avexpr", "logem4"] + controls
    data = df.loc[mask, cols].dropna()
    exog = " + ".join(["1"] + controls)
    # Panel A: 2SLS, avexpr instrumented by logem4.
    iv = IV2SLS.from_formula(
        f"{outcome} ~ {exog} + [avexpr ~ logem4]", data
    ).fit(cov_type="unadjusted")
    # Panel B: first stage.
    first = smf.ols(f"avexpr ~ {exog} + logem4", data).fit()
    # Panel C: OLS of the outcome on avexpr (+ controls).
    ols = smf.ols(f"{outcome} ~ {exog} + avexpr", data).fit()
    return iv, first, ols, len(data)


def cell(params, bse, term):
    if term not in params:
        return ""
    return f"{params[term]:.2f} ({bse[term]:.2f})"


panelA, panelB, panelC, nobs, r2_first = {}, {}, {}, {}, {}
for label, outcome, mask, controls in SPECS:
    iv, first, ols, n = fit_column(outcome, mask, controls)
    panelA[label] = [cell(iv.params, iv.std_errors, t) for t in ["avexpr"] + CONTROLS]
    panelB[label] = [cell(first.params, first.bse, t) for t in ["logem4"] + CONTROLS]
    panelC[label] = [cell(ols.params, ols.bse, "avexpr")]
    r2_first[label] = f"{first.rsquared:.2f}"
    nobs[label] = str(n)

A_rows = ["Avg. expropriation risk (2SLS)", "Latitude", "Asia dummy",
          "Africa dummy", '"Other" continent dummy']
B_rows = ["Log settler mortality", "Latitude", "Asia dummy",
          "Africa dummy", '"Other" continent dummy', "R-squared"]

panelA_df = pd.DataFrame(panelA, index=A_rows)
for label in panelB:
    panelB[label].append(r2_first[label])
panelB_df = pd.DataFrame(panelB, index=B_rows)
panelC_df = pd.DataFrame(panelC, index=["Avg. expropriation risk (OLS)"])
nobs_df = pd.DataFrame({k: [v] for k, v in nobs.items()}, index=["Number of observations"])

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
samples = {"(1)": "base", "(2)": "base", "(3)": "no Neo-Europes",
           "(4)": "no Neo-Europes", "(5)": "no Africa", "(6)": "no Africa",
           "(7)": "base+dummies", "(8)": "base+dummies", "(9)": "loghjypl"}

lines = [
    "TABLE 4 — IV REGRESSIONS OF LOG GDP PER CAPITA",
    "(coefficient; standard error in parentheses)",
    "sample: " + "  ".join(f"{k}={v}" for k, v in samples.items()),
    "",
    "Panel A: Two-Stage Least Squares",
    panelA_df.to_string(),
    "",
    "Panel B: First Stage (avexpr on log settler mortality)",
    panelB_df.to_string(),
    "",
    "Panel C: Ordinary Least Squares",
    panelC_df.to_string(),
    nobs_df.to_string(),
    "",
    f"Headline: column (1) 2SLS coef on avexpr = {panelA['(1)'][0]}  (paper: 0.94 (0.16))",
]
report = "\n".join(lines) + "\n"
print(report)

outpath = ROOT / "tables" / "table4.txt"
outpath.write_text(report)
print(f"saved {outpath}")
