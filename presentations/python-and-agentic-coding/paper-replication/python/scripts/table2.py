"""
Replicate AJR (2001) Table 2: OLS Regressions.

Eight specifications regressing income on protection against expropriation risk,
progressively adding latitude and continent dummies, across the whole-world and
base samples (and, in cols 7-8, using log output per worker as the outcome).
Mirrors the layout on p. 1379. America is the omitted continent dummy.

Known discrepancy vs the printed paper (whole-world columns):
  - The published whole-world N is 110; reproductions get 111. AJR's own
    maketable2.do notes that one observation was *mistakenly dropped* from the
    printed Table 2, so 111 is the corrected sample. We do not try to match 110.
  - The whole-world continent-dummy coefficients (col 4) differ from the printed
    values because the public maketable2.dta ships a residual `other` dummy
    (~AUS/MLT/NZL) rather than AJR's internal classification. We use the dummies
    as provided -- this matches the canonical QuantEcon reproduction exactly
    (avexpr 0.39, africa -0.92, R^2 0.72, N 111). Do NOT recode Europe into
    `other`; that is not what standard reproductions do.
  See https://python.quantecon.org/ols.html (reproduces this same dataset).
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).parent.parent  # paper-replication/python/
DATA = ROOT.parent / "data"          # shared data lives at the repl root
df = pd.read_stata(DATA / "maketable2.dta")

# (label, outcome, regressors, base_sample_only)
SPECS = [
    ("(1)", "logpgp95", ["avexpr"], False),
    ("(2)", "logpgp95", ["avexpr"], True),
    ("(3)", "logpgp95", ["avexpr", "lat_abst"], False),
    ("(4)", "logpgp95", ["avexpr", "lat_abst", "asia", "africa", "other"], False),
    ("(5)", "logpgp95", ["avexpr", "lat_abst"], True),
    ("(6)", "logpgp95", ["avexpr", "lat_abst", "asia", "africa", "other"], True),
    ("(7)", "loghjypl", ["avexpr"], False),
    ("(8)", "loghjypl", ["avexpr"], True),
]

# Rows of the table, in paper order.
TERMS = [
    ("avexpr", "Average protection against expropriation risk, 1985-1995"),
    ("lat_abst", "Latitude"),
    ("asia", "Asia dummy"),
    ("africa", "Africa dummy"),
    ("other", '"Other" continent dummy'),
]

results = {}
for label, outcome, regressors, base_only in SPECS:
    data = df[df["baseco"] == 1] if base_only else df
    formula = f"{outcome} ~ " + " + ".join(regressors)
    fit = smf.ols(formula, data=data).fit()
    results[label] = fit


def cell(fit, term):
    if term not in fit.params:
        return ""
    return f"{fit.params[term]:.2f} ({fit.bse[term]:.2f})"


table = {label: [cell(fit, term) for term, _ in TERMS] for label, fit in results.items()}
for label, fit in results.items():
    table[label].append(f"{fit.rsquared:.2f}")
    table[label].append(str(int(fit.nobs)))

index = [lbl for _, lbl in TERMS] + ["R-squared", "Number of observations"]
out = pd.DataFrame(table, index=index)

# Header note: which outcome / sample each column uses.
outcomes = {label: ("logGDPpc95" if oc == "logpgp95" else "logOutput/worker88")
            for label, oc, _, _ in SPECS}
samples = {label: ("base" if b else "world") for label, _, _, b in SPECS}

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
report = "\n".join([
    "TABLE 2 — OLS REGRESSIONS",
    "(coefficient; standard error in parentheses)",
    "outcome: " + "  ".join(f"{k}={v}" for k, v in outcomes.items()),
    "sample:  " + "  ".join(f"{k}={v}" for k, v in samples.items()),
    "",
    out.to_string(),
]) + "\n"
print(report)

outpath = ROOT / "tables" / "table2.txt"
outpath.write_text(report)
print(f"saved {outpath}")
