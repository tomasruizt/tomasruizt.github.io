"""
Replicate AJR (2001) Table 1: Descriptive Statistics.

Reports mean (with standard deviation in parentheses) for each variable across
columns: whole world, base sample, and base sample split by quartiles of
(potential) settler mortality. Mirrors the layout on p. 1377.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent  # paper-replication/

df = pd.read_stata(ROOT / "data" / "maketable1.dta")
# euro1900 is stored 0-100; the paper reports it as a 0-1 fraction.
df["euro1900"] = df["euro1900"] / 100

# (var, label, na_in_whole_world)
ROWS = [
    ("logpgp95", "Log GDP per capita (PPP) in 1995", False),
    ("loghjypl", "Log output per worker in 1988", False),
    ("avexpr", "Average protection against expropriation risk, 1985-1995", False),
    ("cons90", "Constraint on executive in 1990", False),
    ("cons00a", "Constraint on executive in 1900", False),
    ("cons1", "Constraint on executive in first year of independence", False),
    ("democ00a", "Democracy in 1900", False),
    ("euro1900", "European settlements in 1900", False),
    ("logem4", "Log European settler mortality", True),  # n.a. for whole world
]

# Base sample, split by quartiles of raw settler mortality (extmort4).
# Cutoffs follow the paper's Table 1 notes (65.4 / 78.1 / 280). Note: five
# countries are stored as the float32 value of 78.1 (78.0999984741211); the
# paper assigns them to quartile (2), so we put the Q2/Q3 boundary at 78.15 —
# in the gap before the next group at 78.2 — to reproduce the paper's 14/18/17/15.
base = df[df["baseco"] == 1]
CUTS = [(None, 65.4), (65.4, 78.15), (78.15, 280), (280, None)]
quartiles = []
for lo, hi in CUTS:
    m = base["extmort4"]
    mask = pd.Series(True, index=base.index)
    if lo is not None:
        mask &= m >= lo
    if hi is not None:
        mask &= m < hi
    quartiles.append(base[mask])

columns = [("Whole world", df), ("Base sample", base)]
columns += [(f"({i + 1})", q) for i, q in enumerate(quartiles)]


def cell(series, na):
    if na:
        return "n.a."
    s = series.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.2f} ({s.std():.2f})"


# Build a DataFrame of formatted strings, then print.
table = {}
for name, data in columns:
    na_flag = name == "Whole world"
    table[name] = [cell(data[var], na and na_flag) for var, _, na in ROWS]
# Number of observations row (total rows per column, as in the paper).
for name, data in columns:
    table[name].append(str(len(data)))

index = [label for _, label, _ in ROWS] + ["Number of observations"]
out = pd.DataFrame(table, index=index)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
print("TABLE 1 — DESCRIPTIVE STATISTICS")
print("(mean; standard deviation in parentheses; quartiles by settler mortality)\n")
print(out.to_string())

# Also save a CSV for reuse.
outpath = ROOT / "figures" / "table1.csv"
out.to_csv(outpath)
print(f"\nsaved {outpath}")
