"""
Replicate AJR (2001) Figure 1:
Reduced-form relationship between income and settler mortality.

x = log settler mortality (logem4)
y = log GDP per capita, 1995, PPP (logpgp95)
country-code labels + OLS fit line.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).parent.parent  # paper-replication/python/
DATA = ROOT.parent / "data"          # shared data lives at the repl root

# Figure 1 uses the base sample (the 64 countries in the main analysis).
df = pd.read_stata(DATA / "maketable1.dta")
df = df[df["baseco"] == 1].dropna(subset=["logpgp95", "logem4"])

# Reduced-form OLS fit: log GDP ~ log settler mortality.
fit = smf.ols("logpgp95 ~ logem4", data=df).fit()
xs = np.linspace(df["logem4"].min(), df["logem4"].max(), 100)
ys = fit.params["Intercept"] + fit.params["logem4"] * xs

fig, ax = plt.subplots(figsize=(8, 6))
# Plot country codes as the markers, like the paper.
for _, r in df.iterrows():
    ax.annotate(r["shortnam"], (r["logem4"], r["logpgp95"]),
                fontsize=8, ha="center", va="center")
ax.plot(xs, ys, color="black", linewidth=1)

# Match the original paper's axis ticks and ranges.
ax.set_xlim(1.8, 8.4)
ax.set_ylim(3.8, 10.5)
ax.set_xticks([2, 4, 6, 8])
ax.set_yticks([4, 6, 8, 10])
ax.set_xlabel("Log of Settler Mortality")
ax.set_ylabel("Log GDP per capita, PPP, 1995")
ax.set_title("Figure 1. Reduced-form relationship between income and settler mortality")

figdir = ROOT / "figures"
figdir.mkdir(exist_ok=True)
out = figdir / "figure1.png"
fig.tight_layout()
fig.savefig(out, dpi=150)

print(f"n = {len(df)}")
print(f"slope = {fit.params['logem4']:.3f}  (se {fit.bse['logem4']:.3f}),  R^2 = {fit.rsquared:.3f}")
print(f"saved {out}")
