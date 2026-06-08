"""
Replicate AJR (2001) Figure 2:
OLS relationship between expropriation risk and income.

x = avg. protection against expropriation risk, 1985-95 (avexpr)  [institutions]
y = log GDP per capita, 1995, PPP (logpgp95)
country-code labels + OLS fit line.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).parent.parent  # paper-replication/

# Figure 2 uses the base sample (the 64 countries in the main analysis).
df = pd.read_stata(ROOT / "data" / "maketable1.dta")
df = df[df["baseco"] == 1].dropna(subset=["logpgp95", "avexpr"])

# OLS fit: log GDP ~ expropriation risk (Table 2, column 2).
fit = smf.ols("logpgp95 ~ avexpr", data=df).fit()
xs = np.linspace(df["avexpr"].min(), df["avexpr"].max(), 100)
ys = fit.params["Intercept"] + fit.params["avexpr"] * xs

fig, ax = plt.subplots(figsize=(8, 6))
# Plot country codes as the markers, like the paper.
for _, r in df.iterrows():
    ax.annotate(r["shortnam"], (r["avexpr"], r["logpgp95"]),
                fontsize=8, ha="center", va="center")
ax.plot(xs, ys, color="black", linewidth=1)

# Match the original paper's axis ticks and ranges.
ax.set_xlim(3.5, 10.5)
ax.set_ylim(3.8, 10.5)
ax.set_xticks([4, 6, 8, 10])
ax.set_yticks([4, 6, 8, 10])
ax.set_xlabel("Average Expropriation Risk 1985-95")
ax.set_ylabel("Log GDP per capita, PPP, 1995")
ax.set_title("Figure 2. OLS relationship between expropriation risk and income")

figdir = ROOT / "figures"
figdir.mkdir(exist_ok=True)
out = figdir / "figure2.png"
fig.tight_layout()
fig.savefig(out, dpi=150)

print(f"n = {len(df)}")
print(f"slope = {fit.params['avexpr']:.3f}  (se {fit.bse['avexpr']:.3f}),  R^2 = {fit.rsquared:.3f}")
print(f"saved {out}")
