# Replicate AJR (2001) Figure 2:
# OLS relationship between expropriation risk and income.
#
# x = avg. protection against expropriation risk, 1985-95 (avexpr)  [institutions]
# y = log GDP per capita, 1995, PPP (logpgp95)
# country-code labels + OLS fit line.
#
# R port of python/scripts/figure2.py (reads the same shared data/).

# Resolve this script's location so paths work regardless of cwd.
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

# Figure 2 uses the base sample (the 64 countries in the main analysis).
df <- foreign::read.dta(file.path(DATA, "maketable1.dta"))
df <- df[which(df$baseco == 1 & !is.na(df$logpgp95) & !is.na(df$avexpr)), ]

# OLS fit: log GDP ~ expropriation risk (Table 2, column 2).
fit <- lm(logpgp95 ~ avexpr, data = df)
xs <- seq(min(df$avexpr), max(df$avexpr), length.out = 100)
ys <- coef(fit)[["(Intercept)"]] + coef(fit)[["avexpr"]] * xs

figdir <- file.path(ROOT, "figures")
dir.create(figdir, showWarnings = FALSE)
out <- file.path(figdir, "figure2.png")
png(out, width = 8, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))

# Match the original paper's axis ticks and ranges.
plot(df$avexpr, df$logpgp95, type = "n",
     xlim = c(3.5, 10.5), ylim = c(3.8, 10.5),
     xaxt = "n", yaxt = "n",
     xlab = "Average Expropriation Risk 1985-95",
     ylab = "Log GDP per capita, PPP, 1995",
     main = "Figure 2. OLS relationship between expropriation risk and income")
axis(1, at = c(4, 6, 8, 10))
axis(2, at = c(4, 6, 8, 10))
# Plot country codes as the markers, like the paper.
text(df$avexpr, df$logpgp95, labels = df$shortnam, cex = 0.65)
lines(xs, ys, col = "black", lwd = 1)
invisible(dev.off())

cat(sprintf("n = %d\n", nrow(df)))
cat(sprintf("slope = %.3f  (se %.3f),  R^2 = %.3f\n",
            coef(fit)[["avexpr"]],
            summary(fit)$coefficients["avexpr", "Std. Error"],
            summary(fit)$r.squared))
cat(sprintf("saved %s\n", out))
