# Replicate AJR (2001) Figure 3:
# First-stage relationship between settler mortality and expropriation risk.
#
# x = log settler mortality (logem4)  [the instrument]
# y = avg. protection against expropriation risk, 1985-95 (avexpr)  [institutions]
# country-code labels + OLS fit line.
#
# R port of python/scripts/figure3.py (reads the same shared data/).

# Resolve this script's location so paths work regardless of cwd.
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

# Figure 3 uses the base sample (the 64 countries in the main analysis).
df <- foreign::read.dta(file.path(DATA, "maketable1.dta"))
df <- df[which(df$baseco == 1 & !is.na(df$avexpr) & !is.na(df$logem4)), ]

# First-stage OLS fit: expropriation risk ~ log settler mortality.
fit <- lm(avexpr ~ logem4, data = df)
xs <- seq(min(df$logem4), max(df$logem4), length.out = 100)
ys <- coef(fit)[["(Intercept)"]] + coef(fit)[["logem4"]] * xs

figdir <- file.path(ROOT, "figures")
dir.create(figdir, showWarnings = FALSE)
out <- file.path(figdir, "figure3.png")
png(out, width = 8, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))

# Match the original paper's axis ticks and ranges.
plot(df$logem4, df$avexpr, type = "n",
     xlim = c(1.8, 8.4), ylim = c(3.5, 10.5),
     xaxt = "n", yaxt = "n", cex.main = 0.95,
     xlab = "Log of Settler Mortality",
     ylab = "Average Expropriation Risk 1985-95",
     main = "Figure 3. First-stage relationship between settler mortality and expropriation risk")
axis(1, at = c(2, 4, 6, 8))
axis(2, at = c(4, 6, 8, 10))
# Plot country codes as the markers, like the paper.
text(df$logem4, df$avexpr, labels = df$shortnam, cex = 0.65)
lines(xs, ys, col = "black", lwd = 1)
invisible(dev.off())

cat(sprintf("n = %d\n", nrow(df)))
cat(sprintf("slope = %.3f  (se %.3f),  R^2 = %.3f\n",
            coef(fit)[["logem4"]],
            summary(fit)$coefficients["logem4", "Std. Error"],
            summary(fit)$r.squared))
cat(sprintf("saved %s\n", out))
