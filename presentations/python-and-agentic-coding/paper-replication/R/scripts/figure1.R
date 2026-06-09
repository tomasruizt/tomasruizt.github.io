# Replicate AJR (2001) Figure 1:
# Reduced-form relationship between income and settler mortality.
#
# x = log settler mortality (logem4)
# y = log GDP per capita, 1995, PPP (logpgp95)
# country-code labels + OLS fit line.
#
# R port of python/scripts/figure1.py (reads the same shared data/).

# Resolve this script's location so paths work regardless of cwd.
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

# Figure 1 uses the base sample (the 64 countries in the main analysis).
df <- foreign::read.dta(file.path(DATA, "maketable1.dta"))
df <- df[which(df$baseco == 1 & !is.na(df$logpgp95) & !is.na(df$logem4)), ]

# Reduced-form OLS fit: log GDP ~ log settler mortality.
fit <- lm(logpgp95 ~ logem4, data = df)
xs <- seq(min(df$logem4), max(df$logem4), length.out = 100)
ys <- coef(fit)[["(Intercept)"]] + coef(fit)[["logem4"]] * xs

figdir <- file.path(ROOT, "figures")
dir.create(figdir, showWarnings = FALSE)
out <- file.path(figdir, "figure1.png")
png(out, width = 8, height = 6, units = "in", res = 150)
par(mar = c(4, 4, 3, 1))

# Match the original paper's axis ticks and ranges.
plot(df$logem4, df$logpgp95, type = "n",
     xlim = c(1.8, 8.4), ylim = c(3.8, 10.5),
     xaxt = "n", yaxt = "n",
     xlab = "Log of Settler Mortality",
     ylab = "Log GDP per capita, PPP, 1995",
     main = "Figure 1. Reduced-form relationship between income and settler mortality")
axis(1, at = c(2, 4, 6, 8))
axis(2, at = c(4, 6, 8, 10))
# Plot country codes as the markers, like the paper.
text(df$logem4, df$logpgp95, labels = df$shortnam, cex = 0.65)
lines(xs, ys, col = "black", lwd = 1)
invisible(dev.off())

cat(sprintf("n = %d\n", nrow(df)))
cat(sprintf("slope = %.3f  (se %.3f),  R^2 = %.3f\n",
            coef(fit)[["logem4"]],
            summary(fit)$coefficients["logem4", "Std. Error"],
            summary(fit)$r.squared))
cat(sprintf("saved %s\n", out))
