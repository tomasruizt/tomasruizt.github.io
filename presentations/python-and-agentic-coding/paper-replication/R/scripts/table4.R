# Replicate AJR (2001) Table 4: IV Regressions of Log GDP per Capita (p. 1385).
#
# The headline of the paper. Protection against expropriation risk (avexpr) is
# instrumented with log settler mortality (logem4); the 2SLS coefficient on
# avexpr in column (1) is the famous ~0.94.
#
# Nine columns, three panels:
#   (1) base sample                          (6) (5) + latitude
#   (2) base sample + latitude               (7) base + continent dummies
#   (3) base sample without Neo-Europes      (8) (7) + latitude
#   (4) (3) + latitude                       (9) dep. var = log output per worker
#   (5) base sample without Africa
#   Panel A: 2SLS.  Panel B: first stage.  Panel C: OLS.  America omitted.
#
# Data note: the public maketable4.dta lacks the `other` continent dummy that
# cols (7)-(8) need, so we merge it in from maketable2.dta on `shortnam` (same
# coding the Python table4.py / table2.py rely on).
#
# R port of python/scripts/table4.py. Panel A uses ivreg(); note ivreg applies
# the n-k degrees-of-freedom correction to the IV variance, so its SEs match the
# printed paper (col 1: 0.16) rather than linearmodels' "unadjusted" SE (0.15).
# The point estimates are identical across both implementations.

suppressMessages(library(ivreg))

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

df <- foreign::read.dta(file.path(DATA, "maketable4.dta"))
other <- foreign::read.dta(file.path(DATA, "maketable2.dta"))[, c("shortnam", "other")]
df <- merge(df, other, by = "shortnam", all.x = TRUE)

# label, outcome, sample mask (logical over df), extra exogenous controls
SPECS <- list(
  list("(1)", "logpgp95", df$baseco == 1, character(0)),
  list("(2)", "logpgp95", df$baseco == 1, c("lat_abst")),
  list("(3)", "logpgp95", df$baseco == 1 & df$rich4 == 0, character(0)),
  list("(4)", "logpgp95", df$baseco == 1 & df$rich4 == 0, c("lat_abst")),
  list("(5)", "logpgp95", df$baseco == 1 & df$africa == 0, character(0)),
  list("(6)", "logpgp95", df$baseco == 1 & df$africa == 0, c("lat_abst")),
  list("(7)", "logpgp95", df$baseco == 1, c("asia", "africa", "other")),
  list("(8)", "logpgp95", df$baseco == 1, c("lat_abst", "asia", "africa", "other")),
  list("(9)", "loghjypl", df$baseco == 1, character(0))
)
CONTROLS <- c("lat_abst", "asia", "africa", "other")

fit_column <- function(outcome, mask, controls) {
  cols <- c(outcome, "avexpr", "logem4", controls)
  data <- na.omit(df[which(mask), cols])
  rhs <- paste(c(controls, "avexpr"), collapse = " + ")
  inst <- paste(c(controls, "logem4"), collapse = " + ")
  # Panel A: 2SLS, avexpr instrumented by logem4.
  iv <- ivreg(as.formula(sprintf("%s ~ %s | %s", outcome, rhs, inst)), data = data)
  # Panel B: first stage.
  first <- lm(as.formula(paste("avexpr ~", inst)), data = data)
  # Panel C: OLS of the outcome on avexpr (+ controls).
  ols <- lm(as.formula(paste(outcome, "~", rhs)), data = data)
  list(iv = iv, first = first, ols = ols, n = nrow(data))
}

cell <- function(fit, term) {
  co <- summary(fit)$coefficients
  if (!term %in% rownames(co)) return("")
  sprintf("%.2f (%.2f)", co[term, "Estimate"], co[term, "Std. Error"])
}

labels <- sapply(SPECS, `[[`, 1)
fits <- lapply(SPECS, function(s) fit_column(s[[2]], s[[3]], s[[4]]))

panelA <- sapply(fits, function(f) sapply(c("avexpr", CONTROLS), function(t) cell(f$iv, t)))
panelB <- sapply(fits, function(f) c(sapply(c("logem4", CONTROLS), function(t) cell(f$first, t)),
                                     sprintf("%.2f", summary(f$first)$r.squared)))
panelC <- sapply(fits, function(f) cell(f$ols, "avexpr"))
nobs   <- sapply(fits, function(f) as.character(f$n))

mk <- function(mat, rn) {
  d <- as.data.frame(mat, check.names = FALSE, stringsAsFactors = FALSE)
  colnames(d) <- labels; rownames(d) <- rn
  d
}
panelA_df <- mk(panelA, c("Avg. expropriation risk (2SLS)", "Latitude", "Asia dummy",
                          "Africa dummy", '"Other" continent dummy'))
panelB_df <- mk(panelB, c("Log settler mortality", "Latitude", "Asia dummy",
                          "Africa dummy", '"Other" continent dummy', "R-squared"))
panelC_df <- mk(matrix(panelC, nrow = 1), "Avg. expropriation risk (OLS)")
nobs_df   <- mk(matrix(nobs, nrow = 1), "Number of observations")

samples <- c("base", "base", "no Neo-Europes", "no Neo-Europes", "no Africa",
             "no Africa", "base+dummies", "base+dummies", "loghjypl")

options(width = 300)
report <- c(
  "TABLE 4 — IV REGRESSIONS OF LOG GDP PER CAPITA",
  "(coefficient; standard error in parentheses)",
  paste("sample:", paste(sprintf("%s=%s", labels, samples), collapse = "  ")),
  "",
  "Panel A: Two-Stage Least Squares",
  capture.output(print(panelA_df)),
  "",
  "Panel B: First Stage (avexpr on log settler mortality)",
  capture.output(print(panelB_df)),
  "",
  "Panel C: Ordinary Least Squares",
  capture.output(print(panelC_df)),
  capture.output(print(nobs_df)),
  "",
  sprintf("Headline: column (1) 2SLS coef on avexpr = %s  (paper: 0.94 (0.16))",
          panelA_df[["(1)"]][1])
)
report <- paste0(paste(report, collapse = "\n"), "\n")
cat(report)

dir.create(file.path(ROOT, "tables"), showWarnings = FALSE)
outpath <- file.path(ROOT, "tables", "table4.txt")
writeLines(report, outpath, sep = "")
cat(sprintf("saved %s\n", outpath))
