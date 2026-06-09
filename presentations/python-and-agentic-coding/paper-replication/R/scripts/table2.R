# Replicate AJR (2001) Table 2: OLS Regressions.
#
# Eight specifications regressing income on protection against expropriation
# risk, progressively adding latitude and continent dummies, across the
# whole-world and base samples (cols 7-8 use log output per worker). Mirrors the
# layout on p. 1379. America is the omitted continent dummy.
#
# As in the Python version, the whole-world N is 111 (not the printed 110): AJR's
# maketable2.do notes one observation was mistakenly dropped from the printed
# table, so 111 is the corrected sample (matches the QuantEcon reproduction).
#
# R port of python/scripts/table2.py (reads the same shared data/).

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

df <- foreign::read.dta(file.path(DATA, "maketable2.dta"))

# label, outcome, regressors, base_sample_only
SPECS <- list(
  list("(1)", "logpgp95", c("avexpr"), FALSE),
  list("(2)", "logpgp95", c("avexpr"), TRUE),
  list("(3)", "logpgp95", c("avexpr", "lat_abst"), FALSE),
  list("(4)", "logpgp95", c("avexpr", "lat_abst", "asia", "africa", "other"), FALSE),
  list("(5)", "logpgp95", c("avexpr", "lat_abst"), TRUE),
  list("(6)", "logpgp95", c("avexpr", "lat_abst", "asia", "africa", "other"), TRUE),
  list("(7)", "loghjypl", c("avexpr"), FALSE),
  list("(8)", "loghjypl", c("avexpr"), TRUE)
)

# Rows of the table, in paper order.
TERMS <- list(
  c("avexpr",   "Average protection against expropriation risk, 1985-1995"),
  c("lat_abst", "Latitude"),
  c("asia",     "Asia dummy"),
  c("africa",   "Africa dummy"),
  c("other",    '"Other" continent dummy')
)

cell <- function(fit, term) {
  co <- summary(fit)$coefficients
  if (!term %in% rownames(co)) return("")
  sprintf("%.2f (%.2f)", co[term, "Estimate"], co[term, "Std. Error"])
}

results <- lapply(SPECS, function(s) {
  data <- if (s[[4]]) df[which(df$baseco == 1), ] else df
  formula <- as.formula(paste(s[[2]], "~", paste(s[[3]], collapse = " + ")))
  lm(formula, data = data)
})
labels <- sapply(SPECS, `[[`, 1)

out <- sapply(results, function(fit) {
  cells <- sapply(TERMS, function(t) cell(fit, t[1]))
  c(cells,
    sprintf("%.2f", summary(fit)$r.squared),
    as.character(length(fit$residuals)))
})
out <- as.data.frame(out, check.names = FALSE, stringsAsFactors = FALSE)
colnames(out) <- labels
rownames(out) <- c(sapply(TERMS, `[`, 2), "R-squared", "Number of observations")

# Header note: which outcome / sample each column uses.
outcomes <- sapply(SPECS, function(s) if (s[[2]] == "logpgp95") "logGDPpc95" else "logOutput/worker88")
samples  <- sapply(SPECS, function(s) if (s[[4]]) "base" else "world")

options(width = 300)
report <- c(
  "TABLE 2 — OLS REGRESSIONS",
  "(coefficient; standard error in parentheses)",
  paste("outcome:", paste(sprintf("%s=%s", labels, outcomes), collapse = "  ")),
  paste("sample: ", paste(sprintf("%s=%s", labels, samples), collapse = "  ")),
  "",
  capture.output(print(out))
)
report <- paste0(paste(report, collapse = "\n"), "\n")
cat(report)

dir.create(file.path(ROOT, "tables"), showWarnings = FALSE)
outpath <- file.path(ROOT, "tables", "table2.txt")
writeLines(report, outpath, sep = "")
cat(sprintf("saved %s\n", outpath))
