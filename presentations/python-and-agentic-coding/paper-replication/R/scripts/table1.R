# Replicate AJR (2001) Table 1: Descriptive Statistics.
#
# Reports mean (with standard deviation in parentheses) for each variable across
# columns: whole world, base sample, and base sample split by quartiles of
# (potential) settler mortality. Mirrors the layout on p. 1377.
#
# R port of python/scripts/table1.py (reads the same shared data/).

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- dirname(dirname(normalizePath(script_path)))  # paper-replication/R/
DATA <- file.path(dirname(ROOT), "data")               # shared data at repl root

df <- foreign::read.dta(file.path(DATA, "maketable1.dta"))
# euro1900 is stored 0-100; the paper reports it as a 0-1 fraction.
df$euro1900 <- df$euro1900 / 100

# variable, label, na_in_whole_world
ROWS <- list(
  c("logpgp95", "Log GDP per capita (PPP) in 1995", FALSE),
  c("loghjypl", "Log output per worker in 1988", FALSE),
  c("avexpr",   "Average protection against expropriation risk, 1985-1995", FALSE),
  c("cons90",   "Constraint on executive in 1990", FALSE),
  c("cons00a",  "Constraint on executive in 1900", FALSE),
  c("cons1",    "Constraint on executive in first year of independence", FALSE),
  c("democ00a", "Democracy in 1900", FALSE),
  c("euro1900", "European settlements in 1900", FALSE),
  c("logem4",   "Log European settler mortality", TRUE)  # n.a. for whole world
)

# Base sample, split by quartiles of raw settler mortality (extmort4).
# Cutoffs follow the paper's Table 1 notes (65.4 / 78.1 / 280). Five countries
# are stored as the float32 value of 78.1 (78.0999984741211); the paper assigns
# them to quartile (2), so we put the Q2/Q3 boundary at 78.15 -- in the gap
# before the next group at 78.2 -- to reproduce the paper's 14/18/17/15.
base <- df[which(df$baseco == 1), ]
CUTS <- list(c(-Inf, 65.4), c(65.4, 78.15), c(78.15, 280), c(280, Inf))
quartiles <- lapply(CUTS, function(cut) {
  base[which(base$extmort4 >= cut[1] & base$extmort4 < cut[2]), ]
})

columns <- c(list(`Whole world` = df, `Base sample` = base),
             setNames(quartiles, sprintf("(%d)", seq_along(quartiles))))

cell <- function(x, na) {
  if (na) return("n.a.")
  s <- x[!is.na(x)]
  if (length(s) == 0) return("—")
  sprintf("%.2f (%.2f)", mean(s), sd(s))
}

# Build the table column by column.
out <- sapply(names(columns), function(name) {
  data <- columns[[name]]
  na_flag <- name == "Whole world"
  vals <- sapply(ROWS, function(r) cell(data[[r[1]]], as.logical(r[3]) && na_flag))
  c(vals, as.character(nrow(data)))  # append N
})
out <- as.data.frame(out, check.names = FALSE, stringsAsFactors = FALSE)
rownames(out) <- c(sapply(ROWS, `[`, 2), "Number of observations")

options(width = 300)
report <- c(
  "TABLE 1 — DESCRIPTIVE STATISTICS",
  "(mean; standard deviation in parentheses; quartiles by settler mortality)",
  "",
  capture.output(print(out))
)
report <- paste0(paste(report, collapse = "\n"), "\n")
cat(report)

dir.create(file.path(ROOT, "tables"), showWarnings = FALSE)
outpath <- file.path(ROOT, "tables", "table1.txt")
writeLines(report, outpath, sep = "")
cat(sprintf("saved %s\n", outpath))
