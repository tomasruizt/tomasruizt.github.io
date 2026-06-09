# Runner for the AJR (2001) replication.
#
#   make python   # run all Python scripts -> python/figures, python/tables
#   make R        # run all R scripts      -> R/figures, R/tables
#   make all      # both (default)
#   make report   # compile the LaTeX replication report -> rep-paper/report.pdf
#
# Both languages read the shared data/ at this directory.

PY := .venv/bin/python
SCRIPTS := figure1 figure2 figure3 table1 table2 table4

.PHONY: all python R report

all: python R

python:
	@for s in $(SCRIPTS); do \
		echo "=== python/scripts/$$s.py ==="; \
		$(PY) python/scripts/$$s.py || exit 1; \
	done

R:
	@for s in $(SCRIPTS); do \
		echo "=== R/scripts/$$s.R ==="; \
		Rscript R/scripts/$$s.R || exit 1; \
	done

# rep-paper/img is a symlink to python/figures, so ensure those figures exist
# (they are gitignored and absent on a fresh clone) before compiling.
report: python
	cd rep-paper && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
