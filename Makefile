# Every figure and number in the paper is produced by a script listed here.
# Convention: experiments/fig_*.py write to paper/figs/, experiments/tab_*.py
# write to results/tables/.

PYTHON ?= python

FIGURE_SCRIPTS := $(sort $(wildcard experiments/fig_*.py))
TABLE_SCRIPTS := $(sort $(wildcard experiments/tab_*.py))

.PHONY: data test figures tables paper clean

data:
	@for s in lbnl2019 wang2025 ghalamsiah2026 lbnl2022; do $(PYTHON) scripts/download/$$s.py || exit 1; done

test:
	$(PYTHON) -m pytest

figures:
	@for s in $(FIGURE_SCRIPTS); do echo "$$s"; $(PYTHON) "$$s" || exit 1; done

tables:
	@for s in $(TABLE_SCRIPTS); do echo "$$s"; $(PYTHON) "$$s" || exit 1; done

paper:
	cd paper && latexmk -pdf main.tex

clean:
	cd paper && latexmk -C main.tex
