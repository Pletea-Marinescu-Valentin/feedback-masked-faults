# Every figure and number in the paper is produced by a script listed here.
# Convention: experiments/fig_*.py write to paper/figs/, experiments/tab_*.py
# write to results/tables/.

PYTHON ?= python

FIGURE_SCRIPTS := $(sort $(wildcard experiments/fig_*.py))
TABLE_SCRIPTS := $(sort $(wildcard experiments/tab_*.py))

.PHONY: data test figures tables verify reproduce

data:
	$(PYTHON) scripts/download/ghalamsiah2026.py

test:
	$(PYTHON) -m pytest

figures:
	@for s in $(FIGURE_SCRIPTS); do echo "$$s"; $(PYTHON) "$$s" || exit 1; done

tables:
	@for s in $(TABLE_SCRIPTS); do echo "$$s"; $(PYTHON) "$$s" || exit 1; done

verify:
	$(PYTHON) scripts/verify_numbers.py

reproduce: test figures tables verify
