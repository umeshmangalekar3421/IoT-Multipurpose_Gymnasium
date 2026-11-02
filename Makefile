PY ?= python3

.PHONY: help install test lint demo dashboard benchmark clean

help:
	@echo "make install    install runtime + dev dependencies"
	@echo "make test       run the test suite"
	@echo "make demo       render a synthetic match, analyse it, write evidence"
	@echo "make benchmark  measure accuracy layer by layer against ground truth"
	@echo "make dashboard  serve the review dashboard for the last run"
	@echo "make clean      remove generated runs/data/models"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

demo:
	$(PY) -m smartgym.cli demo --sport badminton --out runs/demo

benchmark:
	$(PY) -m smartgym.cli benchmark --out runs/benchmark

dashboard:
	$(PY) -m smartgym.service.dashboard runs/demo

clean:
	rm -rf runs data models .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
