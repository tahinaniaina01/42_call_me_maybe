PYTHON := uv run python
PATH := $(CURDIR)/.venv/bin:$(PATH)

run:
	$(PYTHON) -m src

install:
	uv sync
	uv pip install flake8 mypy

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".coverage" \) -delete

lint:
	flake8 .
	mypy . --warn-return-any

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: install run debug clean lint lint-strict
