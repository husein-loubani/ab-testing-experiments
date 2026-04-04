.PHONY: help install data lint notebook clean

# -- Settings -----------------------------------------------------------------
PYTHON   = venv/bin/python3
PIP      = venv/bin/pip
MODULE   = ab_testing

## help     : Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/^## //'

## install  : Create venv and install all dependencies
install:
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Environment ready. Activate with: source venv/bin/activate"

## data     : Verify raw CSVs are in place
data:
	@test -f data/raw/WA_Marketing-Campaign.csv && \
		echo "data/raw/WA_Marketing-Campaign.csv found." || \
		echo "WARNING: data/raw/WA_Marketing-Campaign.csv missing."
	@test -f data/raw/cookie_cats.csv && \
		echo "data/raw/cookie_cats.csv found." || \
		echo "WARNING: data/raw/cookie_cats.csv missing."

## lint     : Run ruff linter on the source module
lint:
	$(PYTHON) -m ruff check $(MODULE)/

## notebook : Launch Jupyter
notebook:
	$(PYTHON) -m jupyter lab notebooks/

## clean    : Remove compiled Python files and Jupyter checkpoints
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."
