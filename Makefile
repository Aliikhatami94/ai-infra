SHELL := /bin/bash

.PHONY: help unit unitv test lint type format clean clean-pycache install

help: ## Show available commands
	@echo "Available commands:"
	@echo ""
	@echo "Testing:"
	@echo "  unit              Run unit tests (quiet)"
	@echo "  unitv             Run unit tests (verbose)"
	@echo "  test              Run all tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  format            Format code with ruff"
	@echo "  lint              Lint code with ruff"
	@echo "  type              Type check with mypy"
	@echo "  check             Run lint + type checks"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install dependencies with poetry"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean             Remove caches, build artifacts, logs"
	@echo "  clean-pycache     Remove only __pycache__ directories"
	@echo ""

# --- Setup ---
install:
	@echo "[install] Installing dependencies with poetry"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[install] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry install --no-interaction

# --- Unit tests ---
unit:
	@echo "[unit] Running unit tests (quiet)"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[unit] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run pytest -q tests/unit

unitv:
	@echo "[unit] Running unit tests (verbose)"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[unit] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run pytest -vv tests/unit

# --- Combined test target ---
test:
	@echo "[test] Running all tests"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[test] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run pytest -q tests/

# --- Code Quality ---
format:
	@echo "[format] Formatting with ruff"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[format] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run ruff format .

lint:
	@echo "[lint] Running ruff check"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[lint] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run ruff check .

type:
	@echo "[type] Running mypy"
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "[type] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; \
		exit 2; \
	fi; \
	poetry run mypy src

check: lint type
	@echo "[check] All checks passed"

# --- Cleanup helpers ---
clean:
	@echo "[clean] Removing Python caches, build artifacts, and logs"
	rm -rf **/__pycache__ __pycache__ .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info *.log htmlcov .coverage

clean-pycache:
	@echo "[clean] Removing all __pycache__ directories recursively"
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
