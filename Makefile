SHELL := /bin/bash

# Default for make pr sync flag
sync ?= 0

.PHONY: help unit unitv test lint type typecheck format format-check clean clean-pycache install check ci

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
	@echo "  format-check      Check formatting (ruff format --check)"
	@echo "  lint              Lint code with ruff"
	@echo "  type              Type check with mypy"
	@echo "  typecheck         Alias for 'type'"
	@echo "  check             Run lint + type checks"
	@echo "  ci                Run checks + tests"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install dependencies with poetry"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean             Remove caches, build artifacts, logs"
	@echo "  clean-pycache     Remove only __pycache__ directories"
	@echo ""

# --- Poetry check helper ---
.PHONY: _poetry-check
_poetry-check:
	@command -v poetry >/dev/null 2>&1 || { echo "[error] Poetry is not installed. Please install Poetry (https://python-poetry.org/docs/#installation)"; exit 2; }

# --- Setup ---
install: _poetry-check
	@echo "[install] Installing dependencies with poetry"
	@poetry install --no-interaction

# --- Unit tests ---
unit: _poetry-check
	@echo "[unit] Running unit tests (quiet)"
	@poetry run pytest -q tests/unit

unitv: _poetry-check
	@echo "[unit] Running unit tests (verbose)"
	@poetry run pytest -vv tests/unit

# --- Combined test target ---
test: _poetry-check
	@echo "[test] Running all tests"
	@poetry run pytest -q tests/

# --- Code Quality ---
format: _poetry-check
	@echo "[format] Formatting with ruff"
	@poetry run ruff format .

format-check: _poetry-check
	@echo "[format] Checking formatting (ruff format --check)"
	@poetry run ruff format --check .

lint: _poetry-check
	@echo "[lint] Running ruff check"
	@poetry run ruff check .

type: _poetry-check
	@echo "[type] Running mypy"
	@poetry run mypy src

typecheck: type

check: lint type
	@echo "[check] All checks passed"

ci: check test
	@echo "[ci] All checks + tests passed"

# --- Cleanup helpers ---
clean:
	@echo "[clean] Removing Python caches, build artifacts, and logs"
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info *.log htmlcov .coverage

clean-pycache:
	@echo "[clean] Removing all __pycache__ directories recursively"
	@find . -type d -name '__pycache__' -prune -exec rm -rf {} +

# --- Docs Changelog ---
.PHONY: docs-changelog docs docs-serve docs-build

docs-changelog: ## Generate/update docs/CHANGELOG.json for What's New page
	@./scripts/docs-changelog.sh

docs: docs-serve ## Alias for docs-serve

docs-serve: ## Serve documentation locally with live reload
	@echo "[docs] Starting documentation server..."
	poetry run mkdocs serve

docs-build: ## Build documentation for production
	@echo "[docs] Building documentation..."
	poetry run mkdocs build

# --- Git/PR Automation ---
.PHONY: pr commit

# Usage:
#   make pr m="feat: add new feature"
#   make pr m="fix: bug" sync=1   # optional: rebase feature branch on origin/main before pushing
pr:
ifndef m
	$(error Usage: make pr m="feat: your commit message")
endif
	@set -euo pipefail; \
	if [ -z "$(m)" ]; then echo "[pr] ERROR: Commit message cannot be empty."; exit 1; fi; \
	gh auth status >/dev/null 2>&1 || { echo "[pr] ERROR: gh CLI not authenticated. Run 'gh auth login' first."; exit 1; }; \
	git remote get-url origin >/dev/null 2>&1 || { echo "[pr] ERROR: remote 'origin' not found."; exit 1; }; \
	CURRENT_BRANCH=$$(git branch --show-current || true); \
	if [ -z "$$CURRENT_BRANCH" ]; then \
		echo "[pr] ERROR: Detached HEAD state. Checkout a branch first."; \
		exit 1; \
	fi; \
	SYNC_FLAG="$(sync)"; \
	if [ "$$SYNC_FLAG" != "1" ]; then SYNC_FLAG="0"; fi; \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "[pr] On main - creating new PR for: $(m)"; \
		TIMESTAMP=$$(date -u +%m%d%H%M); \
		MSG_NO_PREFIX=$$(echo "$(m)" | sed -E 's/^[a-zA-Z]+(\([^)]+\))?!?:[ ]*//'); \
		SLUG=$$(echo "$$MSG_NO_PREFIX" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$$//' | cut -c1-40); \
		[ -z "$$SLUG" ] && SLUG="change"; \
		BRANCH="$$SLUG-$$TIMESTAMP"; \
		git fetch origin main >/dev/null; \
		git pull --ff-only origin main || { echo "[pr] ERROR: main is not fast-forwardable. Resolve manually."; exit 1; }; \
		git checkout -b "$$BRANCH"; \
		git add -A; \
		if git diff --cached --quiet; then \
			echo "[pr] No changes to commit. Cleaning up branch."; \
			git checkout main >/dev/null; \
			git branch -D "$$BRANCH" >/dev/null; \
			exit 0; \
		fi; \
		git commit -m "$(m)"; \
		git push --set-upstream origin "$$BRANCH"; \
		if gh pr view "$$BRANCH" >/dev/null 2>&1; then \
			echo "[pr] PR already exists: $$(gh pr view "$$BRANCH" --json url -q .url)"; \
		else \
			gh pr create --title "$(m)" --body "$(m)" --base main --head "$$BRANCH"; \
		fi; \
		git checkout main >/dev/null; \
		echo "[pr] Done!"; \
	else \
		echo "[pr] On branch $$CURRENT_BRANCH - updating/creating PR"; \
		if [ "$$SYNC_FLAG" = "1" ]; then \
			echo "[pr] Sync enabled - rebasing $$CURRENT_BRANCH on origin/main"; \
			git fetch origin main >/dev/null; \
			git rebase origin/main || { echo "[pr] ERROR: Rebase failed. Run 'git rebase --abort' and resolve manually."; exit 1; }; \
		fi; \
		git add -A; \
		COMMITTED=0; \
		if git diff --cached --quiet; then \
			echo "[pr] No changes to commit"; \
		else \
			git commit -m "$(m)"; \
			COMMITTED=1; \
		fi; \
		if [ "$$SYNC_FLAG" = "1" ]; then \
			git push --force-with-lease origin "$$CURRENT_BRANCH"; \
		elif [ "$$COMMITTED" = "1" ] || ! git rev-parse --verify origin/"$$CURRENT_BRANCH" >/dev/null 2>&1; then \
			git push -u origin "$$CURRENT_BRANCH"; \
		fi; \
		if gh pr view "$$CURRENT_BRANCH" >/dev/null 2>&1; then \
			echo "[pr] PR exists: $$(gh pr view "$$CURRENT_BRANCH" --json url -q .url)"; \
		else \
			gh pr create --title "$(m)" --body "$(m)" --base main --head "$$CURRENT_BRANCH"; \
			echo "[pr] PR created."; \
		fi; \
	fi

# Usage: make commit m="feat: add new feature"
# Just commits with proper message (for when you want to batch commits before PR)
commit:
ifndef m
	$(error Usage: make commit m="feat: your commit message")
endif
	git add -A
	git commit -m "$(m)"
