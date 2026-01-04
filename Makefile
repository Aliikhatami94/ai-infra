SHELL := /bin/bash

# Defaults for make pr flags
sync  ?= 0
new   ?= 0
draft ?= 0
b     ?=
base  ?=

.PHONY: help unit unitv test lint type typecheck format format-check clean clean-pycache install check ci report

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
	@echo "  report            Production readiness analysis report"
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

# --- Benchmarks ---
benchmark: _poetry-check
	@echo "[benchmark] Running performance benchmarks"
	@poetry run pytest benchmarks/ --benchmark-only --benchmark-sort=mean

benchmark-save: _poetry-check
	@echo "[benchmark] Running benchmarks and saving results"
	@poetry run pytest benchmarks/ --benchmark-only --benchmark-autosave --benchmark-save-data

benchmark-compare: _poetry-check
	@echo "[benchmark] Comparing with previous benchmark results"
	@poetry run pytest benchmarks/ --benchmark-only --benchmark-compare

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

# --- Production Readiness Report ---
# Minimum coverage threshold (override with: make report COV_MIN=70)
# Strict mode for CI (override with: make report STRICT=1)
# Report mode: full (default) or ci (skip lint/mypy/pytest, assume CI already ran them)
COV_MIN ?= 60
STRICT ?= 0
REPORT_MODE ?= full

.PHONY: report

report: _poetry-check ## Production readiness gate (CI-friendly with exit codes)
	@set -euo pipefail; \
	echo ""; \
	echo "╔══════════════════════════════════════════════════════════════════════════════╗"; \
	echo "║                     PRODUCTION READINESS GATE                              ║"; \
	echo "║                           ai-infra                                           ║"; \
	echo "╚══════════════════════════════════════════════════════════════════════════════╝"; \
	echo ""; \
	if [ "$(REPORT_MODE)" = "ci" ]; then \
		if [ "$${CI:-}" != "true" ]; then \
			echo "[X] ERROR: REPORT_MODE=ci requires CI=true environment variable"; \
			echo "   This mode should only be used in GitHub Actions, not locally."; \
			echo "   Run 'make report' instead for full local checks."; \
			exit 1; \
		fi; \
		: "$${LINT_PASSED:?REPORT_MODE=ci requires LINT_PASSED=1 from upstream job}"; \
		: "$${TYPE_PASSED:?REPORT_MODE=ci requires TYPE_PASSED=1 from upstream job}"; \
		: "$${TESTS_PASSED:?REPORT_MODE=ci requires TESTS_PASSED=1 from upstream job}"; \
	fi; \
	VERSION=$$(poetry version -s 2>/dev/null || echo "unknown"); \
	echo " Package Version: $$VERSION"; \
	echo " Coverage Minimum: $(COV_MIN)%"; \
	if [ "$(STRICT)" = "1" ]; then echo " Strict Mode: ON (fails if score < 9/11)"; fi; \
	if [ "$(REPORT_MODE)" = "ci" ]; then echo " CI Mode: ON (skipping lint/mypy/pytest)"; fi; \
	echo ""; \
	\
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo " RUNNING ALL CHECKS..."; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo ""; \
	\
	SCORE=0; \
	CRITICAL_FAIL=0; \
	\
	echo "1. Linting (ruff)..."; \
	if [ "$(REPORT_MODE)" = "ci" ]; then \
		echo "   ⏭  SKIP (CI mode - already ran in CI)"; \
		LINT_OK=1; SCORE=$$((SCORE + 1)); \
	else \
		set +e; poetry run ruff check src tests >/dev/null 2>&1; LINT_EXIT=$$?; set -e; \
		if [ "$$LINT_EXIT" -eq 0 ]; then \
			echo "   [OK] PASS (1 pt)"; \
			LINT_OK=1; SCORE=$$((SCORE + 1)); \
		else \
			echo "   [X] FAIL - linting errors found:"; \
			poetry run ruff check src tests 2>&1 | head -20; \
			LINT_OK=0; \
		fi; \
	fi; \
	echo ""; \
	\
	echo "2. Type checking (mypy)..."; \
	if [ "$(REPORT_MODE)" = "ci" ]; then \
		echo "   ⏭  SKIP (CI mode - already ran in CI)"; \
		TYPE_OK=1; SCORE=$$((SCORE + 1)); \
	else \
		set +e; poetry run mypy src >/dev/null 2>&1; MYPY_EXIT=$$?; set -e; \
		if [ "$$MYPY_EXIT" -eq 0 ]; then \
			echo "   [OK] PASS (1 pt)"; \
			TYPE_OK=1; SCORE=$$((SCORE + 1)); \
		else \
			echo "   [X] FAIL - type errors found:"; \
			poetry run mypy src 2>&1 | head -20; \
			TYPE_OK=0; \
		fi; \
	fi; \
	echo ""; \
	\
	echo "3. Tests + Coverage (min $(COV_MIN)%)..."; \
	if [ "$(REPORT_MODE)" = "ci" ]; then \
		echo "   ⏭  SKIP (CI mode - already ran in CI)"; \
		TEST_OK=1; COV_OK=1; SCORE=$$((SCORE + 4)); \
	else \
		set +e; COV_OUTPUT=$$(poetry run pytest --cov=src --cov-report=term-missing -q tests/unit 2>&1); TEST_EXIT=$$?; set -e; \
		COV_PCT=$$(echo "$$COV_OUTPUT" | awk '/^TOTAL/ {for(i=1;i<=NF;i++) if($$i ~ /%$$/) {gsub(/%/,"",$$i); print $$i; exit}}'); \
		if [ -z "$$COV_PCT" ]; then \
			if echo "$$COV_OUTPUT" | grep -qE "unrecognized arguments: --cov|pytest_cov|No module named.*pytest_cov"; then \
				echo "   [X] FAIL - pytest-cov not installed (poetry add --group dev pytest-cov)"; \
			else \
				echo "   [X] FAIL - tests failed or no coverage data"; \
				echo "$$COV_OUTPUT" | tail -10; \
			fi; \
			TEST_OK=0; COV_OK=0; CRITICAL_FAIL=1; \
		elif [ "$$TEST_EXIT" -ne 0 ]; then \
			echo "   [X] FAIL - tests failed"; \
			echo "$$COV_OUTPUT" | tail -10; \
			TEST_OK=0; COV_OK=0; CRITICAL_FAIL=1; \
		elif [ "$$COV_PCT" -lt $(COV_MIN) ]; then \
			echo "   [X] FAIL - tests passed but $${COV_PCT}% coverage below $(COV_MIN)%"; \
			TEST_OK=1; COV_OK=0; SCORE=$$((SCORE + 2)); CRITICAL_FAIL=1; \
		else \
			echo "   [OK] PASS - $${COV_PCT}% coverage (4 pts: 2 tests + 2 coverage)"; \
			TEST_OK=1; COV_OK=1; SCORE=$$((SCORE + 4)); \
		fi; \
	fi; \
	echo ""; \
	\
	echo "4. Security: Vulnerability scan (pip-audit)..."; \
	if poetry run pip-audit --version >/dev/null 2>&1; then \
		set +e; poetry run pip-audit >/dev/null 2>&1; AUDIT_EXIT=$$?; set -e; \
		if [ "$$AUDIT_EXIT" -eq 0 ]; then \
			echo "   [OK] PASS - no known vulnerabilities (2 pts)"; \
			VULN_OK=1; SCORE=$$((SCORE + 2)); \
		else \
			echo "   [X] FAIL - vulnerabilities found"; \
			poetry run pip-audit 2>&1 | head -15; \
			VULN_OK=0; CRITICAL_FAIL=1; \
		fi; \
	else \
		if [ "$(STRICT)" = "1" ]; then \
			echo "   [X] FAIL - pip-audit required in STRICT mode (poetry add --group dev pip-audit)"; \
			VULN_OK=0; CRITICAL_FAIL=1; \
		else \
			echo "   [!]  SKIP - pip-audit not installed (0 pts)"; \
			VULN_OK=0; \
		fi; \
	fi; \
	echo ""; \
	\
	echo "5. Package build + verification..."; \
	rm -rf dist/; \
	if poetry build -q 2>/dev/null; then \
		if poetry run twine --version >/dev/null 2>&1 && poetry run twine check dist/* >/dev/null 2>&1; then \
			echo "   [OK] PASS - package builds and passes twine check (2 pts)"; \
			BUILD_OK=1; SCORE=$$((SCORE + 2)); \
		elif poetry run python -m zipfile -t dist/*.whl >/dev/null 2>&1; then \
			echo "   [OK] PASS - package builds, wheel is valid (2 pts)"; \
			BUILD_OK=1; SCORE=$$((SCORE + 2)); \
		else \
			echo "   [OK] PASS - package builds (2 pts)"; \
			BUILD_OK=1; SCORE=$$((SCORE + 2)); \
		fi; \
	else \
		echo "   [X] FAIL - package build failed"; \
		BUILD_OK=0; CRITICAL_FAIL=1; \
	fi; \
	echo ""; \
	\
	echo "6. Documentation..."; \
	DOC_SCORE=0; \
	[ -f README.md ] && DOC_SCORE=$$((DOC_SCORE + 1)); \
	[ -f CHANGELOG.md ] && DOC_SCORE=$$((DOC_SCORE + 1)); \
	[ -d docs ] && DOC_SCORE=$$((DOC_SCORE + 1)); \
	if [ "$$DOC_SCORE" -ge 2 ]; then \
		echo "   [OK] PASS - core docs present (1 pt)"; \
		DOCS_OK=1; SCORE=$$((SCORE + 1)); \
	else \
		echo "   [X] FAIL - missing README.md, CHANGELOG.md, or docs/"; \
		DOCS_OK=0; \
	fi; \
	echo ""; \
	\
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo " RESULTS"; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	echo ""; \
	echo "  Component          Weight    Status"; \
	echo "  ─────────────────────────────────────"; \
	[ "$$LINT_OK" = "1" ] && echo "  Linting            1 pt      [OK]" || echo "  Linting            1 pt      [X]"; \
	[ "$$TYPE_OK" = "1" ] && echo "  Type checking      1 pt      [OK]" || echo "  Type checking      1 pt      [X]"; \
	[ "$$TEST_OK" = "1" ] && echo "  Tests pass         2 pts     [OK]" || echo "  Tests pass         2 pts     [X] CRITICAL"; \
	[ "$$COV_OK" = "1" ] && echo "  Coverage ≥$(COV_MIN)%     2 pts     [OK]" || echo "  Coverage ≥$(COV_MIN)%     2 pts     [X] CRITICAL"; \
	if [ "$$VULN_OK" = "1" ]; then echo "  No vulnerabilities 2 pts     [OK]"; \
	elif [ "$(STRICT)" = "1" ]; then echo "  No vulnerabilities 2 pts     [X] CRITICAL"; \
	else echo "  No vulnerabilities 2 pts     [!]  SKIP"; fi; \
	[ "$$BUILD_OK" = "1" ] && echo "  Package builds     2 pts     [OK]" || echo "  Package builds     2 pts     [X] CRITICAL"; \
	[ "$$DOCS_OK" = "1" ] && echo "  Documentation      1 pt      [OK]" || echo "  Documentation      1 pt      [X]"; \
	echo "  ─────────────────────────────────────"; \
	echo "  TOTAL              11 pts    $$SCORE"; \
	echo ""; \
	\
	PERCENT=$$((SCORE * 100 / 11)); \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	if [ "$$CRITICAL_FAIL" = "1" ]; then \
		echo ""; \
		echo "  [X] NOT READY FOR PRODUCTION ($$PERCENT% - $$SCORE/11 pts)"; \
		echo ""; \
		echo "  Critical failures detected. Fix before release:"; \
		[ "$$TEST_OK" = "0" ] && echo "    • Tests must pass"; \
		[ "$$COV_OK" = "0" ] && echo "    • Coverage must be ≥$(COV_MIN)%"; \
		[ "$$VULN_OK" = "0" ] && [ "$(STRICT)" = "1" ] && echo "    • Vulnerabilities must be resolved"; \
		[ "$$BUILD_OK" = "0" ] && echo "    • Package must build successfully"; \
		echo ""; \
		echo "╚══════════════════════════════════════════════════════════════════════════════╝"; \
		exit 1; \
	elif [ "$(STRICT)" = "1" ] && [ "$$SCORE" -lt 9 ]; then \
		echo ""; \
		echo "  [X] STRICT MODE: Score $$SCORE/11 is below 9/11 threshold"; \
		echo ""; \
		echo "╚══════════════════════════════════════════════════════════════════════════════╝"; \
		exit 1; \
	elif [ "$$SCORE" -ge 9 ]; then \
		echo ""; \
		echo "  [OK] READY FOR PRODUCTION ($$PERCENT% - $$SCORE/11 pts)"; \
		echo ""; \
		echo "╚══════════════════════════════════════════════════════════════════════════════╝"; \
	else \
		echo ""; \
		echo "  [!]  NEEDS WORK ($$PERCENT% - $$SCORE/11 pts)"; \
		echo ""; \
		echo "  No critical failures, but score below 9/11."; \
		echo "  Use STRICT=1 to enforce in CI."; \
		echo ""; \
		echo "╚══════════════════════════════════════════════════════════════════════════════╝"; \
	fi

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
#   make pr m="feat: add new feature"        # Create PR from main or update existing PR
#   make pr m="fix: bug" sync=1              # Rebase feature branch on default branch before pushing
#   make pr m="wip" FORCE=1                  # Override conventional commit check
#   make pr m="feat: new" new=1              # Create new PR from feature branch (new branch from HEAD)
#   make pr m="feat: add" b="feat/my-branch" # Use explicit branch name
#   make pr m="feat: wip" draft=1            # Create PR as draft
#   make pr m="fix: hotfix" base=release     # Target a different base branch
pr:
ifndef m
	$(error Usage: make pr m="feat: your commit message")
endif
	@./scripts/pr.sh "$(m)" "$(sync)" "$(new)" "$(b)" "$${FORCE:-0}" "$(draft)" "$(base)"

# Usage: make commit m="feat: add new feature"
# Just commits with proper message (for when you want to batch commits before PR)
commit:
ifndef m
	$(error Usage: make commit m="feat: your commit message")
endif
	@git add -A && git commit -m "$(m)"
