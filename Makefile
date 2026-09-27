# nefi developer shortcuts.  `make help` lists the targets.
PY ?= python
SRC := nefi tests examples scripts

.PHONY: help install test test-all lint format docs smoke clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## editable install with the dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## fast CPU test-suite (excludes @pytest.mark.slow)
	$(PY) -m pytest -q -m "not slow"

test-all:  ## every test, including the slow ones
	$(PY) -m pytest -q

lint:  ## ruff format --check + ruff check
	ruff format --check $(SRC)
	ruff check $(SRC)

format:  ## ruff format + autofix
	ruff format $(SRC)
	ruff check --fix $(SRC)

docs:  ## regenerate docs/api.md (and build the mkdocs site if mkdocs is installed)
	$(PY) scripts/gen_api_docs.py
	@if command -v mkdocs >/dev/null 2>&1; then mkdocs build; \
	else echo "mkdocs not installed: docs/api.md regenerated (the docs read fine as markdown)"; fi

smoke:  ## run every registered instance with --smoke (outputs in runs/smoke/)
	@set -e; for i in $$($(PY) -c "import nefi; print(' '.join(nefi.list_registered('instance')['instance']))"); do \
		echo "== nefi run $$i --smoke"; nefi run $$i --smoke -q --out runs/smoke/$$i; \
	done

clean:  ## remove caches and smoke outputs
	rm -rf runs/smoke .pytest_cache .ruff_cache .mypy_cache site
