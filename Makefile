.PHONY: setup data smoke-ids run eval reproduce reproduce-full lint clean

PY ?= uv run python

setup:
	git submodule update --init --recursive
	uv sync --frozen

data:
	bash scripts/download_data.sh

smoke-ids:
	$(PY) scripts/build_smoke_ids.py

# `make run SYSTEM=rag-bm25 DATA=s` (DATA in {oracle,s})
run:
	@if [ -z "$(SYSTEM)" ] || [ -z "$(DATA)" ]; then \
		echo "Usage: make run SYSTEM=<name> DATA=<oracle|s>"; exit 2; \
	fi
	$(PY) -m activegraph_lme.cli run --system $(SYSTEM) --dataset $(DATA)

# `make eval RUN=runs/<dir>`
eval:
	@if [ -z "$(RUN)" ]; then echo "Usage: make eval RUN=runs/<dir>"; exit 2; fi
	$(PY) -m activegraph_lme.cli eval --run-dir $(RUN)

reproduce:
	$(PY) scripts/run_matrix.py

reproduce-full:
	$(PY) scripts/run_matrix.py --full

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache **/__pycache__
