.PHONY: setup data smoke-ids run eval reproduce reproduce-full tests baselines-smoke check-resolved-model lint clean

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

# Offline property tests (no API). CI-friendly.
tests:
	$(PY) scripts/property_tests.py

# Resolved-model probe; requires ANTHROPIC_API_KEY.
check-resolved-model:
	$(PY) scripts/check_resolved_model.py

# Live-API smoke: the four baselines (oracle, full-context-s, rag-bm25, rag-dense)
# on the frozen 50-question subset. RAG runs at both turn and session granularity.
# ActiveGraph is intentionally skipped this round (it remains a stub).
# Requires: ANTHROPIC_API_KEY (reader) and OPENAI_API_KEY (judge + dense embeddings).
baselines-smoke:
	$(PY) scripts/run_matrix.py --baselines-only

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache **/__pycache__
