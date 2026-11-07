PYTHON ?= python3
PYTEST ?= pytest
SPLIT ?= public_test
DATA_ROOT ?= .
CONFIG ?= configs/default_config.yaml
OUTPUT ?= submission.jsonl

.PHONY: help install test inspect prepare infer evaluate ablate clean

help:
	@echo "LifeMate POMDP Voice Agent - Zalo AI Challenge 2026"
	@echo "Targets:"
	@echo "  make install    - Install package and dependencies"
	@echo "  make test       - Run all unit and integration tests"
	@echo "  make inspect    - Inspect dataset splits and tool catalogs"
	@echo "  make prepare    - Build BM25 RAG index and POMDP SFT dataset from train split"
	@echo "  make infer      - Run deterministic offline inference on SPLIT (default: public_test)"
	@echo "  make evaluate   - Run official scorer and ablation studies on train/val"
	@echo "  make clean      - Remove temporary build and cache artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=. $(PYTEST) tests/

inspect:
	PYTHONPATH=. $(PYTHON) scripts/00_inspect_data.py --data-root $(DATA_ROOT)

prepare:
	PYTHONPATH=. $(PYTHON) scripts/prepare_rag_and_sft.py --data-root $(DATA_ROOT) --config $(CONFIG)

infer:
	PYTHONPATH=. $(PYTHON) scripts/run_inference.py \
		--data-root $(DATA_ROOT) \
		--split $(SPLIT) \
		--config $(CONFIG) \
		--output $(OUTPUT)

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_and_ablate.py \
		--data-root $(DATA_ROOT) \
		--config $(CONFIG)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +