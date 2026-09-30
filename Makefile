CONFIG ?= configs/default.yaml
VENV ?= .venv
# pypi only ships cpu-only torch wheels for windows, so cuda builds come from the pytorch index
TORCH_INDEX ?= https://download.pytorch.org/whl/cu126

ifeq ($(OS),Windows_NT)
PYTHON ?= py -3.11
PY := $(VENV)/Scripts/python.exe
else
PYTHON ?= python3.11
PY := $(VENV)/bin/python
endif

.PHONY: setup test lint format data leakage-demo train eval report

$(PY):
	$(PYTHON) -m venv $(VENV)

setup: $(PY)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]" --extra-index-url $(TORCH_INDEX)

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

data:
	$(PY) scripts/prepare_data.py --config $(CONFIG)
	$(PY) scripts/inspect_data.py --config $(CONFIG)
	$(PY) scripts/text_examples.py --config $(CONFIG)

leakage-demo:
	$(PY) scripts/leakage_demo.py --config $(CONFIG)

train:
	$(PY) scripts/train_baselines.py --config $(CONFIG)
	$(PY) scripts/train_fusion.py --config $(CONFIG)

eval report:
	$(error make $@ is not implemented yet)
