.PHONY: install test lint demo clean-caches

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check .

demo:
	python -m datagoblin demo --seed 7 --out runs/demo

clean-caches:
	python scripts/clear_local_caches.py --apply

