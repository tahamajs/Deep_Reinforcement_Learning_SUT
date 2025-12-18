# Makefile for CA27
.PHONY: test lint

test:
	python -m pytest -q

lint:
	python -m pip install --upgrade flake8 || true
	flake8 src tests || true

format:
	python -m pip install --upgrade black || true
	black src tests
