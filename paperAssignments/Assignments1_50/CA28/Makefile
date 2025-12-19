.PHONY: report test lint format pre-commit

report:
	latexmk -pdf -silent report.tex

test:
	pytest -q

lint:
	python -m pip install ruff && ruff check .

format:
	python -m pip install ruff && ruff format .

pre-commit:
	python -m pip install pre-commit && pre-commit install
