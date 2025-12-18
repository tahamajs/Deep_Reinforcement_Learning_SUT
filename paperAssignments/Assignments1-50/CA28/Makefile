.PHONY: report test lint

report:
	latexmk -pdf -silent report.tex

test:
	pytest -q

lint:
	python -m pip install ruff && ruff check .
