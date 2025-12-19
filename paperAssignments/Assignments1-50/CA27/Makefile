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

paper:
	@echo "Building LaTeX paper (requires pdflatex or latexmk)"
	@which latexmk > /dev/null 2>&1 && latexmk -pdf -silent paper.tex || (pdflatex paper.tex && bibtex paper.aux && pdflatex paper.tex && pdflatex paper.tex)
