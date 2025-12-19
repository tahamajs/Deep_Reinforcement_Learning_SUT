# Simple Makefile to build the LaTeX report
REPORT=report.tex
PDF=report.pdf

all: report.pdf

report.pdf: $(REPORT)
	pdflatex -interaction=nonstopmode $(REPORT) >/dev/null
	pdflatex -interaction=nonstopmode $(REPORT) >/dev/null
	@echo "Built $(PDF)"

clean:
	rm -f *.aux *.log *.out $(PDF)

.PHONY: all clean
