LATEX=pdflatex
BIBTEX=bibtex

.PHONY: all clean report

all: report

report: report.pdf

report.pdf: report.tex
	$(LATEX) report.tex
	$(BIBTEX) report
	$(LATEX) report.tex
	$(LATEX) report.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.pdf *.out