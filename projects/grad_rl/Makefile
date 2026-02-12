LATEXMK=latexmk
MAIN=university_project

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib report_table_dqn.tex report_table_policy.tex report_table_ac.tex report_table_model.tex report_table_marl.tex
	$(LATEXMK) -pdf $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

cleanall:
	$(LATEXMK) -C $(MAIN).tex

.PHONY: all pdf clean cleanall
