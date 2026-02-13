LATEXMK=latexmk
MAIN_EN=university_project
MAIN_FA=report_fa

all: pdf

pdf: $(MAIN_EN).pdf $(MAIN_FA).pdf

$(MAIN_EN).pdf: $(MAIN_EN).tex references.bib report_table_dqn.tex report_table_policy.tex report_table_ac.tex report_table_model.tex report_table_marl.tex
	$(LATEXMK) -pdf $(MAIN_EN).tex

$(MAIN_FA).pdf: $(MAIN_FA).tex report_table_dqn.tex report_table_policy.tex report_table_ac.tex report_table_model.tex report_table_marl.tex
	$(LATEXMK) -xelatex $(MAIN_FA).tex

analytics:
	MPLBACKEND=Agg PYTHONPATH=. python3.11 scripts/generate_plots.py --suite-dir outputs/suite_reports --runs-root outputs/runs
	python3.11 scripts/build_report_tables.py --aggregate-csv outputs/suite_reports/tables/aggregate_metrics.csv --out-root .

clean:
	$(LATEXMK) -c $(MAIN_EN).tex
	$(LATEXMK) -c $(MAIN_FA).tex

cleanall:
	$(LATEXMK) -C $(MAIN_EN).tex
	$(LATEXMK) -C $(MAIN_FA).tex

.PHONY: all pdf analytics clean cleanall
