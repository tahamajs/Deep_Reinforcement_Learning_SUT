Build instructions for HW7 report

This folder contains metadata and build artifacts for Homework 7.

Files:
- run_info.json : environment and run metadata (seed, device, versions, commit)

To build the LaTeX report (requires a local TeX distribution with `pdflatex` and the `markdown` package):

1. Change directory to the template folder:
   cd Homework-7-Template

2. Run (from `homeworks/HW7_Deep_RL`):
   pdflatex -interaction=nonstopmode -halt-on-error -output-directory ../reports template.tex

Notes:
- If `pdflatex` is not available on your system, install TeX Live (Linux/macOS) or MikTeX (Windows).
- The template uses the `markdown` LaTeX package to import `../HW7_Complete_Solutions.md`. Ensure your TeX distribution has this package, or replace the markdown import with converted LaTeX.
- The repository does not include a pre-built PDF because `pdflatex` was not found in the environment where this step was attempted.





