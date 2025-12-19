# Report Template — CA22

Use this template as a short report (2–4 pages) summarizing your experiment. Aim for clarity and reproducibility.

1. Title and Authors

- Title: Descriptive title
- Authors: Your name(s)

2. Abstract (1 paragraph)

- Brief motivation, methods, key results, and a takeaway.

3. Introduction (short)

- Context and related work (1–3 sentences)
- Problem statement and why it matters

4. Methods (concise)

- Describe the model and learning objective (refer to `src/model.py` and `src/losses.py`).
- Describe dataset (for CA22, describe the synthetic generator and its properties).
- Training details: optimizer, lr, batch size, number of epochs, and how seeds were set.

5. Experiments and Results

- Describe baselines/variants, key metrics, and evaluation protocol.
- Present results as figures and tables; include sample code lines used to generate tables/figures.
- Add captions that interpret numbers succinctly.

6. Discussion

- Interpret the results: why do you think you obtained them? Any failure modes?
- Limitations and potential next steps.

7. Reproducibility Appendix

- Config file(s) and seed used
- Exact commands to reproduce main figures

8. Files to submit

- `report.pdf` or `report.md`
- `configs/*.yaml` used for main experiments
- `outputs/figures/` containing labeled figures
- Any relevant scripts/notebooks to reproduce the results

---

Tips:
- Keep figures simple and readable (labels, legends, units)
- Add a small table of hyperparameters
- Provide a short README for any non-obvious scripts


Submission checklist:
- [ ] Report included (PDF or Markdown)
- [ ] Config files included
- [ ] Figures included with captions
- [ ] Reproducibility commands documented
