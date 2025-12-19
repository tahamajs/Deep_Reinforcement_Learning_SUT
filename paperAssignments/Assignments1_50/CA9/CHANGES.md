Changelog (short)

- 2025-12-18: Fixed Polyak soft-update in `src/models/q_ensemble.py` to use the canonical update target <- (1 - tau)*target + tau*source. This corrects subtle target network drift.
- 2025-12-18: Added "Getting Started" / install / quickstart and testing instructions to `README.md` to support easy reproduction and smoke tests.
- 2025-12-18: Added a Reproducibility section to `report.tex` with commands to reproduce training, evaluation, and compiling the report.
- 2025-12-18: Minor documentation and housekeeping updates (license/citation notes, repository status message).

If you'd like, I can also:
- Add an automated `requirements.txt` or `environment.yml` for reproducibility.
- Add a GitHub Actions workflow to run the tests automatically on push.
