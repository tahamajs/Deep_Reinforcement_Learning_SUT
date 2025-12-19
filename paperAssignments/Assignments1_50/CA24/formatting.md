# Formatting & linting

Recommended: black, isort, flake8. Use pre-commit to enforce formatting on commits.

Example pre-commit config:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
```
