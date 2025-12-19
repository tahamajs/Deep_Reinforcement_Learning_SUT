# Contributing

Thank you for your interest in contributing to CA28!

How to contribute:

- Create an issue describing the change you'd like to make.
- Open a pull request with a clear description of the changes and tests.
- Keep changes small and focused; avoid mixing unrelated features in one PR.

Testing & style:

- Run the test suite: `pytest -q`.
- Run the linter: `make lint` (uses `ruff` as configured in `pyproject.toml`).

Development environment:

- Use Python 3.10+.
- Create a virtualenv: `python -m venv .venv && source .venv/bin/activate`.
- Install packages: `pip install -r requirements.txt`.

Thanks for contributing — please follow our code style and keep tests passing.