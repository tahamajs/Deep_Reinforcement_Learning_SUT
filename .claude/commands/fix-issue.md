Analyze and fix GitHub issue: $ARGUMENTS

Steps:

1. Fetch details: `gh issue view $ARGUMENTS`
2. Identify scope and relevant modules/assignments; read nearest `CLAUDE.md` and `AGENTS.md`.
3. Search codebase with `rg -n "<term>" src assignments-hard` scoped to the issue.
4. Plan changes respecting masking, import-safety, and theory alignment.
5. Implement fix following established patterns; keep configs/dataclasses updated.
6. Add/update tests (targeted pytest) or py_compile checks for touched files.
7. Run `python -m ruff check <files>` (if available) and targeted pytest.
8. Draft Conventional Commit message summarizing intent/scope.
9. If needed, prepare PR with `gh pr create` and note checks run.
10. Include a short reviewer note on risk areas, test coverage, and any skipped checks.
