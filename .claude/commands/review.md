Perform a comprehensive code review of recent changes:

1. Check code follows Python style, typing, mask discipline, and import-safety rules from `CLAUDE.md` and `src/CLAUDE.md`.
2. Verify proper error handling and masking (no silent broadcasting, no unchecked shapes).
3. Ensure documentation is updated where math/behavior changes (docstrings, README/report alignment).
4. Review test coverage for new functionality; confirm targeted pytest or py_compile was run.
5. Check for security risks (secrets, unsafe shell calls, network/file side effects on import).
6. Validate performance implications (tensor shapes, OOM risks, device transfers).
7. Confirm notebooks remain non-executed unless intentionally run; outputs clean.
8. Provide specific, actionable feedback with file/line references.
