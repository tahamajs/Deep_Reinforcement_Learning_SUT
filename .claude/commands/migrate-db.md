Create a database migration: $ARGUMENTS

1. Create migration file (tooling dependent; confirm command first).
2. Write migration up/down with safety checks; avoid data loss.
3. Review migration for reversibility and lock safety.
4. Test migration locally (e.g., `bun db:migrate` or project-specific command if added).
5. Inspect schema after migration.
6. Run targeted tests touching the affected data paths.
7. Document breaking changes and rollout steps.
8. Commit migration file with clear message.
9. Never run migrations on production without explicit approval.
