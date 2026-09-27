## Summary

<!-- What does this change and why? Link the issue it addresses, if any. -->

## Type of change

- [ ] Bug fix
- [ ] New feature (operator, representation, loss, diagnostic, CLI option, …)
- [ ] New instance
- [ ] Documentation
- [ ] Performance

## Checklist

- [ ] `make lint` and `make test` pass on a CPU
- [ ] New behaviour is covered by tests (math, not only shapes: adjoints, analytic cases, estimators)
- [ ] Paper-derived components cite the equation or table in their docstring; `docs/paper_mapping.md` is current
- [ ] New operators and instances pair the inversion with an independent data generator (different `fidelity_tag`)
- [ ] Documentation updated (`docs/`), and `mkdocs build --strict` passes if it changed
- [ ] `CHANGELOG.md` updated under "Unreleased"
