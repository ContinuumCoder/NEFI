---
hide:
  - toc
description: Reference material — the API, the command line, configuration files, the changelog, contributing and the license.
---

# Reference

<div class="grid cards" markdown>

-   :material-api: __[API](../api.md)__

    ---

    Every public module, class and function, generated from the docstrings: core, fields,
    operators, physics, losses, solve, metrics, diagnostics, bench, autotune, baselines,
    instances, viz and utils.

-   :material-console-line: __[Command line](cli.md)__

    ---

    `nefi list · run · bench · bench-merge · diagnose · autotune · ablate · sweep` with every
    option and worked examples.

-   :material-file-cog-outline: __[Configuration files](config.md)__

    ---

    YAML layouts, the curriculum schema, `--set` overrides, smoke presets, plugins and what a
    run writes.

-   :material-history: __[Changelog](changelog.md)__

    ---

    Notable changes, following Keep a Changelog.

-   :material-account-group-outline: __[Contributing](contributing.md)__

    ---

    Adding an operator, a field, a loss or an instance; testing and style rules.

-   :material-scale-balance: __[License & citation](license.md)__

    ---

    MIT License, CAB Lab, Princeton University, and how to cite the papers.

</div>

The [design document](../DESIGN.md) specifies the contracts every module builds against, and the
[paper → code mapping](../paper_mapping.md) traces every equation of the papers to the code.
