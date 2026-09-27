---
hide:
  - toc
description: The research behind nefi — the NeTMY and NeFTY papers, the thesis that the representation is the prior, and the findings the library reports as they are.
---

# Research

nefi is the software form of two papers from the [CAB Lab at Princeton University](https://cablab.scholar.princeton.edu). It keeps their
methods as defaults, their numbers as configuration fields, their explanations as diagnostics and
their evaluation protocol as a benchmark harness, and generalizes the recipe to any
differentiable instrument.

<div class="grid cards" markdown>

-   :material-file-document-multiple-outline: __[The two papers](papers.md)__

    ---

    NeTMY (NV-center noise sensing) and NeFTY (thermal tomography): what each one solves,
    what it contributes, what nefi implements, and how to cite them.

-   :material-shape-outline: __[The representation is the prior](thesis.md)__

    ---

    A parameterized update is the raw gradient filtered by $G_\theta = J_\theta J_\theta^\top$.
    Why that single fact explains center collapse, spectral bias and the choice of
    representation — and how nefi measures it.

-   :material-alert-circle-outline: __[Limitations and caveats](../guides/limitations.md)__

    ---

    The model-error floor of the heat operator, when a TV grid matches the neural field, what an
    accepted refinement certifies, and what tuning cannot supply.

-   :material-map-marker-path: __[Paper → code mapping](../paper_mapping.md)__

    ---

    Every equation, proposition, table and key figure of both papers, mapped to the code that
    implements, measures or reproduces it.

-   :material-file-cog-outline: __[Design document](../DESIGN.md)__

    ---

    The architecture contract: design principles, package layout and the exact signatures every
    module builds against.

-   :material-console-network-outline: __[Reproducing the papers](../reproduce_papers.md)__

    ---

    The server runbook: exact commands for the headline tables, their cost in GPU-hours, and
    the caveats to check first.

</div>
