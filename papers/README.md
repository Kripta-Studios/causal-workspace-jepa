# Papers

This directory stores the working scientific manuscript, source registries, and BibTeX metadata.
Selected official code/data may now be inspected under `gpu_12gb`, but external repositories,
papers, datasets, weights, and generated LaTeX products are not committed.

## Working paper

`causal_workspace_jepa.tex` is the source-of-truth working paper. It gives the common intervention
formalism, numerical controls, Qwen and world-model experiments, negative results, evidence levels,
novelty boundary, limitations, and exact artifact/commit references. It intentionally does **not**
claim a Qwen circuit, JEPA workspace, new Jacobian method, or SOTA result. The recurrent JEPA
action-path result is labeled calibration; its shared-denominator confound requires a new
validation-only control before any protected-test study can even be registered.

Compile from this directory so BibTeX resolves the repository's `references.bib` rather than a
system file with the same generic name:

```powershell
cd papers
latexmk -pdf -interaction=nonstopmode -halt-on-error causal_workspace_jepa.tex
```

The resulting PDF and auxiliary files are ignored. The source must compile with all citations and
cross-references resolved before a documentation milestone is committed.

All 24 required entries and 18 supplemental adversarial/prior-art entries were checked against
their primary paper, project, code, or model-card source by 2026-07-21. `docs/LITERATURE.md` records
authors/version, links, contribution, relevance, assumptions, limitations, reproduction state, and
verification date. `references.bib` contains one entry for every required source. A verified source
is not thereby a reproduced result.

The corresponding source-traceable small reproduction ran as `WM-LEWM-001`. This is not a released
checkpoint or benchmark reproduction: its modeling gate passed, while its replicated causal circuit
gate failed.
