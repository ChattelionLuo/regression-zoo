# GLM Algorithm Arena

A comprehensive, faithful **encyclopedia of concrete algorithms (solvers) for Generalized
Linear Models** — and, in later phases, an **arena** where those algorithms compete on many
datasets.

Every algorithm is reduced to one mapping and documented under a single unified notation:

> Given `X` (covariates), `y` (response), and a link `g`, produce a coefficient estimate `β̂`
> — under a stated configuration of hyperparameters.

## Project phases

1. **Documentation (current).** Faithful algorithm *cards* — exact mathematics only.
2. **Implementation (later).** Python solvers behind `fit(X, y, link, **config) -> beta_hat`.
3. **Arena / analysis (later).** Run all solvers across many datasets; study performance,
   agreement, and correlations between outputs.

## Layout

| Path | Purpose |
|---|---|
| `docs/framework/notation.md` | unified mathematical framework & notation (the backbone) |
| `docs/framework/taxonomy.md` | how algorithms are categorized |
| `docs/framework/card-schema.md` | the algorithm-card template |
| `docs/algorithms/*.md` | one card per algorithm |
| `registry/algorithms.yaml` | aggregated card metadata (for the future arena) |
| `src/glmzoo/` | Python package skeleton (Phase 2) |
| `scripts/` | build/validation helpers |
| `reference.bib` | bibliography (cards cite these keys) |

## Build the website locally

```powershell
pip install -r requirements.txt
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site into ./site
```

Math is rendered with KaTeX; cards use `$...$` / `$$...$$`.

## Adding an algorithm card

1. Copy the stub from `docs/framework/card-schema.md` to `docs/algorithms/<id>.md`.
2. Fill the fixed sections **faithfully** from the source paper, in the unified notation.
3. Ensure every `refs:` bibkey exists in `reference.bib`.
4. Add the card to the catalogue in `docs/algorithms/index.md`.

See `docs/framework/card-schema.md` for the full schema and the faithfulness rule.
