# Card Authoring Guide (for contributors & sub-agents)

Read this, then read the gold-standard examples before writing any card.

## Required reading (in order)
1. `docs/framework/notation.md` — the unified notation. **Use these symbols exactly.**
2. `docs/framework/card-schema.md` — frontmatter fields + the 9 fixed body sections.
3. Gold-standard cards: `docs/algorithms/ols.md` and `docs/algorithms/lasso-cd.md`.
   Match their depth, structure, and tone.

## Hard rules
- **Faithfulness.** Reproduce the estimator/algorithm exactly as in the source paper, only
  translating symbols into the unified notation. Flag any default/simplification *we* add as
  `(Arena note)`.
- **One file per algorithm:** `docs/algorithms/<id>.md`, where `<id>` equals the frontmatter `id`.
- **Frontmatter must include all fields** listed in the schema:
  `id, name, aliases, family, regime, penalty, link_support, output, year, refs, status`.
- **Body must contain all 9 H2 sections, in order:**
  `Setting & assumptions`, `Estimator / objective`, `Algorithm`,
  `Hyperparameters & configuration`, `Mapping to framework`, `Complexity`,
  `Statistical guarantees`, `Variants & related`, `References`.
- **Math:** inline `$...$`, display `$$...$$` (KaTeX). Define the objective as
  $\hat\beta=\arg\min \mathcal L(\beta)+\lambda P(\beta)$ (or an estimating equation) explicitly.
- **Algorithm section:** give faithful pseudocode (a fenced ```text block) with the update rule(s).
- **Set `status: draft`** for newly authored cards (reviewer promotes to `reviewed` later).

## Citations
- Only cite **bibkeys that exist in `reference.bib`**. Find them with, e.g.:
  `Select-String -Path reference.bib -Pattern 'lasso'` or grep the title.
- If the canonical paper is **not** in `reference.bib`, cite the closest available key(s)
  (e.g. the Bühlmann–van de Geer textbook `buhlmann2011statistics`) **and** name the original
  author/year in prose. Do **not** invent bibkeys — the validator will fail.

## Validate before finishing
Run:
```
python scripts/validate_cards.py
```
Fix every `[FAIL]` for the cards you wrote (missing fields/sections, or bibkeys not in the .bib).

## Status badge line
Under the H1, include the `!!! info "At a glance"` admonition mirroring the frontmatter, as in
the gold-standard cards.
