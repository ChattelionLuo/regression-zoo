# Algorithm-Card Schema

Every algorithm is documented in **one Markdown file** under `docs/algorithms/<id>.md`.
A card has two parts: **YAML frontmatter** (machine-readable metadata, consumed by the future
arena) and a **fixed-section body** (human-readable, faithful mathematics).

> **Faithfulness rule.** A card must reproduce the estimator/algorithm *as defined in its
> source(s)*, only translating symbols into the [unified notation](notation.md). Any
> deviation, simplification, or default we add is flagged explicitly as *"(Arena note)"*.

---

## Frontmatter (required fields)

```yaml
---
id: lasso-cd                      # unique slug, matches filename
name: "Lasso via Coordinate Descent"
aliases: ["glmnet lasso", "pathwise coordinate descent (lasso)"]
family: penalized-batch           # Axis A tag (taxonomy)
regime: [high-dim, low-dim]       # Axis B tags
penalty: lasso                    # Axis C tag
link_support: [identity, logit, log, any]   # links the method handles
output: point                     # point | path | point+inference
year: 2007
refs: [friedman2007pathwise, tibshirani1996regression]   # bibkeys in reference.bib
status: reviewed                  # stub | draft | reviewed
---
```

| Field | Meaning |
|---|---|
| `id` | unique slug; must equal the filename without `.md` |
| `name` | human-readable title |
| `aliases` | alternative names / package names |
| `family` | primary family tag (Axis A) |
| `regime` | list of data-regime tags (Axis B) |
| `penalty` | penalty/structure tag (Axis C) |
| `link_support` | links handled (`identity`, `logit`, `log`, `probit`, `any`, ...) |
| `output` | `point` (just $\widehat\beta$), `path` (over $\lambda$), or `point+inference` |
| `year` | year of the primary reference |
| `refs` | list of bibkeys present in `reference.bib` |
| `status` | maturity: `stub` < `draft` < `reviewed` |

---

## Body sections (fixed order)

A card body uses exactly these H2 sections, in order. Empty sections are allowed in a `stub`.

1. **Setting & assumptions** — data regime, link(s), structural assumptions
   (sparsity, design conditions, dispersion known/unknown).
2. **Estimator / objective** — the *exact* definition of $\widehat\beta$ in unified notation:
   the objective $\mathcal L + \lambda P$ or estimating equation. This is the mathematical heart.
3. **Algorithm** — faithful pseudocode of how $\widehat\beta$ is computed (updates, stopping rule).
4. **Hyperparameters & configuration** — every knob: $\lambda$ (and selection rule),
   step sizes, tolerances, weights, initialization, standardization.
5. **Mapping to framework** — the $(X, y, g, \text{config}) \to \widehat\beta$ contract:
   inputs, outputs, supported links, and any required preprocessing. This is the Phase-2 spec.
6. **Complexity** — per-iteration and total cost; memory (esp. for streaming).
7. **Statistical guarantees** *(brief)* — consistency / rates / inference validity, with
   the governing conditions, cited to the source. Keep it short; this is an algorithm zoo,
   not a proof repository.
8. **Variants & related** — links to sibling cards.
9. **References** — the source papers (bibkeys), with one-line notes.

---

## Header block convention

Immediately under the H1 title, cards include a metadata admonition rendered from the
frontmatter values, e.g.:

```markdown
# Lasso via Coordinate Descent

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** high-dim / low-dim · **Penalty:** lasso ·
    **Output:** path · **Links:** identity, logit, log · **Status:** reviewed ·
    **Refs:** friedman2007pathwise, tibshirani1996regression
```

(The admonition duplicates frontmatter for now; a future build script can auto-generate it
from the YAML — see `scripts/`.)

---

## Minimal stub template

```markdown
---
id: my-method
name: "My Method"
aliases: []
family: penalized-batch
regime: [high-dim]
penalty: lasso
link_support: [any]
output: point
year: 2020
refs: [somekey2020]
status: stub
---

# My Method

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** high-dim · **Penalty:** lasso ·
    **Output:** point · **Links:** any · **Status:** stub · **Refs:** somekey2020

## Setting & assumptions
## Estimator / objective
## Algorithm
## Hyperparameters & configuration
## Mapping to framework
## Complexity
## Statistical guarantees
## Variants & related
## References
```
