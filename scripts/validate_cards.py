"""Validate algorithm cards: frontmatter fields, required body sections, and that
every cited bibkey exists in reference.bib.

Usage:  python scripts/validate_cards.py
Exit code is non-zero if any card fails validation.
"""
from __future__ import annotations

import pathlib
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALGO_DIR = ROOT / "docs" / "algorithms"
BIB = ROOT / "reference.bib"

REQUIRED_FM = ["id", "name", "family", "regime", "penalty", "link_support", "output", "year", "refs", "status"]
REQUIRED_SECTIONS = [
    "Setting & assumptions",
    "Estimator / objective",
    "Algorithm",
    "Hyperparameters & configuration",
    "Mapping to framework",
    "Complexity",
    "Statistical guarantees",
    "Variants & related",
    "References",
]


def bib_keys() -> set[str]:
    text = BIB.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1).strip() for m in re.finditer(r"@\w+\{([^,]+),", text)}


def split_frontmatter(text: str):
    if not text.startswith("---"):
        return None, text
    end = text.find("\n---", 3)
    if end == -1:
        return None, text
    return yaml.safe_load(text[3:end]), text[end + 4:]


def main() -> int:
    keys = bib_keys()
    errors = 0
    cards = [m for m in sorted(ALGO_DIR.glob("*.md")) if m.name != "index.md"]
    for md in cards:
        fm, body = split_frontmatter(md.read_text(encoding="utf-8"))
        prob = []
        if fm is None:
            prob.append("missing YAML frontmatter")
            fm = {}
        for f in REQUIRED_FM:
            if f not in fm:
                prob.append(f"missing frontmatter field: {f}")
        if fm.get("id") and fm["id"] != md.stem:
            prob.append(f"id '{fm.get('id')}' != filename '{md.stem}'")
        for sec in REQUIRED_SECTIONS:
            if not re.search(rf"^##\s+{re.escape(sec)}\s*$", body, re.MULTILINE):
                prob.append(f"missing section: ## {sec}")
        for k in (fm.get("refs") or []):
            if k not in keys:
                prob.append(f"bibkey not in reference.bib: {k}")
        if prob:
            errors += 1
            print(f"[FAIL] {md.name}")
            for p in prob:
                print(f"        - {p}")
        else:
            print(f"[ ok ] {md.name}")
    print(f"\n{len(cards)} cards, {errors} with problems.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
