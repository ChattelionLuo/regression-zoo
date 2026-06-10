"""
linkify_refs.py
Wraps each reference entry in algorithm pages with a clickable link
to the official publication (DOI / journal page / arXiv).

Usage:
    python scripts/linkify_refs.py          # dry-run
    python scripts/linkify_refs.py --write  # apply changes
"""
import re
import sys
import os
import glob

# ── Official URLs for every cite-key used in the algorithm cards ──────────────
# Prefer https://doi.org/... for journal articles.
# Use publisher landing page for books.
# Use https://arxiv.org/abs/... only when that IS the primary/only source.
REF_URLS = {
    # ── Classical / OLS / Ridge ──
    "Nocedal2006":          "https://link.springer.com/book/10.1007/978-0-387-40065-5",
    "hoerl1970ridge":       "https://doi.org/10.1080/00401706.1970.10488634",
    "McCullagh1989":        "https://doi.org/10.1007/978-1-4899-3242-6",
    "IWLS1987":             "https://doi.org/10.2307/1403100",   # Green 1987, Int. Stat. Rev.

    # ── Penalized / LASSO family ──
    "tibshirani1996regression": "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
    "fhht2007":             "https://doi.org/10.1214/07-AOAS131",
    "zou2005regularization":"https://doi.org/10.1111/j.1467-9868.2005.00503.x",
    "zou2006adaptive":      "https://doi.org/10.1198/016214506000000735",
    "Yuan2006group":        "https://doi.org/10.1111/j.1467-9868.2005.00532.x",
    "simon2013sparse":      "https://doi.org/10.1080/10618600.2012.681250",
    "Tibshirani2005":       "https://doi.org/10.1111/j.1467-9868.2005.00490.x",
    "zhao2006model":        "https://jmlr.org/papers/v7/zhao06a.html",
    "buhlmann2011statistics":"https://link.springer.com/book/10.1007/978-3-642-20192-9",

    # ── LARS ──
    "Efron2004lars":        "https://doi.org/10.1214/009053604000000067",

    # ── SCAD / MCP / non-convex ──
    "fan2001variable":      "https://doi.org/10.1198/016214501753382273",
    "ZouLi2008":            "https://doi.org/10.1214/009053607000000802",
    "zhang2010nearly":      "https://doi.org/10.1214/09-AOS729",
    "loh2015regularized":   "https://jmlr.org/papers/v16/loh15a.html",

    # ── Proximal / first-order (ISTA, FISTA, FOBOS, RDA, AdaGrad) ──
    "daubechies2004iterative":"https://doi.org/10.1002/cpa.20042",
    "beck2009fast":         "https://doi.org/10.1137/080716542",
    "duchi2009fobos":       "https://jmlr.org/papers/v10/duchi09a.html",
    "xiao2009dual":         "https://jmlr.org/papers/v11/xiao10a.html",
    "AdaGrad2011":          "https://jmlr.org/papers/v12/duchi11a.html",

    # ── Online / SGD / streaming ──
    "truncated_SGD_Langford_2009": "https://jmlr.org/papers/v10/langford09a.html",
    "Robbins1951":          "https://doi.org/10.1214/aoms/1177729586",
    "polyak1992acceleration":"https://doi.org/10.1137/0330046",
    "ruppert1988efficient": None,   # Cornell tech report — no stable URL
    "chen2020statistical":  "https://doi.org/10.1214/19-AOS1850",
    "Toulis2017":           "https://doi.org/10.1214/16-AOS1506",
    "Fang2019":             "https://doi.org/10.1111/sjos.12418",
    "luo2020renewable":     "https://doi.org/10.1111/rssb.12352",
    "Luo2020":              "https://doi.org/10.1111/rssb.12352",
    "Han2023DSGD":          "https://doi.org/10.1093/biomet/asad033",
    "chen2024online":       "https://doi.org/10.1080/01621459.2024.2326156",

    # ── High-dim inference (debiased LASSO, score tests…) ──
    "LDPE":                 "https://doi.org/10.1111/rssb.12026",
    "Zhang_delasso_2014":   "https://doi.org/10.1111/rssb.12026",
    "vandegeer2014":        "https://doi.org/10.1214/14-AOS1221",
    "javanmard2014confidence": "https://jmlr.org/papers/v15/javanmard14a.html",
    "hdi2015":              "https://doi.org/10.1214/15-STS527",
    "10.1111/biom.13587":   "https://doi.org/10.1111/biom.13587",
    "guo2021inference":     "https://jmlr.org/papers/v22/19-1149.html",
    "ma2021global":         "https://doi.org/10.1080/01621459.2020.1770098",
    "jankova2018semiparametric": "https://doi.org/10.1214/17-AOS1622",
    "10.1214/16-AOS1448":   "https://doi.org/10.1214/16-AOS1448",
    "belloni2016post":      "https://doi.org/10.1080/07350015.2016.1166116",
    "rakshit2024statistical": "https://arxiv.org/abs/2410.20671",
    "Deshpande2019OnlineDF":"https://doi.org/10.1080/01621459.2021.1979011",
}

# Regex: one or more <a id="ref-KEY"></a> anchors followed by the citation text
_ANCHOR_PAT = re.compile(
    r'^(?P<indent>[ \t]*-[ \t]+)'              # list bullet
    r'(?P<anchors>(?:<a id="ref-[^"]+"></a>)+)'# one or more id anchors
    r'(?P<gap>\s*)'                            # optional space
    r'(?P<text>.+)$',                          # citation text
    re.MULTILINE
)

def _first_url(anchor_block: str) -> str | None:
    """Return the first URL mapped to any key found in the anchor block."""
    keys = re.findall(r'<a id="ref-([^"]+)"></a>', anchor_block)
    for k in keys:
        url = REF_URLS.get(k)
        if url:
            return url
    return None


def linkify_text(content: str) -> tuple[str, int]:
    """Return (new_content, n_replacements)."""
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        anchor_block = m.group("anchors")
        text = m.group("text")

        # Skip if already wrapped in a link
        if text.startswith('<a href='):
            return m.group(0)

        url = _first_url(anchor_block)
        if url is None:
            return m.group(0)   # no URL known — leave as-is

        count += 1
        return (
            f'{m.group("indent")}{anchor_block}{m.group("gap")}'
            f'<a href="{url}" class="ref-link" target="_blank" '
            f'rel="noopener noreferrer">{text}</a>'
        )

    new_content = _ANCHOR_PAT.sub(_replace, content)
    return new_content, count


def main():
    write = "--write" in sys.argv
    algo_dir = os.path.join(
        os.path.dirname(__file__), "..", "docs", "algorithms"
    )
    md_files = sorted(glob.glob(os.path.join(algo_dir, "*.md")))

    total = 0
    for path in md_files:
        with open(path, encoding="utf-8") as f:
            original = f.read()
        updated, n = linkify_text(original)
        if n:
            total += n
            print(f"  {n:2d} link(s) → {os.path.basename(path)}")
            if write and updated != original:
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(updated)
        else:
            print(f"   0 links  {os.path.basename(path)}")

    print(f"\n{'Applied' if write else 'Would apply'} {total} link(s) across {len(md_files)} files.")
    if not write:
        print("Run with --write to apply changes.")


if __name__ == "__main__":
    main()
