#!/usr/bin/env python3
"""
Apply the three reviewer-round-3 edits to paper/aic_results.md:
  1. McNemar significance line (Overall section)
  2. rfwe arithmetic line (Overall section)
  3. open-question / benchmark-limitation paragraph (Reading the per-type results)

Safe: each replacement asserts the target text exists exactly once before
editing. If any anchor is not found verbatim, the script aborts and changes
nothing, so you can fix the anchor and re-run.

Usage:
    python3 apply_aic_edits.py            # edits paper/aic_results.md in place
    python3 apply_aic_edits.py --dry-run  # prints what would change, writes nothing
"""
import sys
from pathlib import Path

PATH = Path("paper/aic_results.md")

EDITS = [
    # (label, find, replace)
    (
        "1. McNemar significance line",
        "This is a retrieval\n*outcome*, not evidence of graph *causality* \u2014 isolating the latter still requires the\nablations listed in the post.",
        "The discordance is one-directional: of the 29 questions where the systems disagree on\nturn-AIC, ActiveGraph wins 25 and loses 4 (exact McNemar, p = 0.0001), so the +4.5-point\nretrieval edge is statistically established rather than a point estimate. This is a retrieval\n*outcome*, not evidence of graph *causality* \u2014 isolating the latter still requires the\nablations listed in the post.",
    ),
    (
        "2. rfwe arithmetic line",
        "it retrieves enough additional evidence\nto win on accuracy despite a marginally higher reader-fumble rate.",
        "it retrieves enough additional evidence\nto win on accuracy despite a marginally higher reader-fumble rate. The two effects are the\nsame order of magnitude: +21 net turn-AIC hits (25 won, 4 lost) against +8\nreader-failures-with-evidence (35 vs. 27). The significant retrieval advantage is partly\nabsorbed by reader-fumbles on the extra context it surfaces, which is why end-to-end QA\nlands at +1.9 rather than tracking the full retrieval gap.",
    ),
    (
        "3. open-question paragraph",
        "## Caveat",
        "## What the benchmark cannot measure\n\nThe substrate's retrieval ceiling is already high exactly where semantic memory is meant to\nhelp \u2014 knowledge-update sits at ~95% turn / ~99% session recall for both systems \u2014 so the\nbottleneck there is reader reconciliation, not retrieval. Whether typed superseded-fact\nrepresentation actually eases that reconciliation is precisely what LongMemEval-S cannot\nmeasure, and is the real open question for the next experiment.\n\n## Caveat",
    ),
]


def main():
    dry = "--dry-run" in sys.argv
    if not PATH.exists():
        sys.exit(f"ERROR: {PATH} not found. Run from the repo root (ag-run).")

    text = PATH.read_text()
    original = text

    for label, find, replace in EDITS:
        n = text.count(find)
        if n != 1:
            sys.exit(
                f"ABORT on edit '{label}': expected to find the anchor exactly once, "
                f"found {n}. No changes written. "
                f"The file text may differ from what was expected; "
                f"paste the relevant section so the anchor can be fixed."
            )
        text = text.replace(find, replace)
        print(f"OK: applied {label}")

    if dry:
        print("\n--dry-run: no file written. All 3 anchors matched cleanly.")
        return

    PATH.write_text(text)
    print(f"\nWrote {PATH} ({len(original)} -> {len(text)} chars).")
    print("Review with:  git diff paper/aic_results.md")


if __name__ == "__main__":
    main()
