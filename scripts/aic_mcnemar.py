# scripts/aic_mcnemar.py
"""
McNemar test on per-question turn-AIC hit/miss for the two replayed cells
behind the paper's main table (matrix 20260524T050742Z).

Reads the per-question vectors already written by answer_in_context.py
(aic_results.json -> "per_question") for ActiveGraph and dense turn-RAG,
pairs them by question_id, builds the 2x2 discordant table, and runs an
exact (binomial) McNemar test. No new experiments; pure post-processing.

Hit definition mirrors the aggregate "answer-in-context" used in the paper:
prefer turn evidence; if turn_hit is None, fall back to session_hit.
(Same precedence as answer_in_context.py.)

Usage:
    uv run python scripts/aic_mcnemar.py
"""
import json
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

AG = Path(
    "runs/20260524T050742Z__activegraph-det-embedding__s__g-n/"
    "a__activegraph-det-embedding__s__full/aic_results.json"
)
RAG = Path(
    "runs/20260524T050742Z__rag-dense__s__g-turn__rag-dense__s__full/"
    "aic_results.json"
)


def load_rows(p: Path) -> dict[str, dict]:
    if not p.exists():
        raise SystemExit(f"Missing file: {p}\n(run from the repo root, e.g. ~/ag-run)")
    rows = json.loads(p.read_text())["per_question"]
    out = {}
    for r in rows:
        qid = r.get("question_id") or r.get("qid")
        out[qid] = r
    return out


def aic_hit(r: dict) -> bool:
    # mirror answer_in_context.py: turn first, else session
    th = r.get("turn_hit")
    if th is True:
        return True
    if th is None:
        return bool(r.get("session_hit"))
    return False  # th is False


def main():
    ag, rag = load_rows(AG), load_rows(RAG)
    qids = sorted(set(ag) & set(rag))
    only_ag, only_rag = set(ag) - set(rag), set(rag) - set(ag)
    if only_ag or only_rag:
        print(f"WARNING: unmatched qids  ag-only={len(only_ag)}  rag-only={len(only_rag)}")
    print(f"paired questions: {len(qids)}")

    # 2x2: rows = AG (hit/miss), cols = RAG (hit/miss)
    a = b = c = d = 0  # a=both hit, b=AG hit/RAG miss, c=AG miss/RAG hit, d=both miss
    for q in qids:
        ah, rh = aic_hit(ag[q]), aic_hit(rag[q])
        if ah and rh:
            a += 1
        elif ah and not rh:
            b += 1
        elif not ah and rh:
            c += 1
        else:
            d += 1

    table = [[a, b], [c, d]]
    print("\n2x2 contingency (turn-AIC, answer-in-context hit):")
    print("               RAG hit   RAG miss")
    print(f"  AG hit      {a:>7}   {b:>8}")
    print(f"  AG miss     {c:>7}   {d:>8}")
    print(f"\nAG hits  = {a + b}   ({(a + b) / len(qids) * 100:.1f}%)")
    print(f"RAG hits = {a + c}   ({(a + c) / len(qids) * 100:.1f}%)")
    print(f"net discordant in AG's favor (b - c) = {b - c}   (b={b}, c={c})")

    # exact binomial McNemar (b+c is small -> don't use chi-square approx)
    res = mcnemar(table, exact=True)
    print(f"\nExact McNemar (binomial on discordant pairs b={b}, c={c}):")
    print(f"  statistic = {res.statistic:.4f}")
    print(f"  p-value   = {res.pvalue:.4f}")
    verdict = "statistically established" if res.pvalue < 0.05 else "NOT statistically established"
    print(f"\n=> The +AIC retrieval edge is {verdict} at alpha=0.05 (p={res.pvalue:.4f}).")


if __name__ == "__main__":
    main()
