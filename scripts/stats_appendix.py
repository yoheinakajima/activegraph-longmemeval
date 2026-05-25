import argparse, json, math, random
from pathlib import Path
random.seed(42)

def find_eval(d):
    c = list(Path(d).glob("**/*eval-results*"))
    return c[0] if c else None

def load(d):
    f = find_eval(d)
    if not f:
        raise FileNotFoundError(str(d))
    out = {}
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        o = json.loads(ln)
        qid = o.get("question_id") or o.get("id")
        if qid is None:
            continue
        lab = None
        for k in ("autoeval_label","label","is_correct","correct","judge"):
            if k in o:
                v = o[k]
                if isinstance(v, dict):
                    v = v.get("label", v.get("is_correct", v.get("correct")))
                lab = v
                break
        if lab is None:
            continue
        if isinstance(lab, str):
            lab = lab.strip().lower() in ("true","yes","correct","1")
        out[qid] = 1 if lab else 0
    return out

def qtypes(d):
    for m in Path(d).glob("**/manifest.json"):
        man = json.loads(m.read_text())
        qs = man.get("queries", [])
        if qs and "question_type" in qs[0]:
            return {q["question_id"]: q["question_type"] for q in qs}
    return {}

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k/n
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)))/den
    return (c-h, c+h)

def mcnemar(a, b):
    sh = set(a) & set(b)
    b10 = sum(1 for q in sh if a[q]==1 and b[q]==0)
    b01 = sum(1 for q in sh if a[q]==0 and b[q]==1)
    n = b10+b01
    if n == 0:
        return (b01, b10, 1.0)
    k = min(b01, b10)
    cdf = sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return (b01, b10, min(1.0, 2*cdf))

def boot(a, b, iters=10000):
    sh = sorted(set(a) & set(b))
    n = len(sh)
    obs = (sum(a[q] for q in sh) - sum(b[q] for q in sh))/n
    ds = []
    for _ in range(iters):
        s = [sh[random.randrange(n)] for _ in range(n)]
        ds.append((sum(a[q] for q in s) - sum(b[q] for q in s))/n)
    ds.sort()
    return (obs, ds[int(0.025*iters)], ds[int(0.975*iters)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--dataset", default="s")
    ap.add_argument("--focus", default="activegraph-det-embedding")
    a = ap.parse_args()
    rows = json.loads(Path(a.matrix).read_text())
    cells = {}
    for r in rows:
        if r["dataset"] != a.dataset:
            continue
        lab = r["system"] + ("" if r["granularity"] in ("n/a","na") else "/"+r["granularity"])
        cells[lab] = r["run_dir"]
    corr = {}
    for lab, d in cells.items():
        try:
            corr[lab] = load(d)
        except Exception as e:
            print("  [skip]", lab, e)
    print("\nDataset:", a.dataset, "| systems:", len(corr), "\n")
    print("=== Per-system accuracy (Wilson 95% CI) ===")
    for lab in sorted(corr, key=lambda L: -sum(corr[L].values())/max(1,len(corr[L]))):
        v = corr[lab]; n = len(v); k = sum(v.values()); lo, hi = wilson(k, n)
        print("  %-32s %d/%d = %.3f  [%.3f, %.3f]" % (lab, k, n, k/n, lo, hi))
    foc = a.focus if a.focus in corr else next((L for L in corr if L.startswith(a.focus)), None)
    if not foc:
        print("focus not found"); return
    print("\n=== Paired vs", foc, "(b10=focus right & base wrong; b01=reverse) ===")
    for lab in corr:
        if lab == foc:
            continue
        b01, b10, p = mcnemar(corr[foc], corr[lab])
        obs, lo, hi = boot(corr[foc], corr[lab])
        sig = "  *" if p < 0.05 else ""
        print("  vs %-30s net %+d (b10=%d,b01=%d) McNemar p=%.3f%s  d=%+.3f [%+.3f,%+.3f]" % (lab, b10-b01, b10, b01, p, sig, obs, lo, hi))
    qt = {}
    for d in cells.values():
        qt.update(qtypes(d))
    if qt:
        print("\n=== Per-type net vs", foc, "===")
        for lab in corr:
            if lab == foc:
                continue
            sh = set(corr[foc]) & set(corr[lab])
            by = {}
            for q in sh:
                t = qt.get(q, "?"); x = by.setdefault(t, [0,0,0])
                x[0]+=corr[foc][q]; x[1]+=corr[lab][q]; x[2]+=1
            print("\n  %s vs %s:" % (foc, lab))
            for t in sorted(by):
                fc, bc, n = by[t]
                print("    %-28s %d/%d vs %d/%d  net %+d" % (t, fc, n, bc, n, fc-bc))

main()
