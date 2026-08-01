"""
Merge the per-member Colab outputs into the single `canonical_gold.json` that
`steps/12_joint_pipeline/code/build_segments.py` consumes.

Inputs (whatever is present):
  handoff_canon_batch/canonical_part_*.json   <- the Colab runs (flat list, keyed by seg_key)
  --extra PATH ...                            <- any earlier canonicalization of the other
                                                 350 segments, same flat shape
Output:
  steps/12_joint_pipeline/outputs/canonical_gold.json

The report is the point of this script, not the merge. Step 13's `rq_eval.py` sets
`comparison_valid: false` when more than 10% of the 523 segments fall back to raw text,
because at that point the canonical arm is a raw/canonical blend and a null result means
"the pass was incomplete", not "canonicalization does not work". This prints that number
before anyone spends a GPU job on it, plus a `source` histogram -- segments rewritten by
different models are a stated limitation the Step 13 report has to carry.

Usage (CPU):
  python handoff_canon_batch/merge_canonical.py
  python handoff_canon_batch/merge_canonical.py --extra path/to/first_350.json
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CANON_INPUT = REPO / "steps" / "12_joint_pipeline" / "outputs" / "canon_input.json"
OUT_PATH = REPO / "steps" / "12_joint_pipeline" / "outputs" / "canonical_gold.json"
FALLBACK_GATE = 0.10  # must match rq_eval.py's validity gate


def load_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):  # tolerate {'segments': [...]} (notebook 04's shape)
        data = data.get("segments", [])
    return [r for r in data if isinstance(r, dict) and "seg_key" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional canonicalization files (e.g. the earlier 350)")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    sources = sorted(HERE.glob("canonical_part_*.json")) + [Path(p) for p in args.extra]
    if not sources:
        log.error("no canonical_part_*.json in %s and no --extra given -- nothing to merge", HERE)
        return
    for p in sources:
        if not p.exists():
            log.error("missing %s", p)
            return

    merged, dupes = {}, Counter()
    for p in sources:
        recs = load_records(p)
        n_new = 0
        for r in recs:
            if r["seg_key"] in merged:
                dupes[r["seg_key"]] += 1
                # keep the record that actually has text
                if not r.get("canonical", "").strip():
                    continue
            merged[r["seg_key"]] = r
            n_new += 1
        log.info("%-38s %4d records", p.name, len(recs))

    # --- report ----------------------------------------------------------------------
    all_keys = [r["seg_key"] for r in json.loads(CANON_INPUT.read_text(encoding="utf-8"))]
    usable = {k: r for k, r in merged.items() if r.get("canonical", "").strip()}
    covered = [k for k in all_keys if k in usable]
    unknown = set(merged) - set(all_keys)
    n_fallback = len(all_keys) - len(covered)
    frac_fallback = n_fallback / len(all_keys)

    print()
    print(f"  target corpus     : {len(all_keys)} segments (canon_input.json)")
    print(f"  merged records    : {len(merged)}  ({len(usable)} with non-empty canonical)")
    print(f"  covered           : {len(covered)}  ({100 * len(covered) / len(all_keys):.1f}%)")
    print(f"  fall back to raw  : {n_fallback}  ({100 * frac_fallback:.1f}%)")
    if dupes:
        print(f"  duplicate seg_key : {len(dupes)} (kept the one with text)")
    if unknown:
        print(f"  NOT in canon_input: {len(unknown)} (ignored) e.g. {sorted(unknown)[:3]}")

    src_hist = Counter(r.get("source") or "(unrecorded)" for r in usable.values())
    print("\n  source histogram  :")
    for s, n in src_hist.most_common():
        print(f"      {n:4d}  {s}")
    if len(src_hist) > 1:
        print("      ^ more than one rewriter produced the canonical arm. Not fatal, but it")
        print("        is a confound the Step 13 report must state explicitly.")

    ratios = sorted(r["ratio"] for r in usable.values() if isinstance(r.get("ratio"), (int, float)))
    if ratios:
        n_short = sum(r < 0.4 for r in ratios)
        print(f"\n  length ratio      : min {ratios[0]:.2f} | median {ratios[len(ratios)//2]:.2f} "
              f"| max {ratios[-1]:.2f}")
        print(f"      {n_short} segments below 0.4 -- the prompt forbids summarizing "
              f"('אל תסכם ואל תקצר')")
    n_badjson = sum(1 for r in usable.values() if r.get("json_ok") is False)
    if n_badjson:
        print(f"  json failures     : {n_badjson} (canonical text kept, but subject/decision empty)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = [merged[k] for k in all_keys if k in merged]
    out.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote {out} ({len(ordered)} records)")

    print()
    if frac_fallback > FALLBACK_GATE:
        log.warning("COVERAGE TOO LOW: %.1f%% of segments fall back to raw text (gate is %.0f%%).",
                    100 * frac_fallback, 100 * FALLBACK_GATE)
        log.warning("rq_eval.py will set comparison_valid=false. A null result under this "
                    "coverage means the canonicalization pass is incomplete, NOT that "
                    "canonicalization is ineffective. Find the missing segments first.")
    else:
        log.info("Coverage passes the %.0f%% gate. Next: "
                 "python steps/12_joint_pipeline/code/build_segments.py, "
                 "then sbatch sbatch/13_canonical_rq.sh", 100 * FALLBACK_GATE)


if __name__ == "__main__":
    main()
