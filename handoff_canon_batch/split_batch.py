"""
Split a canonicalization batch into N parts so team members can run Colab in parallel.

Balances parts by TOTAL CHARACTERS, not by segment count: generation time tracks
input+output length and the segments span 107..20,677 chars (mean 8,116), so an
equal-count split can leave one member waiting long after the other finishes.
Greedy longest-first bin packing; deterministic (ties broken by seg_key).

Usage (CPU):
  python handoff_canon_batch/split_batch.py                      # 173 remaining -> 2 parts
  python handoff_canon_batch/split_batch.py --parts 3
  python handoff_canon_batch/split_batch.py --input ../steps/12_joint_pipeline/outputs/canon_input.json
"""
import argparse
import json
import logging
from pathlib import Path
from string import ascii_uppercase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def split_balanced(rows, n_parts):
    """Greedy longest-first bin packing on len(raw). Returns list of n_parts row-lists."""
    order = sorted(rows, key=lambda r: (-len(r["raw"]), r["seg_key"]))
    bins = [[] for _ in range(n_parts)]
    loads = [0] * n_parts
    for r in order:
        i = min(range(n_parts), key=lambda j: (loads[j], j))
        bins[i].append(r)
        loads[i] += len(r["raw"])
    # restore input order within each part (chronological-ish; keeps resume logs readable)
    pos = {r["seg_key"]: i for i, r in enumerate(rows)}
    return [sorted(b, key=lambda r: pos[r["seg_key"]]) for b in bins]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "segments_to_canonicalize.json"))
    ap.add_argument("--out_dir", default=str(HERE))
    ap.add_argument("--parts", type=int, default=2)
    args = ap.parse_args()

    rows = json.load(open(args.input, encoding="utf-8"))
    keys = [r["seg_key"] for r in rows]
    assert len(set(keys)) == len(keys), "duplicate seg_key in input"
    total_chars = sum(len(r["raw"]) for r in rows)
    log.info("Input: %d segments, %d chars total", len(rows), total_chars)

    parts = split_balanced(rows, args.parts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for label, part in zip(ascii_uppercase, parts):
        pk = {r["seg_key"] for r in part}
        assert not (pk & seen), "parts overlap"
        seen |= pk
        path = out_dir / f"part_{label}.json"
        json.dump(part, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        chars = sum(len(r["raw"]) for r in part)
        log.info("part_%s.json : %3d segments | %8d chars (%.1f%%) | longest %d",
                 label, len(part), chars, 100 * chars / total_chars,
                 max(len(r["raw"]) for r in part))

    assert seen == set(keys), "union of parts != input"
    log.info("OK: %d parts, disjoint, union == input (%d segments)", args.parts, len(keys))


if __name__ == "__main__":
    main()
