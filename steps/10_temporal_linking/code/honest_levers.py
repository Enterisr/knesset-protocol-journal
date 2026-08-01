"""A fully matched, fully streaming-honest ablation of the three linking levers.

WHY THIS EXISTS
    The lever table quoted in the write-up mixed fitting protocols: the baseline
    (first-span + threshold) and the centroid row came from `streaming_eval_results.json`,
    which fits the transform in BATCH over the whole corpus, while the margin row (0.190)
    came from `streaming_honest_compare.json`, which refits on the observed prefix. Those
    are not comparable -- and reporting them side by side is precisely the error Step 10
    documents elsewhere.

    This recomputes all three rows on one transform (`center`), one growth model (oracle,
    as in Steps 07/08), and one fitting protocol (honest prefix refit), so the only thing
    varying between rows is the lever under test.

Usage (CPU, ~3 min):  python steps/10_temporal_linking/code/honest_levers.py
Output: outputs/honest_levers.json
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07 = STEP_DIR.parent / "07_gold_eval" / "outputs"


def honest_records(e5, ids, kind, entry_repr):
    """Oracle-growth journal, transform refit on the observed prefix at every step.

    entry_repr: 'first'    -- entry is its first span, frozen (the baseline)
                'centroid' -- entry is the running mean of its members
    """
    topics = [r["topic"] for r in ids]
    dates = [datetime.fromisoformat(r["date"]) for r in ids]
    order = sorted(range(len(ids)),
                   key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)

    entry_members, entry_last, topic_to_entry, records = [], [], {}, []
    for p, i in enumerate(order):
        Xt = se.transform(e5[order[: p + 1]], kind)   # honest: prefix only
        x, now = Xt[p], dates[i]
        if entry_members:
            cos = np.empty(len(entry_members))
            for k, mem in enumerate(entry_members):
                v = Xt[mem[:1]].mean(0) if entry_repr == "first" else Xt[mem].mean(0)
                v = v / (np.linalg.norm(v) + 1e-12)
                cos[k] = v @ x
            t = topics[i]
            is_rep = first_seen[t] != i
            records.append({"gold_is_repeat": bool(is_rep), "cos": cos,
                            "gold_entry": topic_to_entry.get(t, -1) if is_rep else -1,
                            "age": np.array([(now - u).days for u in entry_last],
                                            dtype=float)})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_members)
            entry_members.append([p]); entry_last.append(now)
        else:
            e = topic_to_entry[t]
            entry_members[e].append(p); entry_last[e] = now
    return records


def main():
    ids = json.load(open(STEP07 / "embeddings" / "ids.json", encoding="utf-8"))
    e5 = np.load(STEP07 / "embeddings" / "e5.npy")

    out = {"transform": "center", "fit": "honest (prefix refit)",
           "growth": "oracle (as Steps 07/08)", "n_segments": len(ids), "rows": {}}

    for name, entry_repr, rule in [("first_span+threshold", "first", "threshold"),
                                   ("centroid+threshold", "centroid", "threshold"),
                                   ("centroid+margin", "centroid", "margin")]:
        recs = honest_records(e5, ids, "center", entry_repr)
        rows = se.sweep_threshold(recs) if rule == "threshold" else se.sweep_margin(recs)
        out["rows"][name] = se.summarize(rows)
        b, g = out["rows"][name]["best_f1"], out["rows"][name]["best_f1_fmr<=5%"]
        log.info("%-22s best F1=%.4f (P=%.4f R=%.4f) | FMR<=5%%: R=%.4f",
                 name, b["f1"], b["precision"], b["recall"],
                 g["recall"] if g else float("nan"))

    p = OUT / "honest_levers.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", p)


if __name__ == "__main__":
    main()
