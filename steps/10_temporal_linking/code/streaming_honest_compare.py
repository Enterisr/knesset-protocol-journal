"""
Step 10 — which transforms survive streaming-honest fitting?

whiten50 collapsed 0.250 -> 0.094 when refit only on spans-seen-so-far (it needs a
50-dim covariance estimate that doesn't exist early in the stream). This checks the
CHEAPER transforms that need far fewer samples to estimate:
  center  -- just the running mean (estimable from step 2)
  abtt1   -- running mean + top-1 direction (estimable very early)
  abtt2   -- running mean + top-2 directions
all with centroid representation + the margin rule. Reports batch vs honest so we
can pick a flagship that does NOT leak future information.

Usage: python steps/10_temporal_linking/code/streaming_honest_compare.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

import streaming_eval as se

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
STEP07_OUT = STEP_DIR.parent / "07_gold_eval" / "outputs"


def honest_centroid_records(e5, ids, kind):
    topics = [r["topic"] for r in ids]
    dates = [datetime.fromisoformat(r["date"]) for r in ids]
    order = sorted(range(len(ids)), key=lambda i: (ids[i]["date"], ids[i]["doc_id"], ids[i]["seg_idx"]))
    first_seen = {}
    for i in order:
        first_seen.setdefault(topics[i], i)
    entry_members, entry_last, topic_to_entry = [], [], {}
    records = []
    for p, i in enumerate(order):
        seen = order[: p + 1]
        Xt = se.transform(e5[seen], kind)
        x = Xt[p]
        now = dates[i]
        if entry_members:
            cos = np.empty(len(entry_members))
            for k, mem in enumerate(entry_members):
                v = Xt[mem].mean(0); v /= np.linalg.norm(v) + 1e-12
                cos[k] = v @ x
            t = topics[i]; is_rep = first_seen[t] != i
            # age is measured against the journal as it stands BEFORE this arrival
            # is filed, so it uses no information the stream hasn't delivered yet.
            records.append({"gold_is_repeat": bool(is_rep), "cos": cos,
                            "gold_entry": topic_to_entry.get(t, -1) if is_rep else -1,
                            "age": np.array([(now - u).days for u in entry_last], dtype=float)})
        t = topics[i]
        if first_seen[t] == i:
            topic_to_entry[t] = len(entry_members)
            entry_members.append([p]); entry_last.append(now)
        else:
            e = topic_to_entry[t]
            entry_members[e].append(p); entry_last[e] = now
    return records


def main():
    e5 = np.load(STEP07_OUT / "embeddings" / "e5.npy")
    ids = json.load(open(STEP07_OUT / "embeddings" / "ids.json", encoding="utf-8"))
    out = {}
    for kind in ["center", "abtt1", "abtt2"]:
        b_recs = se.stream_records(se.transform(e5, kind), ids, "centroid")
        h_recs = honest_centroid_records(e5, ids, kind)

        batch = se.summarize(se.sweep_margin(b_recs))
        honest = se.summarize(se.sweep_margin(h_recs))
        out[kind] = {"batch": batch["best_f1"], "honest": honest["best_f1"],
                     "batch_fmr5": batch["best_f1_fmr<=5%"], "honest_fmr5": honest["best_f1_fmr<=5%"]}
        log.info("%-7s  batch F1=%.3f (R=%.3f) | honest F1=%.3f (R=%.3f, fmr5_R=%s)",
                 kind, batch["best_f1"]["f1"], batch["best_f1"]["recall"],
                 honest["best_f1"]["f1"], honest["best_f1"]["recall"],
                 f'{honest["best_f1_fmr<=5%"]["recall"]:.3f}' if honest["best_f1_fmr<=5%"] else "n/a")

        # ---- per-arrival score standardization (additive keys; margin rows above
        #      are untouched so the published batch/honest table still reads the same)
        cold = se.coldstart_stats(h_recs)
        assert cold["n_coldstart_skipped_positive"] == 0, cold
        out[kind]["coldstart"] = cold
        # Locality sweep: m = size of the null. m=2 is the margin rule's
        # neighbourhood, m=None is the whole journal.
        rules = [(f"zscore_m{m}", se.sweep_zscore, m) for m in [2, 3, 5, 10, 25, 50]]
        rules += [("zscore", se.sweep_zscore, None),
                  ("zscore_sidak", se.sweep_zscore_sidak, None)]
        for rule, fn, m in rules:
            s = se.summarize(fn(h_recs, m))
            out[kind][f"honest_{rule}"] = s["best_f1"]
            out[kind][f"honest_{rule}_fmr5"] = s["best_f1_fmr<=5%"]
            f = s["best_f1_fmr<=5%"]
            log.info("         %-13s honest F1=%.3f (P=%.3f R=%.3f TP=%d FP=%d) | "
                     "fmr5 R=%s TP=%s", rule, s["best_f1"]["f1"], s["best_f1"]["precision"],
                     s["best_f1"]["recall"], s["best_f1"]["TP"], s["best_f1"]["FP"],
                     f'{f["recall"]:.3f}' if f else "n/a", f["TP"] if f else "n/a")

        # ---- recency gate: shrink the candidate set instead of rescoring it.
        #      90d is the pre-registered headline (2.6x the largest observed
        #      repeat gap); the rest is the frontier. Do not quote the best one.
        rec = {}
        for w in [7, 14, 30, 34, 60, 90, 180]:
            st = se.recency_stats(h_recs, w)
            s = se.summarize(se.sweep_margin_recency(h_recs, w))
            rec[str(w)] = {**st, "best_f1": s["best_f1"], "fmr5": s["best_f1_fmr<=5%"]}
            f = s["best_f1_fmr<=5%"]
            log.info("         recency %3dd  cands %5.1f/%5.1f  lost=%d  "
                     "F1=%.3f (P=%.3f R=%.3f TP=%d FP=%d TN=%d) | fmr5 R=%s TP=%s",
                     w, st["mean_candidates"], st["mean_journal_size"],
                     st["true_repeats_excluded"], s["best_f1"]["f1"],
                     s["best_f1"]["precision"], s["best_f1"]["recall"], s["best_f1"]["TP"],
                     s["best_f1"]["FP"], s["best_f1"]["TN"],
                     f'{f["recall"]:.3f}' if f else "n/a", f["TP"] if f else "n/a")
        assert rec["90"]["true_repeats_excluded"] == 0, rec["90"]
        out[kind]["recency"] = rec

    json.dump(out, open(OUT / "streaming_honest_compare.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "streaming_honest_compare.json")


if __name__ == "__main__":
    main()
