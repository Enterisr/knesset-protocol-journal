"""Recompute the two journal operating points from committed artifacts (CPU only).

WHY THIS EXISTS
    run_journal.py needs a GPU to embed, so the entries/recurring/purity figures the
    write-up quotes for theta=0.156 and theta=0.342 came from a run whose output was
    never committed (`outputs/journal_raw.json` on disk is the OLDER, batch-fit run:
    518 entries / 4 recurring). Nothing on disk backed the quoted numbers.

    The linking maths, however, is a pure function of the embedding matrix. So this
    calls run_journal.link() -- the real implementation, not a reimplementation --
    against Step 07's committed e5.npy, and writes the result as an artifact.

CAVEAT, and it is why the figures differ slightly from the ones first quoted:
    Step 07's e5.npy and run_journal.py's own embedding pass are two different
    encodings of the same segments (different text preparation), so this reproduces
    the operating points to within a couple of entries rather than exactly. Quote
    THESE numbers -- they are the ones with an artifact behind them.

Usage (CPU, ~2 min):  python steps/12_joint_pipeline/code/verify_operating_points.py
"""
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from run_journal import link

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
STEP07 = STEP_DIR.parent / "07_gold_eval" / "outputs"
OUT = STEP_DIR / "outputs"

# Step 10's two defensible streaming-honest operating points on abtt1.
THETAS = {"best_f1": 0.156, "fmr5": 0.342}


def purity(entries, topics):
    """Standard clustering purity: each entry votes with its majority gold topic."""
    hit = sum(Counter(topics[i] for i in e).most_common(1)[0][1] for e in entries)
    return hit / sum(len(e) for e in entries)


def main():
    ids = json.load(open(STEP07 / "embeddings" / "ids.json", encoding="utf-8"))
    emb = np.load(STEP07 / "embeddings" / "e5.npy")
    assert len(ids) == len(emb), f"{len(ids)} ids vs {len(emb)} vectors"

    rows = [{"seg_key": f"{r['doc_id']}__{r['seg_idx']}", "date": r["date"]} for r in ids]
    topics = [r["topic"] for r in ids]

    out = {"source": "steps/07_gold_eval/outputs/embeddings/e5.npy",
           "n_segments": len(rows), "transform": "abtt1", "fit": "honest (prefix refit)",
           "operating_points": {}}

    for name, theta in THETAS.items():
        entries = link(emb, rows, theta, k=1, honest=True)
        multi = [e for e in entries if len(e) > 1]
        rec = {"theta": theta,
               "n_entries": len(entries),
               "n_recurring": len(multi),
               "largest_entry": max((len(e) for e in entries), default=0),
               "purity": round(purity(entries, topics), 4),
               "purity_recurring_only": (round(purity(multi, topics), 4) if multi else None)}
        out["operating_points"][name] = rec
        log.info("theta=%.3f -> %d entries, %d recurring, largest %d, purity %.4f",
                 theta, rec["n_entries"], rec["n_recurring"],
                 rec["largest_entry"], rec["purity"])

    p = OUT / "operating_points.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", p)


if __name__ == "__main__":
    main()
