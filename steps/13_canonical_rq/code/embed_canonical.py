"""
Step 13 — embed the CANONICAL rewrites of our 523 gold segments with the same
encoder used everywhere else (multilingual-e5-large), aligned row-for-row with
Step 07's `embeddings/ids.json`.

The RAW side is NOT re-embedded: Step 07's `embeddings/e5.npy` already holds the
e5 vectors of exactly these segments in exactly this order, so reusing it keeps
the raw-vs-canonical comparison free of any encoder/version drift. This script
only produces the missing half.

Alignment is asserted, not assumed: canonical texts are looked up by seg_key
("{doc_id}__{seg_idx}") and emitted in ids.json order. A segment with no
canonical rewrite falls back to its raw text (same rule as build_segments.py),
and the count of fallbacks is reported — if that number is large the comparison
is diluted toward raw and the result must be read accordingly.

Input:  steps/12_joint_pipeline/outputs/segments_canonical.json (Colab output,
        via build_segments.py)
Output: outputs/embeddings/e5_canonical.npy, outputs/embeddings/meta.json

Usage (GPU): python steps/13_canonical_rq/code/embed_canonical.py
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)
STEP07 = STEP_DIR.parent / "07_gold_eval" / "outputs"
STEP12 = STEP_DIR.parent / "12_joint_pipeline" / "outputs"

E5 = "intfloat/multilingual-e5-large"


def seg_key(doc_id, seg_idx):
    return f"{doc_id}__{seg_idx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=str(STEP12 / "segments_canonical.json"))
    ap.add_argument("--out_name", default="e5_canonical.npy")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    ids = json.load(open(STEP07 / "embeddings" / "ids.json", encoding="utf-8"))
    raw_rows = json.load(open(STEP12 / "segments_raw.json", encoding="utf-8"))
    raw_by_key = {r["seg_key"]: r["text"] for r in raw_rows}

    seg_path = Path(args.segments)
    if not seg_path.exists():
        raise SystemExit(
            f"{seg_path} not found.\n"
            "Run the Colab canonicalizer, drop canonical_gold.json into "
            "steps/12_joint_pipeline/outputs/, then run build_segments.py."
        )
    canon_by_key = {r["seg_key"]: r["text"] for r in json.load(open(seg_path, encoding="utf-8"))}

    texts, n_fallback, missing = [], 0, []
    for r in ids:
        k = seg_key(r["doc_id"], r["seg_idx"])
        if k not in raw_by_key:
            missing.append(k)
            continue
        t = canon_by_key.get(k, "").strip()
        if not t or t == raw_by_key[k]:
            t = raw_by_key[k]
            n_fallback += 1
        texts.append(t)

    if missing:
        raise SystemExit(f"{len(missing)} ids.json rows have no segment text "
                         f"(first: {missing[:3]}); alignment is broken, aborting.")
    assert len(texts) == len(ids), (len(texts), len(ids))
    log.info("Aligned %d segments (%d fell back to raw = %.1f%%)",
             len(texts), n_fallback, 100 * n_fallback / len(texts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(E5, device=device)
    enc.max_seq_length = 512
    emb = enc.encode([f"query: {t}" for t in texts], batch_size=args.batch_size,
                     convert_to_numpy=True, show_progress_bar=False)

    np.save(OUT / args.out_name, emb)
    json.dump({"n": len(texts), "n_fallback_to_raw": n_fallback,
               "encoder": E5, "source": str(seg_path)},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s %s", OUT / args.out_name, emb.shape)


if __name__ == "__main__":
    main()
