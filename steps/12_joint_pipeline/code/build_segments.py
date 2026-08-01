"""
Step 12 — build the {seg_key, date, text} segment files the joint pipeline consumes.

  segments_raw.json        always: text = raw span_text (from export_for_canon.py)
  segments_canonical.json  if outputs/canonical_gold.json exists (the Colab output):
                           text = Tomer-style canonical rewrite of each segment,
                           aligned by seg_key. Segments missing a canonical fall back
                           to raw so the two runs cover the same set.

Usage (CPU): python steps/12_joint_pipeline/code/build_segments.py
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"

# Gemma's canonicalization prompt also emits a `subject`. If the rewritten text
# repeats it as a header line, the canonical embeddings get a clean topic title that
# the raw ones never have -- inflating any topic-separation metric for a reason that
# has nothing to do with register normalization. Drop such a header before embedding.
SUBJECT_PREFIXES = ("נושא:", "נושא -", "כותרת:", "subject:", "topic:")


def strip_subject_header(text, subject=""):
    lines = text.strip().split("\n")
    if not lines:
        return text
    head = lines[0].strip()
    low = head.lower()
    is_header = any(low.startswith(p) for p in SUBJECT_PREFIXES)
    if not is_header and subject:
        s = subject.strip().rstrip(".")
        is_header = bool(s) and head.rstrip(".") == s
    return "\n".join(lines[1:]).strip() if is_header and len(lines) > 1 else text


def main():
    src = json.load(open(OUT / "canon_input.json", encoding="utf-8"))
    raw = [{"seg_key": r["seg_key"], "date": r["date"], "text": r["raw"]} for r in src]
    json.dump(raw, open(OUT / "segments_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("Wrote segments_raw.json (%d segments)", len(raw))

    canon_file = OUT / "canonical_gold.json"
    if canon_file.exists():
        canon = json.load(open(canon_file, encoding="utf-8"))
        # canonical_gold.json: list of {seg_key, subject, canonical, ...}
        cmap = {c["seg_key"]: c for c in canon}
        n_c = n_stripped = 0
        segs = []
        for r in src:
            c = cmap.get(r["seg_key"])
            if c and c.get("canonical", "").strip():
                text = strip_subject_header(c["canonical"], c.get("subject", ""))
                n_stripped += text != c["canonical"]
                n_c += 1
            else:
                text = r["raw"]
            segs.append({"seg_key": r["seg_key"], "date": r["date"], "text": text})
        if n_stripped:
            log.warning("Stripped a leading subject header from %d/%d canonical texts "
                        "(it would hand the canonical arm a clean topic title the raw "
                        "arm never gets -- Step 11's circularity caveat via the text).",
                        n_stripped, n_c)
        json.dump(segs, open(OUT / "segments_canonical.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        log.info("Wrote segments_canonical.json (%d canonical / %d fell back to raw)",
                 n_c, len(segs) - n_c)
    else:
        log.info("No canonical_gold.json yet -- run the Colab notebook, drop its output at %s", canon_file)


if __name__ == "__main__":
    main()
