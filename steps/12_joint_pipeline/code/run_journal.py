"""
Step 12 — the JOINT PIPELINE: segments -> embed -> streaming link (Step 10) ->
context-conditioned log writing (Step 9) -> a finished per-topic journal.

This unifies the two halves of the project. It consumes a segments file that can
be EITHER our raw gold segments OR their canonicalized rewrites (Tomer's upstream),
selected with --segments, so the same command produces the journal both ways and
the raw-vs-canonical effect is measured end-to-end, not just on an intrinsic metric.

Stages (all in one job):
  1. EMBED every segment's text with multilingual-e5-large (GPU).
  2. LINK online with the Step-10 method (ABTT-k1 + accumulated centroid + margin
     gate at a fixed operating point): each segment joins the best existing journal
     entry if its distinctiveness margin clears theta, else opens a new entry.
     The ABTT transform is refit at every step on the prefix seen so far, so the
     linking is streaming-honest (no look-ahead). --batch_transform reverts to the
     leaky whole-corpus fit for comparison.
  3. LOG-WRITE per entry with the Step-9 method (dictalm2): an opening summary for
     the entry's first segment, then an incremental update per later segment,
     conditioned on the running journal (GPU).
  4. WRITE journal.json.

Input segments file: list of {seg_key, date, text} (see export_for_canon.py /
build_raw_segments.py). Output: outputs/journal_<tag>.json.

Usage (GPU, one job):
    python steps/12_joint_pipeline/code/run_journal.py \
        --segments outputs/segments_raw.json --tag raw_fmr5 \
        --theta 0.342 --logwrite_model dicta-il/dictalm2.0-instruct
    python steps/12_joint_pipeline/code/run_journal.py \
        --segments outputs/segments_raw.json --tag raw_f1 --theta 0.156
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

E5 = "intfloat/multilingual-e5-large"
SNIPPET_WORDS = 700

OPENING_PROMPT = """You are opening a journal entry for a legislative matter from an Israeli \
Knesset Finance Committee session. Write a concise opening entry (2-4 sentences, Hebrew) of \
what this matter is about, based ONLY on the text. Respond with ONLY the entry text.

SESSION TEXT (date {date}):
<segment>
{text}
</segment>

JOURNAL ENTRY:"""

UPDATE_PROMPT = """You maintain a chronological journal entry for a legislative matter across \
Knesset Finance Committee sessions. Below is the EXISTING JOURNAL and a NEW session on the same \
matter. Write ONLY the new incremental update (1-3 sentences, Hebrew): new developments, \
decisions, status changes. Do NOT repeat what's already logged; do NOT invent. Respond with \
ONLY the update text.

EXISTING JOURNAL:
{journal}

NEW SESSION TEXT (date {date}):
<segment>
{text}
</segment>

NEW INCREMENTAL UPDATE:"""


def abtt(X, k=1):
    mu = X.mean(0, keepdims=True); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return normalize(Xc - (Xc @ Vt[:k].T) @ Vt[:k])


def snippet(t, n=SNIPPET_WORDS):
    return " ".join(t.split()[:n])


def link(emb, rows, theta, k=1, honest=True):
    """Step-10 online linking: ABTT-k + accumulated centroid + margin gate.

    honest=True refits the ABTT transform at every step on ONLY the segments seen
    so far, which is what the streaming setting actually permits. Fitting it once
    over the whole corpus (honest=False) leaks future segments into the transform
    -- Step 10 measured that leak directly (batch 0.250 -> honest 0.094 for
    whitening) and its reported operating points are the honest ones, so the
    thetas below are only valid under honest=True.

    Returns a list of entries, each a list of row indices in chronological order.
    """
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["date"], rows[i]["seg_key"]))
    entries = []          # entry idx -> list of positions p (into `order`)
    Xt_full = None if honest else abtt(emb, k)

    for p, i in enumerate(order):
        if honest:
            # refit on the prefix; row p of Xt is the current segment
            Xt = abtt(emb[order[: p + 1]], k)
            x = Xt[p]
            members_of = lambda mem: Xt[mem]                       # noqa: E731
        else:
            Xt = Xt_full
            x = Xt[i]
            members_of = lambda mem: Xt[[order[q] for q in mem]]   # noqa: E731

        if entries:
            cos = np.empty(len(entries))
            for e, mem in enumerate(entries):
                v = members_of(mem).mean(0)
                cos[e] = (v / (np.linalg.norm(v) + 1e-12)) @ x
            best = int(np.argmax(cos))
            if len(cos) >= 2:
                margin = cos[best] - np.partition(cos, -2)[-2]
            else:
                # exactly one entry: the margin is undefined. Step 10's sweep_margin
                # falls back to the raw cosine magnitude here; match it, or the very
                # first decision of every run merges unconditionally (a -1.0 runner-up
                # makes the margin cos+1, which clears any theta).
                margin = cos[best]
            if margin >= theta:
                entries[best].append(p)
                continue
        entries.append([p])

    return [[order[p] for p in mem] for mem in entries]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--tag", required=True)
    # Step 10 (abtt1, streaming-honest) gives two defensible operating points. Those
    # thetas were tuned under ORACLE growth (entries keyed by gold topic), so what they
    # do under PREDICTED growth had to be measured separately. On the 523 gold segments:
    #
    #   theta   entries  recurring  largest  purity
    #   0.156      329       76       15      0.658   best-F1
    #   0.342      493       23        5      0.954   false-merge <= 5%   <- default
    #
    # 0.342 is the default: it satisfies the project's asymmetric cost rule
    # (false-merge >> false-split) at 95% purity, and -- once the transform is fit
    # honestly rather than in batch -- it still yields 23 recurring entries, enough to
    # exercise the log-writing half. (The old batch-fit code gave only 4 at this theta.)
    # Run both and report both.
    ap.add_argument("--theta", type=float, default=0.342)
    ap.add_argument("--abtt_k", type=int, default=1)
    ap.add_argument("--batch_transform", action="store_true",
                    help="fit ABTT once over the whole corpus (LEAKS future segments; "
                         "for comparison against the honest default only)")
    ap.add_argument("--logwrite_model", default="dicta-il/dictalm2.0-instruct")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--no_logwrite", action="store_true", help="link only, skip log writing")
    args = ap.parse_args()

    rows = json.load(open(args.segments if Path(args.segments).is_absolute()
                          else STEP_DIR / args.segments, encoding="utf-8"))
    log.info("Loaded %d segments (%s)", len(rows), args.tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. EMBED
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(E5, device=device); enc.max_seq_length = 512
    emb = enc.encode([f"query: {r['text']}" for r in rows], batch_size=32,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float64)
    del enc; torch.cuda.empty_cache()
    emb = emb.astype(np.float64)

    # 2. LINK
    entries = link(emb, rows, args.theta, k=args.abtt_k, honest=not args.batch_transform)
    n_multi = sum(len(e) >= 2 for e in entries)
    log.info("Linked into %d journal entries (%d recurring, %d singletons) "
             "@theta=%.3f abtt_k=%d transform=%s",
             len(entries), n_multi, len(entries) - n_multi, args.theta, args.abtt_k,
             "batch(LEAKY)" if args.batch_transform else "streaming-honest")

    # 3. LOG-WRITE
    journal = []
    if args.no_logwrite:
        for eid, mem in enumerate(entries):
            journal.append({"entry_id": eid,
                            "segments": [rows[i]["seg_key"] for i in mem],
                            "dates": [rows[i]["date"] for i in mem]})
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.logwrite_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.logwrite_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device).eval()

        def gen(prompt):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False, add_generation_prompt=True)
            e = tok(text, return_tensors="pt", truncation=True, max_length=6000).to(device)
            with torch.no_grad():
                o = model.generate(**e, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   repetition_penalty=1.15, pad_token_id=tok.pad_token_id)
            r = tok.batch_decode(o[:, e["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            for stop in ["\nSESSION", "\nNEW SESSION", "\nEXISTING JOURNAL", "\n<segment>"]:
                j = r.find(stop)
                if j != -1:
                    r = r[:j]
            return r.strip()

        for eid, mem in enumerate(entries):
            log_entries = []
            for pos, i in enumerate(mem):
                r = rows[i]
                if pos == 0:
                    txt = gen(OPENING_PROMPT.format(date=r["date"], text=snippet(r["text"])))
                else:
                    js = "\n".join(f"[{le['date']}] {le['text']}" for le in log_entries)
                    txt = gen(UPDATE_PROMPT.format(journal=js, date=r["date"], text=snippet(r["text"])))
                log_entries.append({"date": r["date"], "seg_key": r["seg_key"], "text": txt})
            journal.append({"entry_id": eid,
                            "segments": [rows[i]["seg_key"] for i in mem],
                            "log": log_entries})
            if (eid + 1) % 50 == 0:
                log.info("  logged %d/%d entries", eid + 1, len(entries))

    out = OUT / f"journal_{args.tag}.json"
    json.dump({"tag": args.tag, "theta": args.theta, "abtt_k": args.abtt_k,
               "transform": "batch" if args.batch_transform else "streaming_honest",
               "n_segments": len(rows),
               "n_entries": len(entries), "n_recurring": n_multi, "journal": journal},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
