"""
Step 13 — canonicalization ablation on real gold: content-level vs geometric
normalization of the shared bureaucratic register.

    Does LLM canonicalization remove the register better than ABTT / whitening
    (Steps 03/04/10), and does that convert into ANTI-DUPLICATION performance?

This is ONE ablation, not the project's research question -- canonicalization is
Tomer's question, and CLAUDE.md's three tasks never mention it. See REMAINING_WORK.md
§0 for the project's own RQ (why AUC 0.954 separability yields only F1 0.190 linking).

The primary metric here is recall at false-merge <= 5%, NOT F1: Step 11 found
canonicalization doubles precision (0.073 -> 0.154) while collapsing recall, which
lowers F1 but is the direction the asymmetric cost rule wants. F1 is the wrong
instrument for that trade; Step 11 used it and read the result as merely "mixed".

Step 11 asked this on Tomer's 19 protocols and had to flag two caveats: the
pseudo-labels came from the same LLM pass as the canonical text (circularity),
and 29 exact-subject repeats is too thin to move F1. This script removes both:
labels are our MANUAL gold chains (independent of the canonicalizer) and the
benchmark is the full 523 segments / 22 recurring topics / 36 repeat events.

Two measurement surfaces, both raw-vs-canonical, everything else held fixed
(same encoder, same transforms, same linking rule, same gold):

  1. INTRINSIC  — same/different-matter separability (ROC-AUC over segment pairs).
                  Reported on all pairs and on the recurring-only subset, which is
                  the harder discrimination and mirrors Step 07's setup.
  2. DOWNSTREAM — Step 10's streaming-honest linking (running-fit transform +
                  accumulated centroid + margin gate), at BOTH operating points:
                  best-F1, and the anti-duplication point (false-merge rate <= 5%),
                  since the project's cost model is asymmetric.

Usage (CPU, ~2 min): python steps/13_canonical_rq/code/rq_eval.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP_DIR = Path(__file__).resolve().parents[1]
OUT = STEP_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
STEP07 = STEP_DIR.parent / "07_gold_eval" / "outputs"
STEP10_CODE = STEP_DIR.parent / "10_temporal_linking" / "code"

sys.path.insert(0, str(STEP10_CODE))
import streaming_eval as se                       # noqa: E402
from streaming_honest_compare import honest_centroid_records  # noqa: E402

TRANSFORMS = ["center", "abtt1", "abtt2"]


def intrinsic(X, ids):
    """Same/different-matter ROC-AUC over segment pairs, on L2-normalized vectors."""
    from sklearn.preprocessing import normalize
    Xn = normalize(X.astype(np.float64))
    topics = np.array([r["topic"] for r in ids])
    S = Xn @ Xn.T
    iu = np.triu_indices(len(ids), k=1)
    sims, same = S[iu], (topics[iu[0]] == topics[iu[1]]).astype(int)

    counts = {}
    for t in topics:
        counts[t] = counts.get(t, 0) + 1
    rec = np.array([counts[t] > 1 for t in topics])
    ridx = np.where(rec)[0]
    Sr = Xn[ridx] @ Xn[ridx].T
    tr = topics[ridx]
    ru = np.triu_indices(len(ridx), k=1)
    sims_r, same_r = Sr[ru], (tr[ru[0]] == tr[ru[1]]).astype(int)

    return {
        "auc_all_pairs": float(roc_auc_score(same, sims)),
        "n_pairs": int(len(same)), "n_positive": int(same.sum()),
        "auc_recurring_only": float(roc_auc_score(same_r, sims_r)),
        "n_recurring_segments": int(len(ridx)),
        "n_pairs_recurring": int(len(same_r)), "n_positive_recurring": int(same_r.sum()),
        "mean_cos_same": float(sims[same == 1].mean()),
        "mean_cos_diff": float(sims[same == 0].mean()),
    }


def downstream(X, ids):
    """Step-10 streaming-honest linking (no look-ahead), per transform."""
    res = {}
    for kind in TRANSFORMS:
        s = se.summarize(se.sweep_margin(honest_centroid_records(X, ids, kind)))
        res[kind] = {"best_f1": s["best_f1"], "fmr5": s["best_f1_fmr<=5%"]}
        log.info("  %-7s best_f1=%.3f (P=%.3f R=%.3f) | fmr<=5%% R=%s",
                 kind, s["best_f1"]["f1"], s["best_f1"]["precision"], s["best_f1"]["recall"],
                 f'{s["best_f1_fmr<=5%"]["recall"]:.3f}' if s["best_f1_fmr<=5%"] else "n/a")
    return res


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    w, xs = 0.35, np.arange(2)

    ax = axes[0]
    for j, v in enumerate(variants):
        m = results[v]["intrinsic"]
        ax.bar(xs + j * w, [m["auc_all_pairs"], m["auc_recurring_only"]], w, label=v)
    ax.set_xticks(xs + w / 2); ax.set_xticklabels(["all pairs", "recurring only"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("ROC-AUC"); ax.set_title("Intrinsic separability")
    ax.legend()

    for ax, key, title in ((axes[1], "best_f1", "Downstream linking — best F1"),
                           (axes[2], "fmr5", "Anti-duplication (FMR<=5%) — recall")):
        xs2 = np.arange(len(TRANSFORMS))
        for j, v in enumerate(variants):
            vals = []
            for k in TRANSFORMS:
                d = results[v]["downstream"][k][key]
                vals.append((d["f1"] if key == "best_f1" else d["recall"]) if d else 0.0)
            ax.bar(xs2 + j * w, vals, w, label=v)
        ax.set_xticks(xs2 + w / 2); ax.set_xticklabels(TRANSFORMS)
        ax.set_title(title); ax.set_ylabel("F1" if key == "best_f1" else "recall")
        ax.legend()

    fig.suptitle("Does canonicalization fix what geometry could not? (523 gold segments, 36 repeats)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info("Wrote %s", path)


def main():
    ids = json.load(open(STEP07 / "embeddings" / "ids.json", encoding="utf-8"))
    sources = {"raw": STEP07 / "embeddings" / "e5.npy",
               "canonical": STEP_DIR / "outputs" / "embeddings" / "e5_canonical.npy"}

    results = {}
    for variant, p in sources.items():
        if not p.exists():
            log.warning("SKIP %s — %s not found (run embed_canonical.py)", variant, p)
            continue
        X = np.load(p)
        assert len(X) == len(ids), f"{variant}: {len(X)} vectors vs {len(ids)} ids"
        log.info("=== %s (%s)", variant, X.shape)
        results[variant] = {"intrinsic": intrinsic(X, ids), "downstream": downstream(X, ids)}
        log.info("  intrinsic AUC: all=%.3f recurring=%.3f",
                 results[variant]["intrinsic"]["auc_all_pairs"],
                 results[variant]["intrinsic"]["auc_recurring_only"])

    # ── validity gate ────────────────────────────────────────────────────────
    # Segments with no usable canonical rewrite fall back to RAW text, which makes
    # the "canonical" arm a raw/canonical blend and biases the comparison toward
    # "no effect". A null result is only interpretable if this fraction is small.
    meta_p = STEP_DIR / "outputs" / "embeddings" / "meta.json"
    meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else None
    frac_fb = (meta["n_fallback_to_raw"] / meta["n"]) if meta and meta.get("n") else None
    if frac_fb is not None and frac_fb > 0.10:
        log.warning("=" * 72)
        log.warning("VALIDITY WARNING: %.1f%% of segments fell back to RAW text.",
                    100 * frac_fb)
        log.warning("The 'canonical' arm is a blend; a null result here does NOT mean")
        log.warning("canonicalization is ineffective. Fix the Colab pass before reporting.")
        log.warning("=" * 72)

    out = {"n_segments": len(ids),
           "canonical_meta": meta,
           "frac_fallback_to_raw": frac_fb,
           "comparison_valid": (frac_fb is not None and frac_fb <= 0.10),
           "results": results}
    json.dump(out, open(OUT / "rq_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    log.info("Wrote %s", OUT / "rq_results.json")

    if len(results) == 2:
        plot(results, OUT / "fig_rq.png")
        r, c = results["raw"], results["canonical"]
        log.info("")
        log.info("ANSWER  intrinsic AUC (recurring): raw %.3f -> canonical %.3f  (%+.3f)",
                 r["intrinsic"]["auc_recurring_only"], c["intrinsic"]["auc_recurring_only"],
                 c["intrinsic"]["auc_recurring_only"] - r["intrinsic"]["auc_recurring_only"])
        for k in TRANSFORMS:
            log.info("        %-7s best F1: raw %.3f -> canonical %.3f  (%+.3f)", k,
                     r["downstream"][k]["best_f1"]["f1"], c["downstream"][k]["best_f1"]["f1"],
                     c["downstream"][k]["best_f1"]["f1"] - r["downstream"][k]["best_f1"]["f1"])


if __name__ == "__main__":
    main()
