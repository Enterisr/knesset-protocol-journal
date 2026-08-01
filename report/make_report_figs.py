"""
Figures for the top-level report. Every value is READ FROM a committed JSON artifact
under steps/*/outputs/ -- nothing is hard-coded. (steps/10/code/make_plots.py hard-codes
its leaderboard values and one of them, 0.211, does not match
fingerprint_linking_*.json; that figure is deliberately not reused here.)

Usage: python report/make_report_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
BLUE, GREY, RED = "#1f77b4", "#7f8896", "#c0392b"


def pr_points(curve):
    """Monotone precision-recall frontier from a theta sweep."""
    pts = sorted(((c["recall"], c["precision"]) for c in curve), key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    g = json.load(open(ROOT / "steps/07_gold_eval/outputs/gold_reeval_results.json",
                       encoding="utf-8"))["e5_k1"]
    pw, st = g["pairwise"], g["streaming"]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for d, lab, col in ((pw, "Pairwise (oracle over all pairs)", BLUE),
                        (st, "Streaming (online, growing journal)", RED)):
        xs, ys = pr_points(d["curve"])
        ax.plot(xs, ys, "-", lw=1.8, color=col, label=lab)
    for d, col in ((pw, BLUE), (st, RED)):
        b = d["best_f1"]
        ax.plot([b["recall"]], [b["precision"]], "o", ms=7, color=col, zorder=5)
        ax.annotate(f'best $F_1$={b["f1"]:.3f}', (b["recall"], b["precision"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=col)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Same representation (e5 + ABTT-$k$1), same decision, two protocols\n"
                 "Separability is not the bottleneck; the streaming decision is",
                 fontsize=10)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    p = OUT / "fig_gap.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print("wrote", p)
    print(f"  pairwise  best_f1={pw['best_f1']['f1']}  auc={pw['auc']}")
    print(f"  streaming best_f1={st['best_f1']['f1']}")


if __name__ == "__main__":
    main()
