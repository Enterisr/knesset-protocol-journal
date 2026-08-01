# Remaining Work — Knesset Protocol Journal

Status 2026-07-30, mapped against the pipeline diagram (`Untitled-2026-03-25-1824.png`).
**Scoped to a few days.** What is cut is listed explicitly, with the reason.

Legend: ✅ done · 🟡 partial · ⬜ not started · ❌ cut

---

## 0. The research question

> **Why does near-ceiling intrinsic topic separability fail to produce usable streaming
> subject linking, and what actually closes the gap?**

On the gold benchmark, `e5 + ABTT-k1` scores **F1 0.619** (P 0.50, R 0.81, AUC 0.954) when
judging subject identity over *pairs*. **The same representation, making the same decision
online** against a growing journal, scores **F1 0.125** (P 0.077). Same metric, same data,
same embeddings — a 5× collapse caused purely by the decision setting. That gap between
separability and operational performance is the project's actual finding, and it is what the
three coupled tasks in `CLAUDE.md` are about.

**The answer, already measured across Steps 02–10:**

| lever | effect | evidence |
|---|---|---|
| **Entry representation** | recall 32% → **89%** | accumulated centroid vs frozen first span (Step 10) |
| **Decision rule** | F1 0.103 → **0.190** | distinctiveness margin vs flat cosine threshold (Step 10) |
| **De-anisotropization** | ARI 0.124 → **0.657** | ABTT k=1 on e5; removing *one* direction (Steps 03/07) |
| **Encoder choice** | ARI 0.265 – 0.657 | e5_k1 ≫ tfidf ≫ alephbert_k2 on gold (Step 07) — matters for *separability* |
| **LLM as classifier** | ≤ 0.111 | every verifier variant, single-snippet and full-timeline (Steps 08/10) |
| **LLM as extractor** | 0.190 → **0.211** | fingerprints fused with geometry (Step 10) |

**Why the gap exists:** only 36 of 523 segments are repeats (6.9%). Every arriving segment
is scored against a growing journal, so even a small false-positive rate, multiplied by
hundreds of candidates, floods the true matches. AUC 0.95 is nowhere near enough under
that base rate. **This is precisely why the brief's anti-duplication constraint is hard,
and it is the finding the write-up should be built around.**

**A second contribution, currently buried:** honest streaming evaluation is not optional.
Batch-fit whitening scores 0.250 and collapses to 0.094 when refit without look-ahead
(Step 10) — and two more leaks were found in the joint pipeline today (§3). Several
methods that look like improvements are partly measurement artifacts.

### Where canonicalization sits

Canonicalization is **Tomer's** research question (the diagram's red annotation), not this
project's subject — `CLAUDE.md`'s three tasks and four milestones never mention it. We
cross-evaluate it as **one ablation** in the row above: does *content-level* normalization
beat *geometric* normalization at removing the shared bureaucratic register?

The honest prior is that it will **not** raise F1. Step 11's only downstream measurement:

| variant | F1 | precision | recall |
|---|---|---|---|
| raw | 0.114 | 0.073 | 0.261 |
| canonical | **0.095** | **0.154** | 0.069 |

F1 got *worse*; precision doubled and recall collapsed. The intrinsic AUC gain
(0.947 → 0.986) is the circular one — the labels came from the same LLM pass that wrote
the text. So the right primary metric for canonicalization is **not F1** — it is
**recall at false-merge ≤ 5%**, because precision-up/recall-down is exactly the direction
the asymmetric cost rule wants. Step 11 reported best-F1 and called the result "mixed."
`steps/13_canonical_rq/` re-runs it on real gold with both surfaces.

---

## 1. Diagram → repo mapping

| # | Diagram box | Repo | Status |
|---|---|---|---|
| 1 | Protocols Pre-Processing | Step 01, Step 02 `extract_spans.py` | ✅ |
| 2 | Semantic Segmentation (Gemma-4-31B) | Tomer's pipeline (19 protocols); our side is **manual gold** (Step 06 → 211/213, Step 07 → 523 segments) | 🟡 |
| 2b | LLM-judge eval of segmentation | — | ❌ cut |
| 3 | **Canonicalize each segment (LLM)** | Colab bridge built (Step 12); `canon_input.json` regenerated 2026-07-30 | 🟡 **← the gate** |
| 4 | Per-Segment Embedding — DictaBERT | e5+ABTT-k1 settled (Steps 02–04, 10) | ❌ cut — not the bottleneck |
| 5 | Topic Modeling: cos-sim + threshold | Step 05 → 07 → 10 (centroid + margin, F1 0.190; +fingerprints 0.211) | ✅ |
| 5b | Evaluate w/ tagged protocols | Step 07 gold chains (36 repeat events) | ✅ |
| 5c | LLM-judge eval of linking | — | ❌ cut |
| 6 | 🟨 Journal: RAG per topic | — | ❌ cut (see §4) |
| 7 | 🟨 Unite segmentation per topic | — | ❌ cut (see §4) |
| 8 | 🟨 Topic Journal Entry re-Summarization | Step 09 does incremental append (dictalm2, faithfulness 80.7 / novelty 18.1) | 🟡 frozen as-is |
| 8b | Opus-as-judge for log writing | Step 09 judges with Qwen2.5-3B | ❌ cut |
| — | End-to-end joint run | Step 12 `run_journal.py` | 🟡 demonstration artifact only |

---

## 2. The finding that reshapes the plan

Across **every** config in `steps/10_temporal_linking/outputs/streaming_honest_compare.json`:

| operating point | precision | TP / 36 repeats |
|---|---|---|
| best-F1 | 0.09 – 0.13 | 11–19 |
| false-merge ≤ 5% | 0.08 – 0.11 | **2 – 3** |

**There is no operating point at which linking is good enough to produce a journal worth
reading.** That has been true since Step 07 (precision ~8%) and it is itself a result.

Two consequences, both now baked into the code and the writeup:

1. **Nothing may be measured through predicted links.** Step 09's choice to evaluate
   summarization on *gold* chains is forced, not a convenience. The end-to-end Step 12
   journal is a **demonstration artifact**; reporting an end-to-end quality number would
   conflate linking error with summarization error. Stated explicitly in
   `steps/13_canonical_rq/README.md` and in `sbatch/13_canonical_rq.sh`.
2. **θ=0.34 was never a bug** — it is Step 10's `abtt1 honest_fmr5` point (0.342), the
   deliberate anti-duplication setting. The empty journal (518 entries / 4 recurring) was
   caused by the *batch-fit leak* instead, and fixing that fixed the journal. Measured on
   the 523 gold segments under **predicted** growth (Step 10's θ values were tuned under
   *oracle* growth, so they had to be re-checked here):

   | θ | entries | recurring | largest | purity | |
   |---|---|---|---|---|---|
   | 0.156 | 329 | 76 | 15 | 0.658 | best-F1 |
   | **0.342** | 493 | **23** | 5 | **0.954** | false-merge ≤ 5% ← **default** |

   θ=0.342 now yields **23 recurring entries at 95% purity** (vs 4 under the leaky batch
   fit) — enough to exercise log writing, while honouring the asymmetric cost rule. Both
   points are run and reported.

---

## 3. Code changes made (2026-07-30)

**`steps/12_joint_pipeline/code/run_journal.py`**
- **Fixed a real leak.** It fit ABTT once over the whole corpus — a batch transform using
  future segments, exactly what Step 10 measured as leaky (whitening 0.250 → 0.094 refit
  honestly). `link()` now refits on the prefix seen so far at every step;
  `--batch_transform` keeps the old behaviour for comparison. Step 10's published θ values
  are only valid under the honest fit, so this also makes the θ choice coherent.
- **Fixed a second bug:** with exactly one existing entry the margin is undefined, and the
  code used a `-1.0` runner-up, making the margin `cos+1` — which clears any θ, so the
  first decision of every run merged unconditionally. Step 10's `sweep_margin` falls back
  to the raw cosine magnitude; now matched.
- Default θ kept at **0.342**, but now justified by the measurement above rather than
  transplanted from the oracle-growth sweep. Both operating points documented at the flag.
- `--abtt_k`; output JSON now records θ, k and which transform was used.

**`steps/12_joint_pipeline/code/build_segments.py`**
- Strips a leading subject header from canonical texts if Gemma emits one. Otherwise the
  canonical arm gets a clean topic title the raw arm never has, inflating topic separation
  for a reason unrelated to register — Step 11's circularity caveat sneaking back in
  through the *text* rather than the labels. Reports how many were stripped.

**`steps/13_canonical_rq/`** (new)
- `code/embed_canonical.py` — embeds the canonical variant, aligned row-for-row to Step 07's
  `ids.json` (alignment asserted, not assumed). Raw side reuses Step 07's `e5.npy` so the
  contrast can't be contaminated by encoder drift. Records how many segments fell back to raw.
- `code/rq_eval.py` — the head-to-head: intrinsic ROC-AUC (all pairs + recurring-only) and
  Step 10's streaming-honest linking at both operating points, raw vs canonical, reusing
  Step 10's harness directly (`streaming_eval`, `honest_centroid_records`) rather than a
  reimplementation, so numbers stay comparable to everything already published.
  Includes a **validity gate**: if >10% of segments fell back to raw text, the canonical
  arm is a blend and a null result is uninterpretable — the run warns loudly and records
  `comparison_valid: false`.
- `sbatch/13_canonical_rq.sh` — build → embed → eval → both demo journals.

**Regenerated** `steps/12_joint_pipeline/outputs/canon_input.json` (523 segments) and
`segments_raw.json`; both are gitignored intermediates that were absent locally.

---

## 3b. Data-integrity issue found while writing the report ⚠️

`steps/10_temporal_linking/code/make_plots.py:57` **hard-codes** its leaderboard values:

```python
f1s = [0.103, 0.111, 0.068, 0.190, 0.211]
```

The last value (0.211, "geometry + LLM fingerprints") **does not match**
`outputs/fingerprint_linking_*.json`, which reports **0.2407** (dictalm2 additive) and
**0.2013** (qwen7b additive). Step 10's README repeats the 0.211. Separately,
`fingerprint_linking.py:138` calls `se.transform(e5, "abtt1")` on the **full** matrix — a
batch fit — so all fingerprint numbers sit on the leaky path and are *not* comparable to the
honest 0.190 they are charted beside.

**RESOLVED 2026-08-01.**
- `make_plots.py` no longer hard-codes anything: every leaderboard bar is derived from the
  curves it already computes or read from a committed JSON (`verifier_eval_summary.json`,
  `timeline_verify_dictalm2.json`). The axis limit and baseline rule are derived too.
- Step 10's README now reports the Family D table **from the JSON** (geometry-only 0.2000 →
  additive 0.2407 / 0.2013, hard gate 0.2000 at 11.4% FMR≤5% recall) and states plainly
  that the whole family is **batch-fit** (`fingerprint_linking.py:138`) and therefore not
  comparable to the honest 0.190. The fingerprint rows were removed from the "final honest
  leaderboard", which is what created the discrepancy in the first place.
- The old 0.190/0.211/0.184 triple on a `center` base is in no committed artifact; the
  README now says so explicitly rather than silently dropping it.

**Third instance of the same error, found while writing the report:** the lever table
(baseline 0.103 / centroid / margin 0.190) mixed protocols too — the first two rows came
from `streaming_eval_results.json` (**batch** transform), only 0.190 was honest.
`steps/10_temporal_linking/code/honest_levers.py` recomputes all three on one transform,
one growth model and one fitting protocol → `outputs/honest_levers.json`:
first-span+threshold **0.086**, centroid+threshold **0.087**, centroid+margin **0.190**.

This **changes a claim**: under the matched protocol the threshold rule degenerates (both
threshold rows peak at recall 1.0, precision ≈0.045), so the centroid is not separable
there. Its recall fix (32%→89%) is real but is a *batch-fit abtt1* result and does not on
its own convert into F1. The report now says exactly that.

**Also resolved:** the end-to-end operating points (§2) had no artifact behind them either
— `journal_raw.json` on disk is the pre-fix leaky run. `steps/12_joint_pipeline/code/
verify_operating_points.py` now recomputes them on CPU from Step 07's committed embeddings
and writes `outputs/operating_points.json`: θ=0.156 → 343/80/0.688, θ=0.342 → 491/25/0.950.
These supersede the 329/76/0.658 and 493/23/0.954 quoted above (different embedding pass;
same conclusion).

---

## 4. What is cut, and why

- **`steps/13_journal_rag/`** (unite / RAG / re-summarize + 4-variant ablation + coverage
  metric + parallel Opus judge). This was the plan in the previous revision of this file.
  It is multi-week work; in a few days it half-lands and contaminates the writeup. Worse,
  per §2 it targets the *least broken* component — summarization already scores 80.7
  faithfulness, while linking precision is 0.13. **Cut entirely.**
  *(If any of it is revived later, `13.1 unite segments` is the cheapest and is the only
  piece that helps precision, the direction the asymmetric cost model wants.)*
- **Gemma segmentation over all 213 protocols** — not on the RQ's critical path.
- **Opus-as-judge harness** — not on the RQ's critical path.
- **DictaBERT** — encoder choice is settled; it is not the bottleneck.

---

## 5. Plan for the remaining days

The headline (§0) is answerable from results **already on disk**. Nothing on the critical
path depends on an external run. That is the point of the reordering below.

### ✅ SUBMISSION — `report/acl/main.pdf` (2026-08-01)

**This is the file to submit to Moodle.** 6 pages on the **course ACL template**
(`\documentclass[11pt]{article}` + `\usepackage[review]{ACL2023}`), which the earlier
`report/knesset_journal_report.tex` did **not** use — it was a hand-rolled twocolumn
layout with no citations, non-prescribed section names and inline floats.

`report/acl/` contains `main.tex`, `bib.bib`, `acl2023.sty`, `acl_natbib.bst`,
`fig_gap.png`. Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Compiles with **0 overfull boxes and 0 undefined references/citations**.

What it satisfies that the old one did not:
- prescribed sections **Introduction / Data / Methods / Results** (+ Conclusion,
  Limitations, Reproducibility);
- **9 real citations** — ABTT, multilingual-e5, AlephBERT, DictaLM 2.0, TAC update
  summarization, RAG, two cross-document event coreference papers, anisotropy — with
  `\bibliographystyle{acl_natbib}`;
- **all 6 tables + the figure on the dedicated float page** after `\clearpage`;
- `\And`-separated author block with emails.

⚠️ `[review]` prints line numbers in the margin. Drop it for a camera-ready version —
the template's own comment says so. Left on deliberately.

Every number in it was re-validated against `steps/*/outputs/*.json` on 2026-08-01;
the two that did not check out are recorded in §3b above and were fixed at the source.

*(Superseded: `report/knesset_journal_report.pdf`, the off-template 3-page version.)*

Also note Steps 01–10 each already have their own `report/*.pdf`; only 11/12/13 lack one.
(An earlier revision of this file wrongly claimed Step 10 had none.)

### Day 1 — write the results we already have

1. **Kick off the Colab canonicalization first thing** (it's free to start and runs
   unattended). `canon_input.json` (523 segments, 9.2 MB) is regenerated and ready.
   Upload → run `canonicalize_gold.ipynb` → drop `canonical_gold.json` into
   `steps/12_joint_pipeline/outputs/`. If it doesn't land, we lose **one ablation**, not
   the project.
2. **Write the Step 10 report** — it holds the headline: entry representation, margin
   gate, LLM-as-extractor, and the batch-vs-honest leak.
3. **Write the Step 11 and 12 reports.** Steps 01–09 all have one; these three don't.

### Day 2 — the analysis that ties it together

4. Assemble §0's table into the **top-level report**: AUC 0.954 → F1 0.190, the 6.9%
   base-rate explanation, and the honest-evaluation contribution. This is new *analysis*
   over existing numbers, not new experiments.
5. If canonical landed: `sbatch sbatch/13_canonical_rq.sh` → `rq_results.json`, `fig_rq.png`.
   Read the answer off **FMR≤5% recall first**, F1 second. Check the three validity gates
   in the Step 13 README before believing it.

### Day 3 — finish

6. Step 13 report (or a short "not run / Step 11 numbers stand" note if the Colab pass
   failed), then final pass over the top-level report.

### Critical path

```
results already on disk ──► Step 10/11/12 reports ──► top-level report ──► DONE
                                                             ▲
canon_input.json ✅ ──► Colab ──► rq_eval ──► one extra row ──┘   (bonus; failure is survivable)
```
