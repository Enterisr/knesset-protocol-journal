# Handoff — Knesset Protocol Journal (2026-08-01)

Session ended on credits. This records **what was done**, **what is submittable right now**,
and **what is left**. Nothing here is speculative — every claim below was checked against a
committed artifact during the session.

---

## TL;DR

**Submit `report/acl/main.pdf`.** It is finished: 6 pages total — **3 pages of body**, then
a page of tables + figure, a page of worked examples, and references. Compiles clean
(0 overfull boxes, 0 undefined references/citations).

**Nothing is committed.** All work is in the working tree on `master`. That is the single
outstanding action if you want it preserved.

---

## 1. The submission

```
report/acl/
├── main.tex        ← the report
├── bib.bib         ← 9 citations
├── acl2023.sty     ← course template (copied from template/)
├── acl_natbib.bst
├── fig_gap.png
└── main.pdf        ← THE FILE TO SUBMIT
```

Rebuild:
```bash
cd report/acl
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Page layout (as requested)

| page | content |
|---|---|
| 1–3 | body: Introduction, Data, Methods, Results, Conclusion, Limitations |
| 4 | Tables 1–6 + Figure 1 (precision–recall) |
| 5 | Table 7 — two worked examples, full width |
| 6 | References |

### Template compliance

- `\documentclass[11pt]{article}` + `\usepackage{ACL2023}` — the course template.
- `[review]` is **deliberately off** (it prints margin line numbers; the template's own
  comment says remove it for the final version). Re-add it if the course wants line numbers.
- Prescribed section names: **Introduction / Data / Methods / Results**.
- All floats after `\clearpage` on dedicated pages, as the template requires.
- `\And`-separated author block with emails.
- 9 real citations via `acl_natbib`: ABTT (Mu & Viswanath 2018), multilingual-e5,
  AlephBERT, DictaLM 2.0, TAC update summarization, RAG, two cross-document event
  coreference papers, anisotropy. **Four were verified against the published record**, not
  written from memory.

### The examples page (Table 7)

Two real cases pulled from committed outputs, Hebrew translated by hand:

- **(a) Why linking fails** — gold repeat event 8. A query and its true continuation (same
  ministry, same request no. 177) score cos **0.266**, while an unrelated Transport Ministry
  transfer scores **0.448**. Source: `steps/08_retrieve_verify/outputs/miss_inspection.txt`.
- **(b) Why log writing fails** — the "rising milk prices" chain. Entry 2 restates entry 1
  almost point for point before adding one clause. Source:
  `steps/09_log_writing/outputs/generated_logs_dictalm2.json`.

---

## 2. Validation pass — three claims did not survive

Every headline number was re-derived from `steps/*/outputs/*.json`. Most verified exactly:
523 segments, 36 repeat events, ARI 0.124→0.657, AUC 0.954, F1 0.619 vs 0.125,
recall@10 = 1.000, whitening 0.250→0.094, and all twelve log-writing scores.

These three did not, and **each was fixed at the source, not in the prose**:

| # | claim | artifact says | fixed by |
|---|---|---|---|
| 1 | fingerprint fusion F1 **0.211** | **0.2407** — and batch-fit, so never comparable to 0.190 | `steps/10/code/make_plots.py` now derives every bar; Step 10 README rewritten |
| 2 | journal θ=0.342 → **493/23/0.954** | **491/25/0.950** | new `steps/12/code/verify_operating_points.py` → `operating_points.json` |
| 3 | lever table 0.103 → 0.190 | rows came from **different fitting protocols** | new `steps/10/code/honest_levers.py` → `honest_levers.json` |

### #3 changed a claim — read this if you present the work

The old lever table mixed protocols: the baseline and centroid rows came from
`streaming_eval_results.json` (**batch** transform), while only 0.190 was streaming-honest.
Recomputed on one transform, one growth model, one fitting protocol:

| method (matched: center · honest refit · oracle growth) | F1 | R@FMR≤5% |
|---|---|---|
| first-span + threshold | 0.086 | 0% |
| centroid + threshold | 0.087 | 0% |
| **centroid + margin** | **0.190** | **8.6%** |

Under the matched protocol the threshold rule **degenerates** — both threshold rows peak at
recall 1.0 with precision ≈0.045 — so the centroid's benefit is *not separable there*. Its
32%→89% recall gain is real but is a **batch-fit ABTT-k1** result and does not on its own
convert into F1. **The margin gate is what carries the result.** The report says exactly this.

This was the same protocol-mixing error the repo warns about, reproduced inside the report.

---

## 3. Canonicalisation — resolved, and the answer is "no"

The Colab rewriting **ran and succeeded**: `handoff_canon_batch/canonical_part_{A,B}.json`,
173 records, one model (`gemma-4-31B-it`), none empty, 22 with `json_ok: false` (text kept).
`merge_canonical.py` writes `steps/12_joint_pipeline/outputs/canonical_gold.json` cleanly.

**But it covers the wrong third of the corpus:**

| | |
|---|---|
| segments rewritten | 173 / 523 (33.1%) |
| date range | 2023-05 → 2023-10 (corpus starts 2022-11) |
| **repeat events inside** | **1 of 36** |
| recurring matters with zero coverage | 21 of 22 |

The earlier README claim that the other 350 "were canonicalized somewhere else" was **never
true** — they are not in the repo, in Downloads/Documents/OneDrive, in the `knesset_canon`
Drive folder, or in Tomer's shared folder (which holds only his 19-protocol files at a
different segment granularity). That batch was never produced.

**Two dead ends were tested and ruled out**, so don't re-try them:
1. Restricting both arms to the covered 173 — one positive case is not measurable.
2. A December-2022 window (31 of 36 events in 80 segments, a seventh of the work). Its
   repeat base rate is **38.8% vs 6.9%** corpus-wide, so best-F1 jumps 0.162→0.513 for
   reasons unrelated to canonicalisation, and at FMR≤5% — Step 13's *primary* metric — it
   recovers exactly **one** event. Verified on raw embeddings through Step 10's own harness.

**To actually run Step 13** you must rewrite the remaining 350 segments (3.78M chars ≈ **2.7×
the work already done**):
```bash
python handoff_canon_batch/split_batch.py --input steps/12_joint_pipeline/outputs/canon_input.json
# → new part_A/part_B over everything still raw; same MODEL_KEY ('31B') as before
```
Downstream is already wired: `build_segments.py` → `embed_canonical.py` → `rq_eval.py`, and
`comparison_valid` flips true once fallback drops under 10%. **Recommendation: don't.** It is
a bonus ablation; the report covers canonicalisation honestly on Step 11's numbers.

**One result the batch produced anyway** (already in the report): median compression to
**25%** of original length, 112/173 below 40%, against a prompt that forbids shortening
(`אל תסכם ואל תקצר`). That is the mechanism for Step 11's precision-up/recall-down finding —
a ~130-word distillation drops the ministry names and request numbers linking matches on.

---

## 4. Files changed this session

### New
```
report/README.md                                  ← which PDF is current, and why
report/acl/{main.tex,bib.bib,main.pdf,...}         ← the submission
steps/10_temporal_linking/code/honest_levers.py    ← matched lever ablation (CPU ~3 min)
steps/10_temporal_linking/outputs/honest_levers.json
steps/12_joint_pipeline/code/verify_operating_points.py  ← CPU repro of journal points (~2 min)
steps/12_joint_pipeline/outputs/operating_points.json
```

### Modified
```
steps/10_temporal_linking/code/make_plots.py   ← derives all values; literal f1s list removed
steps/10_temporal_linking/README.md            ← Family D from JSON + batch-fit warning;
                                                 fingerprints removed from honest leaderboard;
                                                 leaderboard replaced with matched table
steps/12_joint_pipeline/README.md              ← operating points + embedding-pass caveat
handoff_canon_batch/README.md                  ← corrected the false "350 exist" claim
REMAINING_WORK.md                              ← §3b closed; submission section rewritten
```

### Removed
`canonical_part_A.json`, `canonical_part_B (1).json` from the repo root — verified
byte-identical (sha256) to the copies now in `handoff_canon_batch/`.

---

## 5. What is left

### Required
- [ ] **Commit.** Nothing is committed. Suggest a branch off `master`, not a direct commit.
- [ ] **Decide on `[review]`** — currently off. One-word change in `main.tex` line 7 if the
      course wants line numbers.

### Optional, in order of value
- [ ] Steps 10 / 11 / 12 have no `report/*.pdf` of their own (01–09 all do). Not required if
      the top-level report stands alone, but it is the one visible asymmetry in the repo.
- [ ] Step 13 canonicalisation — see §3. 2.7× the completed work for one bonus row.
- [ ] `merge_canonical.py` crashes on Windows printing the Hebrew line; needs
      `PYTHONIOENCODING=utf-8` or a one-line encoding fix.

### Do NOT do
- Don't re-run the canonicalisation batch that already completed — it succeeded; the problem
  is which segments it covered.
- Don't reintroduce hard-coded values in `make_plots.py`. That drift is what produced the
  0.211 error.
- Don't put batch-fit and streaming-honest rows in one table. That error occurred **three
  times** in this project and cost real credibility each time.

---

## 6. Environment notes (Windows)

- Python is `py`, not `python` (the `python` on PATH lacks numpy — it is LibreOffice's).
- Prefix scripts that print Hebrew with `PYTHONIOENCODING=utf-8` or they crash on cp1252.
- LaTeX is MiKTeX; `pdflatex`/`bibtex` are on PATH.
- `honest_levers.py` and `verify_operating_points.py` each take 2–3 min (523 SVD refits);
  run them in the background.

---

## 7. The result, for whoever presents this

> A representation that separates same- from different-matter segments at **AUC 0.954**
> collapses to **F1 0.125** when the same decision is made online against a growing journal.
> The cause is not Hebrew embedding quality — retrieval places the correct entry in the top
> 10 for **all 36** repeats. It is the streaming decision itself: at a **6.9%** base rate
> against hundreds of candidates, expected false merges scale as n·FPR while true positives
> stay at one. What closes the gap is structural, not representational — judge
> *distinctiveness*, not *similarity*. Every LLM verifier scored at or below the CPU-only
> geometric method.

Plus a second, genuinely transferable contribution: **two look-ahead leaks** documented in
our own pipeline, and the resulting rule that on streaming benchmarks the *fitting protocol*
must be reported alongside the metric.
