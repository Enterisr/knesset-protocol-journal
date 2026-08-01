# Step 13 — Canonicalization ablation: content-level vs geometric normalization

> **Does *content-level* normalization (LLM canonicalization) remove the shared
> bureaucratic register better than *geometric* normalization (ABTT / whitening) —
> and does that convert into anti-duplication performance?**

## Scope: this is one ablation, not the project's research question

Canonicalization is **Tomer's** research question (the red annotation on the project
diagram). `CLAUDE.md`'s three tasks and four milestones never mention it. This project's
own question is why AUC 0.954 separability yields only F1 0.190 streaming linking — see
`REMAINING_WORK.md` §0. Step 13 slots into that as a single row: content-level
normalization, tested against the geometric normalization of Steps 03/04/10.

**It is a bonus experiment, not the critical path.** If the Colab pass doesn't land, we
lose one ablation and report Step 11's caveated numbers instead.

## The primary metric is NOT F1

Step 11's only downstream measurement:

| variant | F1 | precision | recall |
|---|---|---|---|
| raw | 0.114 | 0.073 | 0.261 |
| canonical | **0.095** | **0.154** | 0.069 |

Canonicalization made F1 *worse* — it doubled precision and collapsed recall. Under this
project's asymmetric cost rule (false-merge ≫ false-split) that trade is the **right
direction**, and F1 is the wrong instrument for detecting it. So the headline metric here
is **recall at false-merge rate ≤ 5%**, with F1 reported alongside for comparability.
Step 11 reported best-F1 and concluded "mixed"; that conclusion is an artifact of the metric.

**Honest prior: we expect precision up, recall down, F1 flat-to-worse.** The open question
is whether the anti-duplication operating point improves — that is what would make
canonicalization worth adopting.

## Why the experiment is still worth running

It is the one every step since 02 has been circling, and each step narrowed it:

| Step | Finding | What it implied |
|---|---|---|
| 02 | dense embeddings severely anisotropic (cross-topic cos ≈ 0.95); TF-IDF beat them | signal is lexical, not semantic |
| 03 | ABTT de-anisotropization recovers dense models | the anisotropy is a *removable* nuisance direction |
| 04 | hybrid TF-IDF+dense, ARI 0.569 | lexical and semantic signal are complementary |
| 08 | root cause, verified on real text: "budget surplus" template makes a Ministry-of-**Health** distractor (cos 0.66) beat the true Ministry-of-**Education** match (cos 0.23) | the confound is **shared bureaucratic register** |
| 10 | geometric fixes work but **whitening leaks** — batch 0.250 → honest 0.094 | geometry can only partly remove it *honestly* |
| 11 | canonicalization sharpens intrinsic AUC 0.947 → 0.986 | content-level normalization may remove what geometry can't |

Step 11 pointed at exactly this experiment as "the clean, definitive test" and named its
own two caveats. Step 13 removes both.

## What Step 11 could not do, and this does

| | Step 11 | Step 13 |
|---|---|---|
| labels | LLM `subject` from the **same pass** that wrote the canonical text → circular | **manual gold chains** (Step 06/07), independent of the canonicalizer |
| scale | 19 protocols, 29 exact-subject repeats | **523 segments, 22 recurring topics, 36 repeat events** |
| linking eval | pseudo-gold, unstable F1 | Step 10's full benchmark, directly comparable to every published number |

## Measurement surfaces

Everything held fixed except the text (same encoder, transforms, linking rule, gold):

1. **Intrinsic** — same/different-matter ROC-AUC over segment pairs, reported on all
   pairs and on the recurring-only subset (the harder discrimination; Step 07's setup).
2. **Downstream** — Step 10's streaming-honest linking (running-fit transform +
   accumulated centroid + margin gate), at **both** operating points:
   - **best-F1**, θ ≈ 0.156 for abtt1
   - **anti-duplication**, false-merge rate ≤ 5%, θ ≈ 0.342 — the point the project's
     asymmetric cost model actually cares about

**Surface 2 at FMR≤5% is the headline.** At that point the method links only 2–3 of 36
repeats, so an intrinsic-AUC gain that does not move it means canonicalization sharpened
the geometry without solving the problem the project actually poses. Conversely, a gain
there would be worth adopting *even if F1 drops*, which is the outcome Step 11 saw and
misread as "mixed".

## Run

```bash
# 0. PREREQ (the gate — not under our control, start it first):
#    upload steps/12_joint_pipeline/outputs/canon_input.json to Drive,
#    run steps/12_joint_pipeline/notebooks/canonicalize_gold.ipynb (Gemma-4-31B 4-bit),
#    drop its output at steps/12_joint_pipeline/outputs/canonical_gold.json
sbatch sbatch/13_canonical_rq.sh
```

The `raw` side needs no GPU work — Step 07's `embeddings/e5.npy` already holds e5 vectors
for exactly these 523 segments in exactly this order, so reusing it keeps the comparison
free of encoder drift. Only the canonical half is embedded.

## Outputs
- `outputs/embeddings/e5_canonical.npy`, `outputs/embeddings/meta.json`
- `outputs/rq_results.json` — intrinsic + downstream, raw vs canonical
- `outputs/fig_rq.png`

## Validity gates — check these before believing the answer

1. **Fallback fraction.** Segments with no usable canonical rewrite fall back to raw text,
   which makes the canonical arm a raw/canonical *blend* and biases the comparison toward
   "no effect". `rq_eval.py` records `frac_fallback_to_raw` and sets
   `comparison_valid: false` above 10%, with a loud warning. **A null result under a high
   fallback fraction means the Colab pass failed, not that canonicalization is ineffective.**
2. **Subject leakage.** If Gemma prepends its `subject` to the rewritten text, the canonical
   arm gets a clean topic title the raw arm never has — inflating separation for a reason
   unrelated to register. `build_segments.py` strips such headers and reports the count.
   Still worth eyeballing 3–5 canonical texts when the file lands.
3. **Input truncation.** 169 of 523 segments exceeded the 3500-word cap in
   `export_for_canon.py`, so for a third of the corpus the canonical text is a distillation
   of a *truncated* input. Must be stated in the report. (Raw is truncated too — e5 only
   reads ~512 tokens — which is precisely why a short canonical distillation of the whole
   segment might beat it.)

## Interpreting the outcome — all three directions are a result

- **FMR≤5% recall improves** → content-level normalization does what geometric
  normalization could not do honestly. The one outcome that would justify adopting
  canonicalization in the pipeline.
- **Precision up, recall down, FMR≤5% flat** (the expected outcome) → canonicalization
  makes the representation more conservative but does not shift the anti-duplication
  frontier. Confirms Step 11 on clean gold and *explains* its "mixed" result as a metric
  artifact.
- **Nothing moves** → Step 11's intrinsic gain was label circularity, and the ceiling is
  set elsewhere — consistent with §0's base-rate argument, that the bottleneck is the
  streaming decision structure rather than representation quality.

All three are reportable, and none of them is the project's headline — that comes from
the linking analysis in `REMAINING_WORK.md` §0, which depends on nothing external.

## Honest scope note

The **end-to-end journals** (`journal_{raw,canonical}_*.json` from Step 12) are
**demonstration artifacts, not an evaluation surface.** Linking precision is ~0.13 at
best-F1 and recall is 2–3/36 at the anti-duplication point, so an end-to-end quality
number would conflate linking error with summarization error. Summarization is evaluated
on **gold chains** (Step 09's precedent, which is forced rather than convenient).
