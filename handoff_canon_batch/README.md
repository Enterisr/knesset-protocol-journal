# Canonicalization batch — split across two members, run on Colab

Finishes the canonicalization of the Step-7 gold segments so **Step 13** can compare
streaming linking on RAW vs CANONICAL text of the same segments.

## ⚠️ STATUS 2026-08-01 — this batch is DONE, and it is not enough

Both parts ran to completion on `google/gemma-4-31B-it`: `canonical_part_A.json` (86)
and `canonical_part_B.json` (87), 173 records, none empty, 22 with `json_ok: false`
(canonical text kept, `subject`/`decision` empty). `merge_canonical.py` writes
`steps/12_joint_pipeline/outputs/canonical_gold.json` cleanly.

**The earlier claim on this line — that the other 350 segments "were canonicalized
somewhere else" — was never true.** They are not in this repo, not in the
`knesset_canon` Drive folder, and not in Tomer's shared folder (which holds only his
own 19-protocol `*.canonical.json` files, cut at a different segment granularity and
not keyed by our `seg_key`). Nothing was lost; that batch was never produced.

Coverage is therefore **173/523 = 33.1%**, 66.9% falls back to raw, and `rq_eval.py`
sets `comparison_valid: false`.

### Worse than the coverage number: it is the wrong third

The batch spans **2023-05 → 2023-10** (64 of 211 protocols) — the tail of the corpus.
Recurring matters cluster earlier, so:

| | |
|---|---|
| repeat events fully inside the batch | **1 of 36** |
| recurring matters with zero coverage | 21 of 22 |
| segments of the 22 recurring matters that were rewritten | 2 of 58 |

So the tempting fallback — run both arms on the covered 173 only — is also dead: a
raw-vs-canonical contrast decided by one positive case is a coin flip.

A December-2022 window was checked as an alternative (31 of 36 events in just 80
segments) and rejected: the repeat base rate there is 38.8% vs 6.9% corpus-wide, so
best-F1 jumps 0.162 → 0.513 for reasons unrelated to canonicalization, and at the
FMR≤5% operating point — Step 13's *primary* metric — the window recovers exactly one
event. Numbers from it cannot sit beside anything else in the report.

### To actually run Step 13

Rewrite the remaining **350** segments (3.78M chars ≈ 2.7× the work already done):

```bash
python handoff_canon_batch/split_batch.py --input steps/12_joint_pipeline/outputs/canon_input.json
# -> new part_A/part_B over everything still raw; same MODEL_KEY as before ('31B')
```

Per `REMAINING_WORK.md` §5 this is a **bonus ablation, not the critical path** — Step
11's caveated numbers stand and the project is unaffected without it.

### One result this batch produced anyway

Median canonical/raw length ratio **0.25**, with **112 of 173 below 0.4**, against a
prompt that explicitly forbids shortening (`אל תסכם ואל תקצר`). The canonical arm is
therefore aggressive abstractive summarization, not register normalization alone —
which is a mechanism for Step 11's precision-up / recall-down result, since a ~130-word
distillation drops the ministry names and request numbers linking matches on. Quotable
in the report whether or not Step 13 ever runs.

## The split

`split_batch.py` balances by **total characters**, not segment count: segments run
107–20,677 chars (mean 8,116) and generation time tracks length, so an equal-count split
would leave one member waiting long after the other.

| file | segments | chars | share |
|---|---|---|---|
| `part_A.json` | 86 | 702,012 | 50.0% |
| `part_B.json` | 87 | 702,082 | 50.0% |

Disjoint, and their union is exactly the 173 input keys (asserted by the script).

## Who does what

| | member | file | Drive |
|---|---|---|---|
| **Part A** | you | `part_A.json` | your `MyDrive/knesset_canon/` |
| **Part B** | your partner | `part_B.json` | their `MyDrive/knesset_canon/` |

Each member works in their **own** Drive — nobody needs write access to the other's.

## Steps

1. Send your partner `part_B.json` and `canonicalize_part.ipynb`.
2. Each member: upload your part to `MyDrive/knesset_canon/part_<X>.json`, open
   `canonicalize_part.ipynb` in Colab, set **`PART`** in Cell 1 to your letter, run.
3. **Agree on `MODEL_KEY` before starting.** Default `'31B'`
   (`google/gemma-4-31B-it`, ~18 GB at 4-bit) needs an A100. Cell 1 checks the GPU and
   stops with the fix if it does not fit. If either member has to drop to `'26B-A4B'` or
   `'12B'`, **both should** — otherwise the canonical arm is a blend of two rewriters.
4. **Do not skip the Cell 5 smoke test.** One segment, ~1 minute, and it prints an ETA.
   Stop and message the group if JSON did not parse, the length `ratio` is below ~0.4
   (Gemma is summarizing, which the prompt forbids), or the text opens with the subject
   as a header line (Step 13 validity gate 2).
5. Cell 8 renders an RTL raw→canonical review page, in the style of `04_review.ipynb`.
6. **Only once Cell 7 reports `missing: 0`**, both members put their downloaded file here
   as `canonical_part_<X>.json`, then:

```bash
python handoff_canon_batch/merge_canonical.py            # add --extra <file> for the 350
python steps/12_joint_pipeline/code/build_segments.py
sbatch sbatch/13_canonical_rq.sh
```

`merge_canonical.py` prints coverage against all 523, a `source` histogram and the length
ratio distribution, and warns loudly if coverage misses the 10% gate. **Read that report
before spending the GPU job.**

## Files

| file | what |
|---|---|
| `segments_to_canonicalize.json` | the 173 remaining segments (input) |
| `AGENT_INSTRUCTIONS.md` | original brief — the prompt and output schema |
| `04_review.ipynb` | Tomer's NLP_ADV review notebook; the house style this follows |
| `split_batch.py` | char-balanced split → `part_A.json`, `part_B.json` |
| `make_part_notebook.py` | generates `canonicalize_part.ipynb` (edit here, not the .ipynb) |
| `canonicalize_part.ipynb` | **the Colab notebook** — one file, `PART` switch |
| `merge_canonical.py` | parts → `steps/12_joint_pipeline/outputs/canonical_gold.json` |

## Notes

- The notebook uses **Tomer's exact** `CANON_SYSTEM` / `CANON_INSTRUCTIONS` and greedy
  decoding, so these segments are canonicalized the same way notebook 03 does its own.
- Records are a **flat list keyed by `seg_key`** — `build_segments.py:52` maps on that
  key. 04's per-protocol `{'segments': [...]}` nesting is deliberately not used; only its
  scalar fields (`n_chars_raw`, `n_chars_canonical`, `ratio`) are.
- Output carries `source` (the model id) so the merge report can show how blended the
  canonical arm is.
- The run is **resumable** — the output is rewritten to Drive after every segment, so a
  Colab disconnect costs at most one segment.
- Input was already capped at 3,500 words by `export_for_canon.py`; 169 of the 523 hit
  that cap, so for those the canonical text distils a *truncated* input. Step 13's README
  lists this as validity gate 3 and it must be stated in the report.

## Observed compression — must go in the Step 13 report

First smoke test (`25_ptv_2827710__2`, part A, Gemma-4-31B): **6,634 chars → 720 chars,
ratio 0.109**, JSON parsed cleanly. Not truncation — a truncated generation is cut off
mid-JSON and fails to parse. Gemma genuinely compresses, despite the prompt's
`אל תסכם ואל תקצר`.

The cause is our segment granularity: Tomer's segments are fine-grained, ours are coarse
(up to 5,000 words), so the model condenses rather than rewrites. The prompt is
deliberately left unchanged — altering it would break comparability with both the 350
already canonicalized and Tomer's own pipeline.

Two consequences for the write-up:

1. The canonical arm tests **aggressive abstractive summarization + register
   normalization**, not register normalization alone. Step 12's README predicted a
   "~600-word canonical distillation"; at ratio ~0.11 the output is closer to 130 words,
   i.e. *shorter* than the ~512 tokens e5 would happily consume.
2. It offers a mechanism for Step 11's precision-up / recall-down result: a 130-word
   summary drops the entity detail (ministry, request number) that linking matches on, so
   fewer pairs clear threshold — higher precision, collapsed recall. Worth stating, since
   Step 13's honest prior expects exactly that direction.

`merge_canonical.py` prints the full ratio distribution over the merged corpus; quote that
in the report rather than this single observation.
