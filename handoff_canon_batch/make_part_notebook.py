"""
Generate `canonicalize_part.ipynb` -- the Colab notebook each team member runs on THEIR
half of the remaining segments.

Written in the style of Tomer's `04_review.ipynb`: numbered `# === Cell N ===` sections,
a configuration cell whose asserts each name the fix, a print summary after every stage,
and a final RTL HTML review page written to Drive.

ONE notebook for both members, with a `PART = 'A' | 'B'` switch in the config cell. Two
copies would be free to drift in prompt or decoding config, which would make the two
halves of the canonical arm incomparable; a single artifact cannot.

Two deliberate departures from 04_review:
  * `BASE` stays on the member's own `MyDrive/knesset_canon` (commit fdc2ab0 moved off the
    shared `NLP ADVANCED/FinalProject/NLP_ADV` folder on purpose). Two members on two
    Drives also means neither needs write access to the other's.
  * Records are a FLAT list keyed by `seg_key`, not 04's per-protocol
    `{'segments':[{seg_id,...}]}` nesting -- `build_segments.py` maps on `seg_key`.
    Only 04's scalar fields (`n_chars_raw`, `n_chars_canonical`, `ratio`) are adopted.

Usage (CPU): python handoff_canon_batch/make_part_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": src.splitlines(keepends=True)}


INTRO = r"""# Canonicalize your part of the gold segments (Gemma-4, Tomer's prompt)

Runs the NLP_ADV canonicalization over **your half** of the 173 remaining Step-7 gold
segments, so Step 13 can compare streaming linking on RAW vs CANONICAL text of the *same*
segments. Same prompt and decoding as notebook 03; only the input set differs.

## Runbook

1. **Runtime -> Change runtime type -> GPU.** Which GPU you get decides `MODEL_KEY` in
   Cell 1 — that cell checks the fit and stops with the fix if it is wrong.
2. **Upload your part** to Drive at `MyDrive/knesset_canon/part_<YOUR PART>.json`
   (`part_A.json` / `part_B.json`, from `handoff_canon_batch/split_batch.py`).
3. **Set `PART`** in Cell 1 to `'A'` or `'B'` — whichever file you uploaded.
4. Run every cell top to bottom. **Cell 5 is a one-segment smoke test — read its output**
   before letting Cell 6 run for hours.
5. Cell 8 renders an RTL review page (raw -> canonical, 04-style) so you can eyeball the
   result. Then download `canonical_part_<PART>.json` and see the last markdown cell.

**It is resumable.** Colab will disconnect. After a disconnect the runtime is empty, so
re-run **Cells 1–4, then Cell 6** — finished segments are skipped, and the output is
rewritten to Drive after every segment, so a disconnect costs at most one segment.

No HF token needed: `google/gemma-4-31B-it` is not a gated repo.
"""

CELL1 = r"""# === Cell 1: Drive mount + configuration ===
import json
from pathlib import Path
import torch
from google.colab import drive
drive.mount('/content/drive')

PART = 'A'          # <<< 'A' or 'B' -- must match the file you uploaded

# 4-bit footprints below are weights only; the rest of the GPU holds the KV cache for
# inputs running to ~10k Hebrew tokens, so `need_gb` is a total-device-memory floor.
MODELS = {
    '31B':     dict(model_id='google/gemma-4-31B-it',     need_gb=35),  # reference model (A100)
    '26B-A4B': dict(model_id='google/gemma-4-26B-A4B-it', need_gb=28),  # MoE, 4B active -- faster
    '12B':     dict(model_id='google/gemma-4-12B-it',     need_gb=15),  # T4 / L4 fallback
}
MODEL_KEY = '31B'   # <<< drop to '26B-A4B' or '12B' if the GPU assert below fires.
                    #     IF YOU CHANGE THIS, TELL YOUR PARTNER -- both parts should use the
                    #     same model, or the canonical arm is a blend of two rewriters.

BASE     = Path('/content/drive/MyDrive/knesset_canon')   # your own Drive, not the shared
                                                          # NLP ADVANCED/FinalProject/NLP_ADV
EVAL_DIR = BASE / 'eval'
IN_PATH  = BASE / f'part_{PART}.json'
OUT_PATH = BASE / f'canonical_part_{PART}.json'
# max_new is a cap, not a target -- generation stops at EOS, so a generous cap is free.
# 1024 was tight: a long segment that wants more gets cut off mid-JSON, which shows up as
# a parse failure rather than as truncation.
CFG = dict(**MODELS[MODEL_KEY], max_new=1536, cap_words=3500)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

assert PART in ('A', 'B'), "PART must be 'A' or 'B'"
assert IN_PATH.exists(), f'missing {IN_PATH} - upload part_{PART}.json to that folder'
assert torch.cuda.is_available(), 'no GPU - Runtime -> Change runtime type -> GPU'
gpu, gpu_gb = torch.cuda.get_device_properties(0), torch.cuda.get_device_properties(0).total_memory / 1e9
assert gpu_gb >= CFG['need_gb'], (
    f"{gpu.name} has {gpu_gb:.0f}GB but {MODEL_KEY} needs ~{CFG['need_gb']}GB at 4-bit - "
    f"pick a smaller MODEL_KEY above (and tell your partner), or get a bigger runtime")

SEG_IN = json.loads(IN_PATH.read_text(encoding='utf-8'))
print('part          :', PART)
print('gpu           :', gpu.name, f'{gpu_gb:.0f}GB')
print('model         :', CFG['model_id'])
print('segments      :', len(SEG_IN))
print('chars         :', f"{sum(len(r['raw']) for r in SEG_IN):,}")
print('already done  :', len(json.loads(OUT_PATH.read_text(encoding='utf-8'))) if OUT_PATH.exists() else 0)
"""

CELL2 = r"""# === Cell 2: load Gemma-4 (4-bit nf4) ===
!pip -q install -U transformers accelerate bitsandbytes
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

processor = AutoProcessor.from_pretrained(CFG['model_id'])
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16)
model_hf = AutoModelForMultimodalLM.from_pretrained(
    CFG['model_id'], quantization_config=bnb, device_map='auto')
model_hf.eval()
print('loaded        :', CFG['model_id'])
print('gpu allocated :', round(torch.cuda.memory_allocated() / 1e9, 1), 'GB')
"""

CELL3 = r"""# === Cell 3: helpers (Hebrew-robust JSON extraction) ===
# Gemma writes Hebrew abbreviations with straight quotes for gershayim (מע"מ, בג"ץ), which
# breaks json.loads from inside a string. _json_objects finds balanced {...} spans without
# trusting the quoting; _salvage_json escapes a quote sitting between two Hebrew letters,
# which is always an abbreviation mark and never a JSON delimiter.
import json, re

def _json_objects(s):
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield s[start:i + 1]

def _salvage_json(s):
    return re.sub(r'(?<=[֐-׿])"(?=[֐-׿])', '\\"', s)

_probe = '{"canonical": "ועדת הכספים דנה במע"מ", "subject": "מע"מ"}'
assert json.loads(_salvage_json(next(_json_objects(_probe))))['subject'] == 'מע"מ'
print('helpers ok')
"""

CELL4 = r"""# === Cell 4: canonicalization prompt (Tomer's exact CANON_SYSTEM + CANON_INSTRUCTIONS) ===
import json, torch

CANON_SYSTEM = ('אתה עורך לשוני. אתה מקבל מקטע דיון מוועדת הכספים של הכנסת, ובו דוברים שונים '
                'בסגנונות ואוצר מילים שונים. תפקידך לנסח מחדש את תוכן הדיון בלשון אחידה, '
                'ניטרלית ועניינית - לשמר את כל המידע (נושאים, עמדות, נתונים, החלטות) אך '
                'להסיר סגנון אישי, רטוריקה, ומאפייני דובר. אתה לא מסכם ולא מקצר - אתה מנסח מחדש.')

CANON_INSTRUCTIONS = '''נסח מחדש את מקטע הדיון הבא בלשון אחידה וניטרלית.

עקרונות:
- שמר את כל התוכן העובדתי: מה נדון, אילו עמדות הוצגו, נתונים מספריים, החלטות והצבעות.
- אחֵד את הרגיסטר: אותו אוצר מילים ענייני לכל הדוברים, ללא סלנג, רטוריקה, ברכות או ציטוט סגנוני.
- אל תסכם ואל תקצר באופן אגרסיבי - שמור על אורך דומה לתוכן המהותי של המקטע.
- כתוב כטקסט רציף בגוף שלישי ("הוצגה עמדה ש...", "סוכם כי...", "התקיימה הצבעה ש...").
- אל תוסיף פרשנות או מידע שאינו במקטע.

החזר JSON יחיד בלבד:
{"subject": "<כותרת נושא קצרה 3-6 מילים>", "canonical": "<הניסוח המחדש הניטרלי>", "decision": "<אושר/נדחה/ללא הצבעה/נדחה להמשך>", "amounts": ["<סכומים או מספרי פניות>"]}'''

@torch.no_grad()
def _generate_canon(prompt):
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': CANON_SYSTEM}]},
                {'role': 'user',   'content': [{'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors='pt').to(model_hf.device)
    out = model_hf.generate(**inputs, max_new_tokens=CFG['max_new'], do_sample=False)
    return processor.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

def _first_canon_json(txt):
    for obj in _json_objects(txt):
        for cand in (obj, _salvage_json(obj)):
            try: d = json.loads(cand)
            except json.JSONDecodeError: continue
            if 'canonical' in d: return d
    raise ValueError('no canonical json')

def canonicalize(row):
    # -> a flat record keyed by seg_key (build_segments.py maps canonical text on that key)
    raw_txt = ' '.join(row['raw'].split()[:CFG['cap_words']])
    gen = _generate_canon(f"{CANON_INSTRUCTIONS}\n\n--- המקטע ---\n{raw_txt}")
    try:
        c, ok = _first_canon_json(gen), True
    except Exception:
        # Keep the text rather than lose the segment, but flag it: a record whose JSON never
        # parsed has no subject/decision and its `canonical` may carry a preamble.
        c, ok = {'subject': '', 'canonical': gen.strip(), 'decision': '', 'amounts': []}, False
    canon = c.get('canonical', '')
    return {'seg_key': row['seg_key'], 'date': row['date'],
            'subject': c.get('subject', ''), 'canonical': canon,
            'decision': c.get('decision', ''), 'amounts': c.get('amounts', []),
            'n_chars_raw': len(raw_txt), 'n_chars_canonical': len(canon),
            'ratio': round(len(canon) / max(len(raw_txt), 1), 3),
            'source': CFG['model_id'], 'part': PART, 'json_ok': ok}
print('prompt ready  :', len(CANON_INSTRUCTIONS), 'chars of instructions')
"""

SMOKE_MD = r"""## Smoke test — do not skip

One segment, about a minute. It catches a broken chat template, a bad prompt or an OOM
*before* you spend hours on it, and it gives you the per-segment rate to project an ETA.
"""

CELL5 = r"""# === Cell 5: smoke test on one segment ===
import time
t0 = time.time()
rec = canonicalize(SEG_IN[0])
dt = time.time() - t0

print(f"{rec['seg_key']} | {dt:.0f}s | json parsed: {rec['json_ok']}")
print('subject  :', rec['subject'])
print('decision :', rec['decision'], '| amounts:', rec['amounts'])
print('length   :', f"raw {rec['n_chars_raw']} -> canonical {rec['n_chars_canonical']} chars (x{rec['ratio']})")
print(f"ETA for your {len(SEG_IN)} segments at this rate: ~{dt * len(SEG_IN) / 3600:.1f} h")
print('-' * 70)
print(rec['canonical'][:800])

# STOP and message the group if any of these is true -- do not start the long run:
#  * json parsed: False           -> prompt or decoding is broken
#  * ratio below ~0.4             -> Gemma is summarizing, which the prompt forbids
#                                    ("אל תסכם ואל תקצר") and which would confound Step 13
#  * the text starts with the subject as a header line
#                                 -> Step 13 validity gate 2; build_segments.py strips the
#                                    common forms, but tell the group so it gets checked
#  * the text is English, or is JSON rather than prose
"""

RUN_MD = r"""## The long run

Hours. Resumable — after a disconnect the runtime is wiped, so re-run **Cells 1–4** (the
model reload is the slow one) and then this cell. It picks up where it stopped.
"""

CELL6 = r"""# === Cell 6: resumable batch over your part ===
import time

done = {c['seg_key']: c for c in (json.loads(OUT_PATH.read_text(encoding='utf-8')) if OUT_PATH.exists() else [])}
todo = [r for r in SEG_IN if r['seg_key'] not in done]
print(f'{len(done)}/{len(SEG_IN)} already done; {len(todo)} to go')

t0, n_bad = time.time(), 0
for n, row in enumerate(todo, 1):
    rec = canonicalize(row)
    n_bad += (not rec['json_ok'])
    done[rec['seg_key']] = rec
    OUT_PATH.write_text(json.dumps(list(done.values()), ensure_ascii=False), encoding='utf-8')
    if n % 5 == 0 or n == len(todo):
        rate = (time.time() - t0) / n
        print(f'  {len(done)}/{len(SEG_IN)} | {rate/60:.1f} min/seg | {n_bad} json failures'
              f' | ETA {rate * (len(todo) - n) / 3600:.1f} h')

print(f'DONE part {PART}: {len(done)}/{len(SEG_IN)} -> {OUT_PATH}')
"""

CELL7 = r"""# === Cell 7: verify + download ===
out = json.loads(OUT_PATH.read_text(encoding='utf-8'))
missing = {r['seg_key'] for r in SEG_IN} - {c['seg_key'] for c in out}
empty   = [c['seg_key'] for c in out if not c.get('canonical', '').strip()]
ratios  = sorted(c['ratio'] for c in out if c.get('n_chars_raw'))

print(f'segments      : {len(out)}/{len(SEG_IN)}')
print(f'missing       : {len(missing)}')
print(f'empty canon   : {len(empty)}')
print(f'json failures : {sum(1 for c in out if not c.get("json_ok", True))}')
if ratios:
    print(f'ratio         : min {ratios[0]:.2f} | median {ratios[len(ratios)//2]:.2f} | max {ratios[-1]:.2f}')
    print(f'  (ratio < 0.4 means summarized rather than rewritten: {sum(r < 0.4 for r in ratios)} segments)')
if missing:
    print('MISSING - re-run Cell 6:', sorted(missing)[:10])

from google.colab import files
files.download(str(OUT_PATH))
"""

CELL8 = r"""# === Cell 8: RTL review page (raw -> canonical), 04-style ===
import html
from IPython.display import HTML, display

N_SHOW    = 10   # how many segments to render
MAX_CHARS = 0    # 0 = full raw text; set e.g. 1500 to truncate

def esc(s):
    return html.escape(str(s or ''))

def build_page(rows, canon_by_key, n_show=N_SHOW, max_chars=MAX_CHARS):
    head = f'part {PART} | {len(canon_by_key)}/{len(rows)} canonicalized | {CFG["model_id"]}'
    out = ["<style>body{background:#fafafa}</style><meta charset='utf-8'>",
           "<div dir='rtl' style='font-family:sans-serif;max-width:960px;margin:auto;"
           "background:#fafafa;color:#111;padding:8px 16px'>",
           f"<h2 dir='ltr' style='text-align:center;color:#111'>{esc(head)}</h2>"]
    for row in rows[:n_show]:
        c = canon_by_key.get(row['seg_key'])
        txt = row['raw'].strip()
        if max_chars and len(txt) > max_chars:
            txt = txt[:max_chars] + ' …'
        out.append("<div style='background:#f5f5f0;color:#111;padding:12px;margin:14px 0 0;"
                   "border-radius:8px 8px 0 0;border:1px solid #ddd'>")
        out.append(f"<b>— {esc(row['seg_key'])} · {esc(row['date'])} · "
                   f"{len(row['raw'])} chars —</b>")
        out.append(f"<p style='margin:6px 0'>{esc(txt)}</p></div>")
        if c:
            flag = '' if c.get('json_ok', True) else " · <span style='color:#993c1d'>JSON לא נפרס</span>"
            meta = ' · '.join(x for x in [
                esc(c.get('decision')),
                esc(', '.join(map(str, c.get('amounts') or []))),
                f"raw {c['n_chars_raw']} → canonical {c['n_chars_canonical']} chars "
                f"(x{c['ratio']})"] if x)
            out.append("<div style='background:#e8f3e8;color:#111;padding:12px;margin:0 0 14px;"
                       "border-radius:0 0 8px 8px;border:1px solid #cde3cd;border-top:0'>")
            out.append(f"<b>✎ קנוניזציה — {esc(c.get('subject'))}</b>"
                       f"<div style='color:#555;font-size:13px'>{meta}{flag}</div>")
            out.append(f"<p style='margin:8px 0 0'>{esc(c.get('canonical'))}</p></div>")
        else:
            out.append("<div style='background:#fdf0f0;padding:8px 12px;margin:0 0 14px;"
                       "border-radius:0 0 8px 8px;border:1px solid #eecccc;border-top:0;"
                       "color:#993c1d'>✗ אין קנוניזציה למקטע זה</div>")
    out.append('</div>')
    return '\n'.join(out)

canon_by_key = {c['seg_key']: c for c in json.loads(OUT_PATH.read_text(encoding='utf-8'))}
# Render the segments that HAVE a canonical, so the page is useful mid-run too -- otherwise
# a partial run shows ten red "no canonicalization" strips and none of the green blocks.
rows = [r for r in SEG_IN if r['seg_key'] in canon_by_key] or SEG_IN
page = build_page(rows, canon_by_key)
out_fp = EVAL_DIR / f'review_canon_part_{PART}.html'
out_fp.write_text(page, encoding='utf-8')
print('saved ->', out_fp)
display(HTML(page))
"""

HANDBACK = r"""## Handing the result back

Put the downloaded file in the repo at `handoff_canon_batch/canonical_part_<PART>.json`
(both parts, from both members), then:

```bash
python handoff_canon_batch/merge_canonical.py
```

It merges both parts with anything already canonicalized, writes
`steps/12_joint_pipeline/outputs/canonical_gold.json`, and reports coverage against all
523 segments plus a histogram of `source` and `ratio`.

**Read that report before running `sbatch/13_canonical_rq.sh`.** `rq_eval.py` sets
`comparison_valid: false` when more than 10% of segments fall back to raw text — under
that condition a null result means the canonicalization pass was incomplete, not that
canonicalization is ineffective.
"""

cells = [md(INTRO), code(CELL1), code(CELL2), code(CELL3), code(CELL4),
         md(SMOKE_MD), code(CELL5), md(RUN_MD), code(CELL6), code(CELL7), code(CELL8),
         md(HANDBACK)]

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}

out = HERE / "canonicalize_part.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, f"({len(cells)} cells)")
