"""
workers/ai_annotations.py — GPT-4o (vision) generates rich annotation events.

Each annotation:
  {
    "type":        "underline" | "double_underline" | "circle" | "box" | "arrow",
    "start_time":  float   (seconds, semantically aligned to spoken concept),
    "target_text": str     (EXACT OCR text — used to locate bbox at render time),
    "bbox":        [x, y, w, h]  (OCR pixel coordinates),
  }

Key idea: the slide is PARAPHRASED from the script (not literal). The model
must do semantic matching — for each meaningful concept in the spoken script,
find the closest matching word / phrase / line on the slide and annotate it.
"""

import json
import re
from typing import Any

from openai import OpenAI
from lib.config import config

_openai = OpenAI(api_key=config.OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are an expert video annotation director for educational explainer videos.

You receive, for ONE chunk of a video:
  1. The slide image (visual layout, colors, emphasis)
  2. The Gamma slide-generation prompt (original creative intent — heading + bullets)
  3. The original script (native language, e.g. Telugu)
  4. The transliterated script (Latin letters — matches the timestamps)
  5. Word-level timestamps for the spoken audio (Latin script)
  6. The full OCR dump of the slide (every word/line with bbox + confidence)
  7. Total audio duration for the chunk

YOUR GOAL
Produce 12–25 high-quality annotations that visually highlight EVERY meaningful
concept being spoken, by drawing on the right place on the slide AT the right
moment in time.

CRITICAL: SEMANTIC MATCHING (not keyword search)
The slide wording is PARAPHRASED from the script — they say the same things
in different words. Examples:
  • Script says "essential facts for competitive exams"
    → Slide shows "Important points for competitive exams"
    → Annotate "Important points for competitive exams" (whole line)
  • Script says "Virat Kohli scored the most runs"
    → Slide shows "Top scorer: V. Kohli"
    → Annotate "V. Kohli" (single word/phrase)
  • Script says a number like "twenty teams"
    → Slide shows "20 Teams"
    → Annotate "20" or "20 Teams"

DO NOT do dumb keyword matching. Think about MEANING. For every concept the
narrator mentions, scan the WHOLE slide (OCR + image) and find the most
semantically related text on the slide. If multiple OCR words form a bullet
line that means the same thing, annotate the WHOLE LINE. If it's just a key
term/number/name, annotate that SINGLE WORD.

ANNOTATION TYPES (vary them intentionally based on slide layout):
  double_underline → THE main slide heading. Use AT MOST ONCE per chunk.
  underline        → SHORT phrases only (2–5 words). Use sparingly — a real
                     tutor doesn't underline full sentences. Max 2 full-line
                     underlines per chunk; the rest must be short phrases.
  circle           → PREFERRED for spotlighting a single keyword, number,
                     name, or 1–3 word phrase. Use this most often.
                     HARD RULE: target_text for a circle MUST be ≤ 4 words
                     AND ≤ 30 characters AND fit on a single visual line.
                     NEVER circle a whole bullet, sentence, or multi-line
                     block — circles are tight keyword loops, not lassos
                     around paragraphs. If the bbox height is more than
                     1.8× the average word height, it's too tall for a
                     circle — switch to a short underline instead.
  box              → frame a statistic, formula, or grouped block
                     (only for already-grouped callouts, not bullets).
  arrow            → point to a list item or a concept being introduced

TIMING RULES
  • start_time = the audio time (in seconds) when the corresponding spoken
    concept BEGINS in the timestamps. Use semantic alignment — find the
    transliterated word(s) in the timestamps that correspond to the
    concept, take the start time of the first such word.
  • Annotations should flow in chronological order, matching the narration.
  • Two annotations may share a start_time only when truly simultaneous.

TARGET TEXT RULES
  • target_text MUST be the EXACT OCR text from the slide (so the renderer
    can look up its bbox). Copy it character-for-character.
  • For multi-word annotations, concatenate the OCR words with single spaces
    EXACTLY as they appear in the OCR list, preserving order.
  • bbox: for a single OCR word, use that word's bbox. For a multi-word
    phrase, use the bounding rectangle that covers all the words in that
    phrase (min x, min y, max x+w, max y+h → [x, y, w, h]).

QUANTITY & DISTRIBUTION
  • Aim for 12–25 annotations per chunk (more if the slide is text-dense,
    fewer if it's very sparse — but always try to cover EVERY major concept
    spoken in the script).
  • Spread annotations across the entire chunk duration. Do not cluster
    them all in the first 2 seconds.
  • Do not annotate the same OCR text more than once unless the script
    references it at clearly distinct times.

OUTPUT FORMAT — return ONLY valid JSON (no markdown fences, no commentary):
{"annotations": [
  {"type": "double_underline", "start_time": 0.0, "target_text": "IPL 2026", "bbox": [137, 349, 121, 65]},
  ...
]}"""


def _ocr_lines(ocr_words: list[dict]) -> str:
    out = []
    for i, w in enumerate(ocr_words):
        text = w.get("text", "")
        conf = w.get("confidence") or w.get("conf") or 0
        out.append(
            f'  [{i:03d}] "{text}"  bbox=[{w.get("x",0)},{w.get("y",0)},{w.get("w",0)},{w.get("h",0)}]  conf={conf}'
        )
    return "\n".join(out)


def _ts_lines(ts_words: list[dict]) -> str:
    out = []
    for w in ts_words:
        word = w.get("word") or w.get("text") or ""
        orig = w.get("original") or ""
        suffix = f"  ({orig})" if orig and orig != word else ""
        out.append(f'  {float(w.get("start",0)):.2f}s → {float(w.get("end",0)):.2f}s  "{word}"{suffix}')
    return "\n".join(out)


def generate_annotations(
    ocr_words: list[dict],
    ts_words: list[dict],
    chunk_text: str,
    chunk_number: int,
    slide_image_url: str | None = None,
    slide_prompt: str | None = None,
    original_script: str | None = None,
) -> list[dict[str, Any]]:
    """
    Call GPT-4o (vision) to generate rich, semantically-aligned annotations.
    """
    print(f"[AI] generating annotations for chunk {chunk_number} "
          f"(ocr_words={len(ocr_words)}, ts_words={len(ts_words)}, "
          f"slide_image={'yes' if slide_image_url else 'no'})")

    total_dur = ts_words[-1].get("end", 10.0) if ts_words else 10.0

    user_text = f"""CHUNK {chunk_number}  —  audio duration: {float(total_dur):.2f}s

=== GAMMA SLIDE PROMPT (original creative intent) ===
{slide_prompt or "(not available)"}

=== ORIGINAL SCRIPT (native language) ===
{original_script or chunk_text or "(not available)"}

=== TRANSLITERATED SCRIPT (matches timestamps) ===
{chunk_text or "(not available)"}

=== WORD-LEVEL TIMESTAMPS (Latin) ===
{_ts_lines(ts_words)}

=== FULL OCR DUMP (every word on the slide with bbox) ===
{_ocr_lines(ocr_words)}

=== TASK ===
1. Look at the slide image — note the heading, bullets, layout.
2. Read the script and identify EVERY meaningful concept the narrator says
   (key terms, numbers, names, definitions, phrases). Aim for one annotation
   roughly every 3–5 seconds of audio, distributed across the WHOLE chunk
   (NOT clustered at the end).
3. For each concept, find the matching word/phrase/line on the slide
   (semantic match — slide wording is paraphrased from the script).
4. STRONGLY PREFER short keyword targets: single words, numbers, names, or
   2–4 word phrases. CIRCLE them when possible — like a tutor circling a
   key term ("Focus", "Facts", "20", "Kohli"). Full-line underlines are
   visually heavy; cap them at MAX 2 per chunk, and only use them when the
   narrator literally summarises the whole bullet.
5. start_time = the timestamp of the FIRST word the narrator says that
   maps to this concept. Use the transliterated timestamps as ground truth.
6. target_text MUST be the EXACT OCR text (copy character-for-character).
   For multi-word targets, concatenate consecutive OCR words with single
   spaces in the order they appear in the OCR dump.
7. Vary types. double_underline at most ONCE (the heading). Lean heavily
   on `circle` for keywords. Mix in short `underline`, occasional `box`
   or `arrow` so it feels like a real tutor.
8. Return 15–25 annotations, chronologically ordered, spread across the
   full duration. NEVER cluster more than 3 annotations in the same 2-second
   window.

Return ONLY the JSON object."""


    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if slide_image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": slide_image_url, "detail": "high"},
        })

    response = _openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=4000,
        temperature=0.3,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group(0)) if m else {}

    annotations = parsed.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    # Sanitise + RECOMPUTE bbox from OCR (GPT often returns wrong bboxes for
    # multi-word phrases — it copies the title's bbox or the first OCR bbox).
    allowed_types = {"underline", "double_underline", "circle", "box", "arrow"}
    clean: list[dict[str, Any]] = []
    for ann in annotations:
        t = ann.get("type")
        if t not in allowed_types:
            continue
        target = str(ann.get("target_text") or "").strip()
        if not target:
            continue

        located = _locate_phrase_bbox(target, ocr_words)
        if located is None:
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                located = [int(v) for v in bbox]
            except (TypeError, ValueError):
                continue

        try:
            clean.append({
                "type":        t,
                "start_time":  max(0.0, float(ann.get("start_time") or 0)),
                "target_text": target,
                "bbox":        located,
            })
        except (TypeError, ValueError):
            continue

    # Safety net: demote oversized "circle" annotations to a short "underline".
    # GPT sometimes circles entire bullets / multi-line blocks, which looks like
    # a lasso around a paragraph. If the bbox is tall (multi-line) or the target
    # text has too many words/chars, switch to underline so it reads as a
    # highlight under the phrase instead of a giant loop.
    if ocr_words:
        avg_h = sum(int(w.get("h", 0)) for w in ocr_words) / max(len(ocr_words), 1)
    else:
        avg_h = 0
    for ann in clean:
        if ann["type"] != "circle":
            continue
        words = ann["target_text"].split()
        bbox_h = ann["bbox"][3] if len(ann["bbox"]) == 4 else 0
        too_tall = avg_h > 0 and bbox_h > avg_h * 1.8
        too_wordy = len(words) > 4 or len(ann["target_text"]) > 30
        if too_tall or too_wordy:
            ann["type"] = "underline"

    # Drop duplicate bboxes (GPT often collapses several phrases onto the same
    # heading bbox — keep only the first occurrence per bbox).
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for ann in clean:
        key = tuple(ann["bbox"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ann)
    clean = deduped

    clean.sort(key=lambda a: a["start_time"])

    print(f"[AI] {len(clean)} annotations generated for chunk {chunk_number}")
    return clean


# ── Phrase → bbox locator ────────────────────────────────────────────────────

_NORM_RE = re.compile(r"[^a-z0-9]+")

def _norm(s: str) -> str:
    return _NORM_RE.sub("", s.lower())


def _locate_phrase_bbox(phrase: str, ocr_words: list[dict]) -> list[int] | None:
    """
    Find the contiguous OCR word run whose joined text best matches `phrase`
    and return the union bbox [x, y, w, h]. Returns None if no decent match.
    """
    if not phrase or not ocr_words:
        return None
    target = _norm(phrase)
    if not target:
        return None

    norm_words = [_norm(w.get("text", "")) for w in ocr_words]
    n = len(ocr_words)
    best: tuple[float, int, int] | None = None

    for i in range(n):
        joined = ""
        for j in range(i, min(n, i + 40)):
            joined += norm_words[j]
            if not joined:
                continue
            if target in joined:
                overshoot = len(joined) - len(target)
                score = 1.0 - (overshoot / max(len(target), 1)) * 0.2
                if best is None or score > best[0]:
                    best = (score, i, j + 1)
                break
            if joined in target:
                cov = len(joined) / len(target)
                if cov >= 0.6:
                    score = cov * 0.9
                    if best is None or score > best[0]:
                        best = (score, i, j + 1)
            if len(joined) > len(target) * 2:
                break

    if best is None:
        return None
    _, i, j = best

    xs, ys, x2s, y2s = [], [], [], []
    for k in range(i, j):
        w = ocr_words[k]
        x = int(w.get("x", 0)); y = int(w.get("y", 0))
        ww = int(w.get("w", 0)); hh = int(w.get("h", 0))
        xs.append(x); ys.append(y); x2s.append(x + ww); y2s.append(y + hh)
    if not xs:
        return None
    return [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]

