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
from difflib import SequenceMatcher
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

ANNOTATION TYPES (mix them so the slide feels alive, not just underlined):
  underline → SHORT phrases only (2–5 words). Use sparingly — a real tutor
              doesn't underline full sentences. Max 4 per chunk.
  circle    → PREFERRED for spotlighting a single keyword, number, name,
              or 1–3 word phrase. Use this often.
              HARD RULE: target_text MUST be ≤ 4 words AND ≤ 30 characters
              AND fit on a single visual line. Never circle a paragraph.
  box       → frame a statistic, formula, or grouped callout (only for
              already-grouped blocks, not individual bullets).
  arrow     → point AT a bullet, name, number, or callout when the narrator
              draws attention to it. Use arrows often — they add motion and
              break the monotony of underlines/circles.

  DO NOT use "double_underline" — it is deprecated. Use "underline" instead.

  TARGET MIX per chunk (rough guideline, 15 annotations):
    ~4 circles · ~4 underlines · ~3 arrows · ~2 boxes

TIMING — CRITICAL CHANGE
  • You DO NOT pick start_time anymore. Python will compute it from the
    real word timestamps. Your job is to tell Python WHICH spoken words
    correspond to each annotation, via `script_phrase`.
  • `script_phrase` MUST be a short consecutive run of words (2–6 words
    is ideal, 1 word OK for unique terms) copied VERBATIM from the
    WORD-LEVEL TIMESTAMPS list (the Latin/transliterated column on the
    right of the arrow). Example: if the timestamps contain
       34.66s → 34.90s  "yild"
       34.96s → 35.22s  "tapiks"
    and the slide says "High Yield Topics", then for that annotation:
       "script_phrase": "yild tapiks"
  • Pick the script_phrase that BEST identifies WHEN the narrator is
    talking about this concept. If a phrase repeats in the timestamps
    (e.g. "SSC" appears twice), pick the occurrence that matches the
    chronological order of the slide narration.
  • Still keep `start_time` in the JSON as a fallback hint (your best
    guess from the timestamps), but Python will overwrite it.

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
  • Annotations MUST be in CHRONOLOGICAL ORDER of their script_phrase
    appearance in the timestamps.
  • Do not annotate the same OCR text more than once unless the script
    references it at clearly distinct times.

OUTPUT FORMAT — return ONLY valid JSON (no markdown fences, no commentary):
{"annotations": [
  {"type": "circle", "start_time": 6.40, "script_phrase": "SSC CGL", "target_text": "SSC CGL 2026", "bbox": [262,343,640,81]},
  {"type": "arrow",  "start_time": 34.66, "script_phrase": "hai yild tapiks", "target_text": "high yield topics", "bbox": [356,971,280,34]},
  ...
]}"""


def _ocr_lines(ocr_words: list[dict]) -> str:
    out = []
    for i, w in enumerate(ocr_words):
        text = w.get("text", "")
        out.append(
            f'  [{i:03d}] "{text}"  bbox=[{w.get("x",0)},{w.get("y",0)},{w.get("w",0)},{w.get("h",0)}]'
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
    speech_start = float(ts_words[0].get("start", 0.0)) if ts_words else 0.0
    speech_end   = float(ts_words[-1].get("end",   total_dur)) if ts_words else float(total_dur)
    speech_window = max(0.1, speech_end - speech_start)

    user_text = f"""CHUNK {chunk_number}  —  audio duration: {float(total_dur):.2f}s

=== SPEECH WINDOW (CRITICAL) ===
First spoken word starts at: {speech_start:.2f}s
Last spoken word ends at:    {speech_end:.2f}s
Total speech window:         {speech_window:.2f}s
The audio has ~{speech_start:.1f}s of intro/silence before the narrator begins.
HARD RULES:
  • NEVER place an annotation with start_time < {speech_start:.2f}s.
  • All annotations MUST fall within [{speech_start:.2f}s, {speech_end:.2f}s].
  • Space annotations at LEAST 4–5 seconds apart across the speech window.
  • Match each start_time to when that specific word is actually spoken,
    using the WORD-LEVEL TIMESTAMPS below as ground truth.

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
2. Read the script and identify EVERY meaningful concept the narrator says.
3. For each concept, find the matching word/phrase/line on the slide
   (semantic match — slide wording is paraphrased from the script).
4. STRONGLY PREFER short keyword targets: single words, numbers, names, or
   2–4 word phrases. CIRCLE them when possible. Cap full-line underlines at
   MAX 2 per chunk.
5. For EACH annotation, copy a `script_phrase` of 2–6 consecutive Latin
   words from the WORD-LEVEL TIMESTAMPS that the narrator says when this
   concept is mentioned. Python will look it up and assign the real
   start_time. If a phrase repeats (e.g. "SSC" twice), pick the occurrence
   matching chronological order with your other annotations.
6. target_text MUST be the EXACT OCR text (copy character-for-character).
   For multi-word targets, concatenate consecutive OCR words with single
   spaces in the order they appear in the OCR dump.
7. Vary types. DO NOT use double_underline. Lean on `circle` for keywords,
   short `underline` for phrases, plenty of `arrow` for callouts, and the
   occasional `box` for grouped callouts.
8. Return 12–20 annotations, ORDERED by appearance of their script_phrase
   in the timestamps.

Return ONLY the JSON object."""


    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if slide_image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": slide_image_url, "detail": "auto"},
        })

    import time as _time
    last_err: Exception | None = None
    response = None
    for attempt in range(6):
        try:
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
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            is_429 = "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()
            if not is_429 or attempt == 5:
                raise
            wait_s = 12.0
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                try: wait_s = float(m.group(1)) + 2.0
                except Exception: pass
            wait_s = min(60.0, max(wait_s, 5.0 * (attempt + 1)))
            print(f"[AI] 429 rate-limited (attempt {attempt+1}/6) — sleeping {wait_s:.1f}s")
            _time.sleep(wait_s)
    if response is None:
        raise last_err or RuntimeError("OpenAI call failed")

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group(0)) if m else {}

    annotations = parsed.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    # Sanitise + RECOMPUTE bbox from OCR + RECOMPUTE start_time from real
    # ElevenLabs timestamps using the model-supplied `script_phrase`.
    # GPT can't be trusted with numeric grounding — it hallucinates timestamps
    # 10–20 seconds off the actual spoken word. We use Python to look up the
    # phrase in ts_words and copy the real start time.
    allowed_types = {"underline", "double_underline", "circle", "box", "arrow"}
    clean: list[dict[str, Any]] = []
    last_assigned_start = -1.0  # enforce chronological ordering for repeats

    for ann in annotations:
        t = ann.get("type")
        if t == "double_underline":
            t = "underline"
        if t not in allowed_types:
            continue
        target = str(ann.get("target_text") or "").strip()
        if not target:
            continue

        # ── bbox: locate via OCR (never trust GPT's bbox).
        located = _locate_phrase_bbox(target, ocr_words)
        if located is None:
            print(f"[AI] DROP: target_text '{target}' not found in OCR")
            continue

        # ── start_time: look up script_phrase in real word timestamps.
        script_phrase = str(ann.get("script_phrase") or "").strip()
        gpt_start = None
        try:
            gpt_start = float(ann.get("start_time") or 0)
        except (TypeError, ValueError):
            gpt_start = None

        real_start = _locate_phrase_start_time(
            script_phrase, ts_words, min_start=last_assigned_start
        )
        target_start = _locate_target_start_time(
            target, ts_words, min_start=last_assigned_start
        )
        if target_start is not None and (
            real_start is None or abs(float(target_start) - float(real_start)) > 1.25
        ):
            if real_start is not None:
                print(f"[AI] REPAIR: target '{target}' timing {real_start:.2f}s → {target_start:.2f}s")
            real_start = target_start
        if real_start is None and gpt_start is not None:
            # Fallback: use GPT's number but warn loudly in logs.
            print(f"[AI] WARN: script_phrase '{script_phrase}' not found in "
                  f"ts_words for target '{target}' — falling back to GPT "
                  f"start_time {gpt_start:.2f}s")
            real_start = gpt_start
        if real_start is None:
            print(f"[AI] DROP: no script_phrase and no start_time for '{target}'")
            continue

        last_assigned_start = max(last_assigned_start, real_start)

        clean.append({
            "type":        t,
            "start_time":  max(0.0, float(real_start)),
            "target_text": target,
            "bbox":        located,
        })

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

    # Safety net: clamp into speech window + de-cluster ONLY exact overlaps.
    # The old +4s cascade was destroying timing on dense slides — a single
    # tight cluster early on shoved every later annotation 4s, 8s, 12s late,
    # which is why "B12" was firing 20s+ after the narrator said it.
    # Now: trust GPT's timestamps (they came from real word timestamps), only
    # nudge true overlaps by 0.4s and DO NOT propagate the nudge further.
    if clean and ts_words:
        sw_start = float(ts_words[0].get("start", 0.0))
        sw_end   = float(ts_words[-1].get("end",   total_dur))
        spaced: list[dict[str, Any]] = []
        used: list[float] = []
        for ann in clean:
            t = max(sw_start, min(sw_end, float(ann["start_time"])))
            # Soft de-cluster: if within 0.4s of an already-placed annotation,
            # nudge by 0.4s. Single-pass — never cascade.
            for u in used:
                if abs(t - u) < 0.4:
                    t = min(sw_end, u + 0.4)
                    break
            ann["start_time"] = round(t, 3)
            spaced.append(ann)
            used.append(t)
        clean = spaced
        clean.sort(key=lambda a: a["start_time"])

    print(f"[AI] {len(clean)} annotations generated for chunk {chunk_number} "
          f"(window {ts_words[0].get('start',0) if ts_words else 0:.2f}s → "
          f"{ts_words[-1].get('end',0) if ts_words else 0:.2f}s)")
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


# ── Phrase → start_time locator (deterministic, no LLM) ──────────────────────

def _locate_phrase_start_time(
    phrase: str,
    ts_words: list[dict],
    min_start: float = -1.0,
) -> float | None:
    """
    Find a consecutive run of ts_words whose joined normalized text contains
    (or is contained by) the normalized `phrase`, and return the .start of
    the first matched word. Prefers matches with start >= min_start to keep
    annotations chronologically advancing when the same phrase repeats.

    Returns None if no acceptable match exists.
    """
    if not phrase or not ts_words:
        return None
    target = _norm(phrase)
    if not target:
        return None

    # Normalize each ts_word.
    norm = []
    for w in ts_words:
        s = w.get("word") or w.get("text") or ""
        norm.append(_norm(s))
    n = len(ts_words)

    candidates: list[tuple[float, int, float]] = []  # (score, i, start)

    for i in range(n):
        joined = ""
        for j in range(i, min(n, i + 12)):  # max 12-word window
            joined += norm[j]
            if not joined:
                continue
            score = None
            if target in joined:
                overshoot = len(joined) - len(target)
                score = 1.0 - (overshoot / max(len(target), 1)) * 0.2
            elif joined in target and len(joined) / len(target) >= 0.6:
                score = (len(joined) / len(target)) * 0.85
            if score is not None:
                try:
                    start = float(ts_words[i].get("start", 0.0))
                except (TypeError, ValueError):
                    start = 0.0
                candidates.append((score, i, start))
                if target in joined:
                    break
            if len(joined) > len(target) * 2.5:
                break

    if not candidates:
        return None

    # Prefer the earliest match whose start >= min_start (chronological).
    forward = [c for c in candidates if c[2] >= min_start - 0.01]
    pool = forward if forward else candidates
    # Among the pool, take the best score; tie-break by smallest start.
    pool.sort(key=lambda c: (-c[0], c[2]))
    return pool[0][2]


def _locate_target_start_time(
    target_text: str,
    ts_words: list[dict],
    min_start: float = -1.0,
) -> float | None:
    """Map visible slide text back to spoken timestamp words.

    GPT sometimes supplies a weak `script_phrase` for a good visual target
    (for example target_text="acceleration" but script_phrase="adugutaru").
    This deterministic repair looks for the target itself, plus common
    Telugu-English phonetic spellings produced by forced alignment.
    """
    if not target_text or not ts_words:
        return None

    variants = _target_variants(target_text)
    if not variants:
        return None

    norm_words = [_norm(w.get("word") or w.get("text") or "") for w in ts_words]
    candidates: list[tuple[float, float]] = []  # (score, start)
    for i in range(len(ts_words)):
        joined = ""
        for j in range(i, min(len(ts_words), i + 8)):
            joined += norm_words[j]
            if not joined:
                continue
            for variant in variants:
                if not variant:
                    continue
                score = 0.0
                if variant in joined or joined in variant:
                    score = min(len(variant), len(joined)) / max(len(variant), len(joined), 1)
                    score = max(score, 0.88)
                else:
                    score = SequenceMatcher(None, variant, joined).ratio()
                if score >= 0.78:
                    try:
                        start = float(ts_words[i].get("start", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        start = 0.0
                    if start >= min_start - 0.01:
                        candidates.append((score, start))
            if len(joined) > 80:
                break

    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][1]


def _target_variants(target_text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", target_text.lower())
    variants: set[str] = set()
    if words:
        variants.add("".join(words))
        for w in words:
            if len(w) >= 3:
                variants.add(w)

    phrase_map = {
        "skyacademy": ["skyacademy"],
        "ssccgl2026": ["ssccgl2026", "ssccgl"],
        "highyield": ["highyield", "haiyild", "yild"],
        "topics": ["topics", "tapiks", "tapik"],
        "physics": ["physics", "phijiks", "phijik", "fijiks"],
        "motion": ["motion", "mosn"],
        "force": ["force", "phors", "fors"],
        "mass": ["mass", "mas"],
        "important": ["important", "impartemt", "impartment"],
        "acceleration": ["acceleration", "yaksilresn", "aksilresn", "accilresn"],
        "fma": ["fma"],
    }
    compact = "".join(words)
    for key, vals in phrase_map.items():
        if key in compact or compact in key:
            variants.update(vals)
    for w in words:
        variants.update(phrase_map.get(w, []))

    return sorted({_norm(v) for v in variants if _norm(v)}, key=len, reverse=True)
