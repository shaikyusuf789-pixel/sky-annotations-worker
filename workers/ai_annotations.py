# Updated: 2026-06-04 - Programmatic start_time lookup (GPT was guessing wrong)
"""
workers/ai_annotations.py — GPT-4o generates annotation events.

Each annotation:
  {
    "type":        "underline" | "circle",
    "start_time":  float   (seconds, resolved programmatically from ts_words),
    "target_text": str     (word/phrase from OCR),
    "bbox":        [x, y, w, h]  (OCR pixel coordinates),
  }

GPT is NOT asked for start_time — it was consistently wrong.
start_time is resolved here by fuzzy-matching target_text against ts_words.
"""

import json
import re
from typing import Any

from openai import OpenAI
from lib.config import config

def _get_openai_client():
    return OpenAI(api_key=config.OPENAI_API_KEY)


_SYSTEM_PROMPT = """You are a video annotation assistant for English educational slides.

CRITICAL: ONLY TWO ANNOTATION TYPES ARE ALLOWED:
1. "underline" → Use for approximately 60% of annotations.
2. "circle"    → Use for approximately 30% of annotations.
3. "pen"       → Use for approximately 20% of annotations. A white ink pen stroke drawn through the middle of the text.

STRICT NEGATIVE CONSTRAINTS:
- NEVER use "arrow". It is strictly forbidden.
- NEVER use "box", "double_underline", or any other type.
- ONLY "underline", "circle", and "pen" are valid.

RULES:
1. Generate 3–8 annotations total. Do not exceed 8.
2. Distribution: Aim for 50% underline, 30% circle, 20% pen.
3. Do NOT annotate the main title/heading at the very start of the clip. Focus on core content.
4. bbox MUST come from the OCR data provided.
5. target_text must be the English word or phrase visible on the slide (from OCR).

DO NOT include start_time — it will be computed automatically.

OUTPUT FORMAT — return ONLY valid JSON:
{"annotations": [{"type": "underline", "target_text": "sample", "bbox": [0,0,10,10]}, {"type": "circle", "target_text": "test", "bbox": [20,20,5,5]}]}"""


_ANNOTATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "slide_annotations",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "annotations": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["underline", "circle", "pen"]},
                            "target_text": {"type": "string"},
                            "bbox": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["type", "target_text", "bbox"],
                    },
                }
            },
            "required": ["annotations"],
        },
    },
}


def _find_start_time(target_text: str, ts_words: list[dict]) -> float:
    """
    Fuzzy-match target_text (English phrase) against the word timestamps list.
    Tries each word in target_text against each ts entry (case-insensitive prefix match).
    Returns the earliest matching timestamp, or 0.5s if nothing matches.
    """
    if not ts_words:
        return 0.5

    target_words = [w.lower().strip(".,!?;:\"'") for w in target_text.split()]

    best_time = None
    for tw in target_words:
        if len(tw) < 2:
            continue
        for entry in ts_words:
            ew = (entry.get("word") or "").lower().strip(".,!?;:\"'")
            # prefix match in either direction (handles "joule"/"Joule", "pH"/"ph", etc.)
            if ew.startswith(tw) or tw.startswith(ew):
                t = float(entry.get("start") or 0)
                if best_time is None or t < best_time:
                    best_time = t

    if best_time is None:
        # fallback: spread evenly — use annotation index offset from audio midpoint
        total = ts_words[-1]["end"] if ts_words else 10.0
        best_time = total * 0.1

    return best_time


def generate_annotations(
    ocr_words: list[dict],
    ts_words: list[dict],
    chunk_text: str,
    chunk_number: int,
) -> list[dict[str, Any]]:
    if not config.OPENAI_API_KEY or len(config.OPENAI_API_KEY) < 20:
        raise ValueError(f"Invalid OPENAI_API_KEY format (len={len(config.OPENAI_API_KEY)})")

    print(f"[AI] generating annotations for chunk {chunk_number}")

    ocr_lines = "\n".join(
        f'  "{w["text"]}" → bbox:[{w["x"]},{w["y"]},{w["w"]},{w["h"]}]'
        for w in ocr_words
    )
    total_dur = ts_words[-1]["end"] if ts_words else 10.0

    user_prompt = f"""Chunk {chunk_number} — total audio duration: {total_dur:.2f}s

=== SCRIPT TEXT ===
{chunk_text}

=== OCR WORDS on slide (text + bounding boxes) ===
{ocr_lines}

Pick 3–8 key terms from the OCR words above that are important for the student.
Return ONLY type, target_text, and bbox — no start_time needed."""

    try:
        response = _get_openai_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=_ANNOTATION_RESPONSE_FORMAT,
            max_tokens=1000,
            temperature=0.2,
        )
    except Exception as e:
        print(f"[AI] OpenAI error: {e}")
        raise RuntimeError(f"OpenAI error: {str(e)}")

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group(0)) if m else {}

    annotations = parsed.get("annotations", [])
    if not isinstance(annotations, list):
        annotations = []

    clean = []
    for ann in annotations:
        t = ann.get("type")
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        target = str(ann.get("target_text") or "")
        start_time = _find_start_time(target, ts_words)
        clean.append({
            "type":        t,
            "start_time":  start_time,
            "target_text": target,
            "bbox":        [int(v) for v in bbox],
        })
        print(f"[AI]   {t:10s} '{target}' → start={start_time:.2f}s  bbox={bbox}")

    print(f"[AI] {len(clean)} annotations generated for chunk {chunk_number}")
    return clean
