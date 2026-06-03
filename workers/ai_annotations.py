"""
workers/ai_annotations.py — GPT-4o generates annotation events.

Each annotation:
  {
    "type":        "underline" | "circle",
    "start_time":  float   (seconds, synced to when the word is spoken),
    "target_text": str     (word/phrase from OCR),
    "bbox":        [x, y, w, h]  (OCR pixel coordinates),
  }
"""

import json
import re
from typing import Any

from openai import OpenAI
from lib.config import config

_openai = OpenAI(api_key=config.OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are a video annotation assistant for English educational slides.

Given:
- OCR words with pixel bounding boxes from a slide image
- Word-level audio timestamps (when each word is spoken)
- The script text for this segment

Your job: pick the 3–8 MOST IMPORTANT words or phrases and decide how to annotate them.

ANNOTATION TYPES:
  underline → use for ~60% of annotations (phrases or sub-headings)
  circle    → use for ~40% of annotations (key terms or single words)

RULES:
1. Generate 3–8 annotations. Quality over quantity.
2. ONLY use "underline" and "circle" types.
3. Distribution: Aim for 60% underlines and 40% circles across the chunk.
4. start_time MUST match when that word is spoken (use the timestamps).
5. bbox MUST come from the OCR data — never invent coordinates.
6. target_text must match OCR text exactly.

OUTPUT FORMAT — return ONLY valid JSON (no markdown, no extra text):
{"annotations": [{"type": "...", "start_time": 0.0, "target_text": "...", "bbox": [x, y, w, h]}, ...]}"""


def generate_annotations(
    ocr_words: list[dict],
    ts_words: list[dict],
    chunk_text: str,
    chunk_number: int,
) -> list[dict[str, Any]]:
    """
    Call GPT-4o to generate annotation events.

    Returns list of annotation dicts.
    """
    print(f"[AI] generating annotations for chunk {chunk_number}")

    ocr_lines = "\n".join(
        f'  "{w["text"]}" → bbox:[{w["x"]},{w["y"]},{w["w"]},{w["h"]}]'
        for w in ocr_words
    )
    ts_lines = "\n".join(
        f'  {w["start"]:.2f}s–{w["end"]:.2f}s  "{w["word"]}"'
        for w in ts_words
    )
    total_dur = ts_words[-1]["end"] if ts_words else 10.0

    user_prompt = f"""Chunk {chunk_number} — total audio duration: {total_dur:.2f}s

=== SCRIPT TEXT ===
{chunk_text}

=== OCR WORDS (text + bounding boxes) ===
{ocr_lines}

=== WORD TIMESTAMPS ===
{ts_lines}

Generate 3–8 annotations. Return JSON only."""

    response = _openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1500,
        temperature=0.2,
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

    # Sanitise
    allowed_types = {"underline", "circle"}
    clean = []
    for ann in annotations:
        t = ann.get("type")
        if t not in allowed_types:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        clean.append({
            "type":        t,
            "start_time":  max(0.0, float(ann.get("start_time") or 0)),
            "target_text": str(ann.get("target_text") or ""),
            "bbox":        [int(v) for v in bbox],
        })

    print(f"[AI] {len(clean)} annotations generated")
    return clean
