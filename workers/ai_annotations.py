# Updated: 2026-06-03 17:35
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

CRITICAL: ONLY TWO ANNOTATION TYPES ARE ALLOWED:
1. "underline" → Use for approximately 60% of annotations.
2. "circle"    → Use for approximately 40% of annotations.

STRICT NEGATIVE CONSTRAINTS:
- NEVER use "arrow". It is strictly forbidden.
- NEVER use "box", "double_underline", or any other type.
- ONLY "underline" and "circle" are valid.

RULES:
1. Generate 3–8 annotations total. Do not exceed 8.
2. Distribution: Aim for exactly 60% underlines and 40% circles.
3. Do NOT annotate the main title/heading at the very start of the clip (e.g., "SSC CGL 2026"). Focus on the core content.
4. start_time MUST match the exact second when the first word of the target_text is spoken.
5. bbox MUST come from the OCR data.
6. target_text must match OCR text exactly.

OUTPUT FORMAT — return ONLY valid JSON:
{"annotations": [{"type": "underline", "start_time": 0.0, "target_text": "sample", "bbox": [0,0,10,10]}, {"type": "circle", "start_time": 1.0, "target_text": "test", "bbox": [20,20,5,5]}]}"""


def _rebalance_annotation_types(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hard guarantee: only underline/circle, roughly 60/40 split."""
    if not annotations:
        return annotations

    total = len(annotations)
    target_circles = round(total * 0.4)

    normalised = []
    for ann in annotations:
        ann_type = ann.get("type")
        if ann_type not in {"underline", "circle"}:
            ann_type = "underline"
        normalised.append({**ann, "type": ann_type})

    circle_count = sum(1 for ann in normalised if ann["type"] == "circle")
    if circle_count < target_circles:
        underline_indexes = [
            idx for idx, ann in enumerate(normalised)
            if ann["type"] == "underline"
        ]
        underline_indexes.sort(
            key=lambda idx: (
                len(str(normalised[idx].get("target_text", "")).split()),
                idx,
            )
        )
        for idx in underline_indexes[:target_circles - circle_count]:
            normalised[idx]["type"] = "circle"
    elif circle_count > target_circles:
        circle_indexes = [
            idx for idx, ann in enumerate(normalised)
            if ann["type"] == "circle"
        ]
        circle_indexes.sort(
            key=lambda idx: (
                -len(str(normalised[idx].get("target_text", "")).split()),
                idx,
            )
        )
        for idx in circle_indexes[:circle_count - target_circles]:
            normalised[idx]["type"] = "underline"

    return normalised


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
            t = "circle" if len(clean) % 5 in (2, 4) else "underline"
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        clean.append({
            "type":        t,
            "start_time":  max(0.0, float(ann.get("start_time") or 0)),
            "target_text": str(ann.get("target_text") or ""),
            "bbox":        [int(v) for v in bbox],
        })

    clean = _rebalance_annotation_types(clean)

    print(f"[AI] {len(clean)} annotations generated")
    return clean
