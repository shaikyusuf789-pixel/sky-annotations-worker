"""
workers/ocr.py — Tesseract OCR for slide PNG images.

Returns a list of word dicts:
  { "text": str, "x": int, "y": int, "w": int, "h": int, "conf": float }
"""

import json
from typing import Any

import pytesseract
from PIL import Image


def run_ocr(image_path: str) -> list[dict[str, Any]]:
    """
    Run Tesseract on a slide PNG and return word bounding boxes.

    Parameters
    ----------
    image_path : local path to the PNG file

    Returns
    -------
    list of {"text", "x", "y", "w", "h", "conf"}  — only words with conf > 30
    """
    print(f"[OCR] running Tesseract on {image_path}")
    img = Image.open(image_path)

    data = pytesseract.image_to_data(
        img,
        lang="eng",
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    words: list[dict[str, Any]] = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = float(data["conf"][i])
        if not text or conf < 30:
            continue
        words.append({
            "text": text,
            "x":    int(data["left"][i]),
            "y":    int(data["top"][i]),
            "w":    int(data["width"][i]),
            "h":    int(data["height"][i]),
            "conf": round(conf, 1),
        })

    print(f"[OCR] found {len(words)} words with conf > 30")
    return words
