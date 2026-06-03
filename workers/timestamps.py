"""
workers/timestamps.py — OpenAI Whisper word-level timestamps.

Returns a list of:
  { "word": str, "start": float, "end": float }
"""

from openai import OpenAI
from lib.config import config

_openai = OpenAI(api_key=config.OPENAI_API_KEY)


def get_timestamps(audio_path: str) -> tuple[list[dict], float]:
    """
    Transcribe an MP3 with Whisper and return word-level timestamps.

    Returns
    -------
    (words, duration)
      words    — list of {"word", "start", "end"}
      duration — total audio duration in seconds
    """
    print(f"[TS] transcribing {audio_path}")

    with open(audio_path, "rb") as f:
        transcription = _openai.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            language="en",
        )

    raw_words    = getattr(transcription, "words",    None) or []
    raw_segments = getattr(transcription, "segments", None) or []

    words = [
        {
            "word":  (w.word if hasattr(w, "word") else w.get("word", "")).strip(),
            "start": round(float(w.start if hasattr(w, "start") else w.get("start", 0)), 3),
            "end":   round(float(w.end   if hasattr(w, "end")   else w.get("end",   0)), 3),
        }
        for w in raw_words
        if (w.word if hasattr(w, "word") else w.get("word", "")).strip()
    ]

    # Compute total duration from segments (more reliable than last word end)
    if raw_segments:
        last = raw_segments[-1]
        duration = float(last.end if hasattr(last, "end") else last.get("end", 0))
    elif words:
        duration = words[-1]["end"]
    else:
        duration = 0.0

    print(f"[TS] {len(words)} words, duration={duration:.2f}s")
    return words, duration
