"""
workers/timestamps.py — OpenAI Whisper word-level timestamps.

Strategy for Indian-language audio (Telugu / Hindi / etc.):
  - Let Whisper auto-detect the spoken language (do NOT force language="en";
    forcing English produces hallucinated Devanagari output for Telugu audio).
  - After transcription, transliterate each word to Latin script using
    Unidecode so downstream AI prompts get a stable romanized form.
    This is transliteration (script conversion), NOT translation —
    meaning is preserved, only the script changes.

Returns a list of:
  { "word": str, "start": float, "end": float }
"""

from openai import OpenAI
from unidecode import unidecode

from lib.config import config

_openai = OpenAI(api_key=config.OPENAI_API_KEY)


def _to_latin(text: str) -> str:
    """Transliterate any script to Latin (ASCII). Keeps ASCII as-is."""
    if not text:
        return ""
    # Fast path: already ASCII
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        pass
    return unidecode(text).strip()


def _group_into_phrases(
    words: list[dict],
    min_dur: float = 1.5,
    max_dur: float = 4.0,
    max_words: int = 6,
    pause_break: float = 0.35,
) -> list[dict]:
    """Group micro word-timings into moderate phrase chunks (1.5–4s, ~3-6 words).

    Breaks on: punctuation-ending word, long inter-word pause (>pause_break),
    max word count, or max duration reached. Gives the AI better semantic
    anchors than raw word-by-word stamps.
    """
    if not words:
        return []
    phrases: list[dict] = []
    cur_words: list[dict] = []
    cur_start = words[0]["start"]
    prev_end = words[0]["start"]

    def _flush():
        if not cur_words:
            return
        text = " ".join(w["word"] for w in cur_words)
        orig = " ".join(w.get("original") or w["word"] for w in cur_words)
        phrases.append({
            "word":     text,
            "original": orig,
            "start":    round(cur_words[0]["start"], 3),
            "end":      round(cur_words[-1]["end"], 3),
        })

    for w in words:
        gap = w["start"] - prev_end
        cur_dur = (cur_words[-1]["end"] - cur_start) if cur_words else 0.0
        ends_punct = bool(cur_words) and cur_words[-1]["word"][-1:] in ".!?,;:।"
        should_break = (
            cur_words
            and (
                (cur_dur >= min_dur and (gap > pause_break or ends_punct))
                or len(cur_words) >= max_words
                or cur_dur >= max_dur
            )
        )
        if should_break:
            _flush()
            cur_words = []
            cur_start = w["start"]
        if not cur_words:
            cur_start = w["start"]
        cur_words.append(w)
        prev_end = w["end"]
    _flush()
    return phrases


def get_timestamps(audio_path: str) -> tuple[list[dict], float]:
    """
    Transcribe an MP3 with Whisper and return phrase-level timestamps
    (moderate 1.5–4s chunks, ~3-6 words each), with each phrase
    transliterated to Latin script. This gives downstream AI better
    semantic anchors than raw word-by-word timings.
    """
    print(f"[TS] transcribing {audio_path} (language=te Telugu, transliterate to Latin)")

    with open(audio_path, "rb") as f:
        transcription = _openai.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            language="te",  # Force Telugu — auto-detect occasionally mis-picks Tamil/Hindi
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    raw_words    = getattr(transcription, "words",    None) or []
    raw_segments = getattr(transcription, "segments", None) or []

    words: list[dict] = []
    for w in raw_words:
        original = (w.word if hasattr(w, "word") else w.get("word", "")).strip()
        if not original:
            continue
        latin = _to_latin(original)
        if not latin:
            continue
        words.append({
            "word":     latin,
            "original": original,
            "start":    round(float(w.start if hasattr(w, "start") else w.get("start", 0)), 3),
            "end":      round(float(w.end   if hasattr(w, "end")   else w.get("end",   0)), 3),
        })

    # Group micro word-stamps into phrase chunks for the AI / annotation layer
    phrases = _group_into_phrases(words)

    if raw_segments:
        last = raw_segments[-1]
        duration = float(last.end if hasattr(last, "end") else last.get("end", 0))
    elif words:
        duration = words[-1]["end"]
    else:
        duration = 0.0

    detected = getattr(transcription, "language", None)
    print(f"[TS] detected={detected!r}, {len(words)} words → {len(phrases)} phrases, duration={duration:.2f}s")
    return phrases, duration
