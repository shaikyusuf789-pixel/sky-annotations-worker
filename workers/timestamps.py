"""
workers/timestamps.py — OpenAI word-level timestamps for Telugu+English audio.

Strategy:
  - Primary model: gpt-4o-transcribe (better word-boundary detection for
    Indic languages mixed with English than whisper-1).
  - Fallback model: whisper-1 (only if gpt-4o-transcribe fails).
  - We FORCE language="en" so Telugu words come back romanized
    (e.g. "neenu mii mentor") instead of Telugu script. This is critical:
    downstream AI matching compares spoken words against English slide OCR,
    so we want phonetic Latin output, not native script.
  - We pass a domain prompt hinting it's a Telugu educational explainer
    with English technical terms — this nudges the model to keep proper
    word boundaries instead of merging multiple Telugu words into one token.

Coverage repair:
  After transcription, if word-level coverage spans < 70% of the audio
  duration (Whisper sometimes collapses 5 Telugu words into 1 entry that
  spans 11s), we fall back to segment-level data and synthesize per-word
  timings by evenly splitting each segment across its words. We replace
  the word list with the synthetic timings ONLY if they cover more of the
  audio than the original.

Returns: (words, duration)
  words = [{ "word": str, "original": str, "start": float, "end": float }, ...]
"""

from openai import OpenAI
from unidecode import unidecode

from lib.config import config

_openai = OpenAI(api_key=config.OPENAI_API_KEY)

_TRANSCRIBE_PROMPT = (
    "This is a Telugu educational explainer video for competitive exam students. "
    "The narrator speaks Telugu mixed with English technical terms (e.g. 'mentor', "
    "'science', 'physics', 'chemistry', 'biology', 'SSC', 'CGL', 'exam'). "
    "Transcribe each Telugu word phonetically in English letters (romanization), "
    "preserving natural word boundaries. Do not merge multiple words into one token."
)


def _to_latin(text: str) -> str:
    if not text:
        return ""
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        pass
    return unidecode(text).strip()


def _transcribe(audio_path: str, model: str) -> object:
    print(f"[TS] trying model={model}")
    with open(audio_path, "rb") as f:
        return _openai.audio.transcriptions.create(
            file=f,
            model=model,
            language="en",  # force romanized Latin output for Telugu
            prompt=_TRANSCRIBE_PROMPT,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )


def _extract_words(raw_words) -> list[dict]:
    out: list[dict] = []
    for w in raw_words or []:
        original = (w.word if hasattr(w, "word") else w.get("word", "")).strip()
        if not original:
            continue
        latin = _to_latin(original)
        if not latin:
            continue
        out.append({
            "word":     latin,
            "original": original,
            "start":    round(float(w.start if hasattr(w, "start") else w.get("start", 0)), 3),
            "end":      round(float(w.end   if hasattr(w, "end")   else w.get("end",   0)), 3),
        })
    return out


def _split_multiword_entries(entries: list[dict]) -> list[dict]:
    """OpenAI can return phrase-sized timestamp entries for Telugu audio.
    Split any multi-word entry into true per-word timings so downstream
    annotation matching has one timestamp per spoken token.
    """
    out: list[dict] = []
    for idx, entry in enumerate(entries):
        text = str(entry.get("word") or "").strip()
        original = str(entry.get("original") or text).strip()
        toks = [t for t in text.split() if t]
        orig_toks = [t for t in original.split() if t]
        if not toks:
            continue

        s = float(entry.get("start", 0.0) or 0.0)
        e = float(entry.get("end", 0.0) or 0.0)
        if e <= s:
            next_start = None
            for nxt in entries[idx + 1:]:
                ns = float(nxt.get("start", 0.0) or 0.0)
                if ns > s:
                    next_start = ns
                    break
            e = next_start if next_start and next_start > s else s + max(0.25, 0.32 * len(toks))

        step = max(0.08, (e - s) / len(toks))
        for i, tok in enumerate(toks):
            ws = s + i * step
            we = min(e, ws + step)
            out.append({
                "word":     tok,
                "original": orig_toks[i] if i < len(orig_toks) else tok,
                "start":    round(ws, 3),
                "end":      round(max(we, ws + 0.08), 3),
            })
    return out


def _coverage(words: list[dict], duration: float) -> float:
    if not words or duration <= 0:
        return 0.0
    span = words[-1]["end"] - words[0]["start"]
    return max(0.0, min(1.0, span / duration))


def _synthesize_from_segments(raw_segments, duration: float) -> list[dict]:
    """Evenly split each segment's time window across its words."""
    out: list[dict] = []
    for seg in raw_segments or []:
        text = (seg.text if hasattr(seg, "text") else seg.get("text", "")) or ""
        s = float(seg.start if hasattr(seg, "start") else seg.get("start", 0))
        e = float(seg.end   if hasattr(seg, "end")   else seg.get("end",   0))
        if e <= s:
            continue
        toks = [t for t in text.strip().split() if t]
        if not toks:
            continue
        step = (e - s) / len(toks)
        for i, tok in enumerate(toks):
            ws = s + i * step
            we = ws + step
            latin = _to_latin(tok)
            if not latin:
                continue
            out.append({
                "word":     latin,
                "original": tok,
                "start":    round(ws, 3),
                "end":      round(we, 3),
            })
    return out


def _group_into_phrases(
    words: list[dict],
    min_dur: float = 1.5,
    max_dur: float = 4.0,
    max_words: int = 6,
    pause_break: float = 0.35,
) -> list[dict]:
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


def _elevenlabs_forced_alignment(audio_path: str, script_text: str) -> tuple[list[dict], float]:
    """Use ElevenLabs Forced Alignment API to align known script text to audio.
    Returns true per-word timings (~50-100ms accuracy). Much more accurate than
    transcription because we already know what was said.

    API: POST https://api.elevenlabs.io/v1/forced-alignment
    Form fields: file=<audio>, text=<script>
    Response: { "words": [{"text","start","end","loss"}], "characters": [...], "loss": ... }
    """
    import httpx

    api_key = config.ELEVENLABS_API_KEY
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not configured")
    if not script_text or not script_text.strip():
        raise RuntimeError("forced alignment requires script_text")

    print(f"[TS/ELEVEN] aligning {audio_path} against {len(script_text)} chars of script")
    with open(audio_path, "rb") as f:
        resp = httpx.post(
            "https://api.elevenlabs.io/v1/forced-alignment",
            headers={"xi-api-key": api_key},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={"text": script_text},
            timeout=300.0,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs forced-alignment failed {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    raw_words = payload.get("words") or []
    words: list[dict] = []
    for w in raw_words:
        original = (w.get("text") or "").strip()
        if not original:
            continue
        latin = _to_latin(original)
        if not latin:
            continue
        words.append({
            "word":     latin,
            "original": original,
            "start":    round(float(w.get("start", 0.0) or 0.0), 3),
            "end":      round(float(w.get("end",   0.0) or 0.0), 3),
        })

    duration = words[-1]["end"] if words else 0.0
    chars = payload.get("characters") or []
    if chars:
        try:
            duration = max(duration, float(chars[-1].get("end", duration) or duration))
        except Exception:
            pass

    print(f"[TS/ELEVEN] aligned {len(words)} words, duration={duration:.2f}s, loss={payload.get('loss')}")
    return words, duration


def _openai_transcribe_pipeline(audio_path: str) -> tuple[list[dict], float]:
    """Legacy fallback: OpenAI transcription (gpt-4o-transcribe → whisper-1)."""
    transcription = None
    last_err: Exception | None = None
    for model in ("gpt-4o-transcribe", "whisper-1"):
        try:
            transcription = _transcribe(audio_path, model)
            print(f"[TS/OPENAI] success with model={model}")
            break
        except Exception as exc:
            last_err = exc
            print(f"[TS/OPENAI] model={model} failed: {exc!r}")
            transcription = None
    if transcription is None:
        raise RuntimeError(f"all transcription models failed: {last_err!r}")

    raw_words    = getattr(transcription, "words",    None) or []
    raw_segments = getattr(transcription, "segments", None) or []

    if raw_segments:
        last = raw_segments[-1]
        duration = float(last.end if hasattr(last, "end") else last.get("end", 0))
    elif raw_words:
        lw = raw_words[-1]
        duration = float(lw.end if hasattr(lw, "end") else lw.get("end", 0))
    else:
        duration = 0.0

    words = _split_multiword_entries(_extract_words(raw_words))
    cov = _coverage(words, duration)
    print(f"[TS/OPENAI] primary coverage = {cov*100:.1f}% ({len(words)} words / {duration:.2f}s)")

    if cov < 0.70 and raw_segments:
        synth = _synthesize_from_segments(raw_segments, duration)
        synth_cov = _coverage(synth, duration)
        print(f"[TS/OPENAI] synthesized coverage = {synth_cov*100:.1f}% ({len(synth)} words)")
        if synth_cov > cov and synth:
            words = synth

    return words, duration


def get_timestamps(audio_path: str, script_text: str | None = None) -> tuple[list[dict], float]:
    """Primary: ElevenLabs Forced Alignment (needs script text + ELEVENLABS_API_KEY).
    Fallback: OpenAI transcription (gpt-4o-transcribe → whisper-1).
    """
    print(f"[TS] transcribing {audio_path}, script_text={'yes' if script_text else 'no'}")

    if script_text and config.ELEVENLABS_API_KEY:
        try:
            words, duration = _elevenlabs_forced_alignment(audio_path, script_text)
            if words:
                print(f"[TS] returning {len(words)} forced-aligned words, duration={duration:.2f}s")
                return words, duration
            print("[TS] forced alignment returned 0 words — falling back to OpenAI")
        except Exception as exc:
            print(f"[TS] forced alignment failed, falling back to OpenAI: {exc!r}")
    else:
        if not script_text:
            print("[TS] no script_text provided — skipping forced alignment")
        if not config.ELEVENLABS_API_KEY:
            print("[TS] ELEVENLABS_API_KEY not set — skipping forced alignment")

    words, duration = _openai_transcribe_pipeline(audio_path)
    print(f"[TS] returning {len(words)} OpenAI word-level timestamps, duration={duration:.2f}s")
    return words, duration

