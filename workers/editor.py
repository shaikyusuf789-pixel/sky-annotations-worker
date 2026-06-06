"""
workers/editor.py — Apply user-defined cuts to a video and overwrite the original.

Input: a list of [start, end] ranges (in seconds) to DELETE from the source video.
The worker computes the inverse (kept ranges), runs ffmpeg with a select/aselect
filter graph to concatenate the kept ranges, re-encodes (cuts are not on
keyframes, so stream-copy won't work), and uploads the result back to the SAME
storage path — overwriting the original. No duplicates.

Status is tracked in the `app_metadata` table under key:
    editor:{script_id}:{slide_source}

Value shape:
    { status: "queued"|"running"|"done"|"error",
      url: <public url>|null,
      error: <string>|null,
      bucket: <string>, path: <string>,
      kept_seconds: <float>, removed_seconds: <float> }

This module is intentionally standalone — it does NOT import or modify any of
the existing OCR / timestamps / annotations / render / merge code paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import List, Tuple

import httpx

from lib.config import config
from lib.supabase_client import get_supabase
from lib.storage import cleanup


def _meta_key(script_id: str, slide_source: str) -> str:
    return f"editor:{script_id}:{slide_source}"


def _update_meta(script_id: str, slide_source: str, **fields) -> None:
    sb = get_supabase()
    key = _meta_key(script_id, slide_source)
    # Read existing value first so we don't clobber unrelated fields.
    try:
        res = sb.table("app_metadata").select("value").eq("key", key).limit(1).execute()
        current = (res.data[0]["value"] if res.data else {}) or {}
    except Exception:
        current = {}
    current.update(fields)
    sb.table("app_metadata").upsert(
        {"key": key, "value": current},
        on_conflict="key",
    ).execute()


# ── ffprobe / ffmpeg helpers ─────────────────────────────────────────────────

def _ffprobe_duration(path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


def _normalize_cuts(cuts: List[List[float]], duration: float) -> List[Tuple[float, float]]:
    """Sort, clip to [0,duration], drop empties, merge overlaps."""
    cleaned: List[Tuple[float, float]] = []
    for c in cuts or []:
        if len(c) < 2:
            continue
        s = max(0.0, float(c[0]))
        e = min(duration, float(c[1]))
        if e - s > 0.02:  # ignore <20ms
            cleaned.append((s, e))
    cleaned.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in cleaned:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _kept_ranges(cuts: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    kept: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in cuts:
        if s > cursor:
            kept.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        kept.append((cursor, duration))
    # Drop micro-segments
    return [(s, e) for (s, e) in kept if e - s > 0.05]


def _build_filter(kept: List[Tuple[float, float]]) -> str:
    """
    Build an ffmpeg -filter_complex string that keeps only the given ranges
    and concatenates them with proper PTS reset for both video and audio.
    """
    if not kept:
        raise ValueError("Nothing left after cuts — refusing to produce empty video")
    parts = []
    for i, (s, e) in enumerate(kept):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(kept)))
    parts.append(f"{concat_inputs}concat=n={len(kept)}:v=1:a=1[outv][outa]")
    return ";".join(parts)


# ── Storage helpers (local to this module — no shared state with mega merge) ──

def _download_storage(bucket: str, path: str, suffix: str = ".mp4") -> str:
    url = f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
    print(f"[EDITOR] download {bucket}/{path}")
    r = httpx.get(url, follow_redirects=True, timeout=300)
    if r.status_code == 404:
        raise FileNotFoundError(f"Source video not found: {bucket}/{path}")
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def _upload_overwrite(local_path: str, bucket: str, storage_path: str) -> str:
    sb = get_supabase()
    with open(local_path, "rb") as f:
        data = f.read()
    print(f"[EDITOR] upload (overwrite) {bucket}/{storage_path} ({len(data)//1024} KB)")
    sb.storage.from_(bucket).upload(
        storage_path,
        data,
        {"content-type": "video/mp4", "upsert": "true"},
    )
    return f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"


# ── Main entry ───────────────────────────────────────────────────────────────

def apply_cuts(
    script_id: str,
    slide_source: str,
    bucket: str,
    path: str,
    cuts: List[List[float]],
) -> dict:
    """
    Download the source video at bucket/path, drop the given cut ranges,
    re-encode the remaining segments, and overwrite the original at bucket/path.
    """
    tmp_in = tmp_out = None
    try:
        print(f"[EDITOR] apply_cuts script={script_id} src={bucket}/{path} cuts={len(cuts)}")
        _update_meta(
            script_id, slide_source,
            status="running", error=None, url=None,
            bucket=bucket, path=path,
        )

        tmp_in = _download_storage(bucket, path, ".mp4")
        duration = _ffprobe_duration(tmp_in)
        norm_cuts = _normalize_cuts(cuts, duration)
        kept = _kept_ranges(norm_cuts, duration)
        kept_seconds = sum(e - s for s, e in kept)
        removed_seconds = duration - kept_seconds
        print(f"[EDITOR] duration={duration:.2f}s kept={kept_seconds:.2f}s "
              f"removed={removed_seconds:.2f}s segments={len(kept)}")

        if not kept:
            raise ValueError("All content was cut — refusing to produce empty video")

        # Fast path: no actual cuts → just no-op, leave original in place.
        if not norm_cuts:
            url = f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"
            _update_meta(
                script_id, slide_source,
                status="done", url=url, error=None,
                kept_seconds=kept_seconds, removed_seconds=0.0,
            )
            return {"ok": True, "url": url, "noop": True}

        filter_complex = _build_filter(kept)
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            tmp_out,
        ]
        print(f"[EDITOR] ffmpeg: {' '.join(cmd[:6])} … (+filter_complex {len(filter_complex)} chars)")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-25:])
            raise RuntimeError(f"ffmpeg failed: {tail}")

        url = _upload_overwrite(tmp_out, bucket, path)
        _update_meta(
            script_id, slide_source,
            status="done", url=url, error=None,
            kept_seconds=kept_seconds, removed_seconds=removed_seconds,
        )
        print(f"[EDITOR] done → {url}")
        return {"ok": True, "url": url, "kept_seconds": kept_seconds,
                "removed_seconds": removed_seconds, "segments": len(kept)}

    except Exception as e:
        print(f"[EDITOR] ERROR: {e}")
        _update_meta(script_id, slide_source, status="error", error=str(e)[:500])
        raise
    finally:
        cleanup(tmp_in, tmp_out)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT — re-encode current video at a chosen quality preset and upload to
# a temp path under the same bucket. Returns a public URL the browser can
# download. Does NOT overwrite the original.
# ══════════════════════════════════════════════════════════════════════════════

import time

QUALITY_PRESETS = {
    "low":    {"vbitrate": "600k",  "abitrate": "96k",  "scale": "iw*0.6:ih*0.6", "preset": "veryfast", "crf": "30"},
    "medium": {"vbitrate": "1800k", "abitrate": "128k", "scale": "iw:ih",         "preset": "veryfast", "crf": "24"},
    "high":   {"vbitrate": "4500k", "abitrate": "192k", "scale": "iw:ih",         "preset": "medium",   "crf": "20"},
}


def _export_meta_key(script_id: str, slide_source: str) -> str:
    return f"editor_export:{script_id}:{slide_source}"


def _update_export_meta(script_id: str, slide_source: str, **fields) -> None:
    sb = get_supabase()
    key = _export_meta_key(script_id, slide_source)
    try:
        res = sb.table("app_metadata").select("value").eq("key", key).limit(1).execute()
        current = (res.data[0]["value"] if res.data else {}) or {}
    except Exception:
        current = {}
    current.update(fields)
    sb.table("app_metadata").upsert({"key": key, "value": current}, on_conflict="key").execute()


def export_video(script_id: str, slide_source: str, bucket: str, path: str, quality: str) -> dict:
    """Download the current video at bucket/path, transcode at the quality preset,
    upload to bucket/exports/{script_id}/{slide_source}_{quality}_{ts}.mp4, return URL."""
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality '{quality}'")
    preset = QUALITY_PRESETS[quality]
    tmp_in = tmp_out = None
    try:
        print(f"[EDITOR/EXPORT] script={script_id} q={quality} src={bucket}/{path}")
        _update_export_meta(script_id, slide_source, status="running", quality=quality,
                            error=None, url=None, size_bytes=None)
        tmp_in = _download_storage(bucket, path, ".mp4")
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in,
            "-vf", f"scale={preset['scale']}",
            "-c:v", "libx264", "-preset", preset["preset"], "-crf", preset["crf"],
            "-b:v", preset["vbitrate"], "-maxrate", preset["vbitrate"], "-bufsize", preset["vbitrate"],
            "-c:a", "aac", "-b:a", preset["abitrate"],
            "-movflags", "+faststart",
            tmp_out,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-25:])
            raise RuntimeError(f"ffmpeg failed: {tail}")

        size_bytes = os.path.getsize(tmp_out)
        ts = int(time.time())
        export_path = f"exports/{script_id}/{slide_source}_{quality}_{ts}.mp4"
        sb = get_supabase()
        with open(tmp_out, "rb") as f:
            data = f.read()
        sb.storage.from_(bucket).upload(export_path, data,
                                        {"content-type": "video/mp4", "upsert": "true"})
        url = f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{export_path}"
        _update_export_meta(script_id, slide_source, status="done", quality=quality,
                            url=url, size_bytes=size_bytes, error=None,
                            export_path=export_path)
        print(f"[EDITOR/EXPORT] done {quality} {size_bytes//1024} KB → {url}")
        return {"ok": True, "url": url, "size_bytes": size_bytes, "quality": quality}

    except Exception as e:
        print(f"[EDITOR/EXPORT] ERROR: {e}")
        _update_export_meta(script_id, slide_source, status="error", error=str(e)[:500])
        raise
    finally:
        cleanup(tmp_in, tmp_out)
