"""
lib/storage.py — Supabase Storage helpers for download and upload.

Bucket layout (all public):
  slides      → {script_id}/slide_{NNN}.png          (chunk_number zero-padded to 3 digits)
  audio-files → {script_id}/audio_{N}.mp3
  video-clips → {script_id}/clip_{NNN}_{slide_source}.mp4
"""

import os
import tempfile
from pathlib import Path

import httpx
from lib.supabase_client import get_supabase
from lib.config import config

VIDEO_CLIPS_BUCKET = "video-clips"
SLIDES_BUCKET      = "slides"
AUDIO_BUCKET       = "audio-files"


def _public_url(bucket: str, path: str) -> str:
    return f"{config.SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"


def download_to_tmp(bucket: str, path: str, suffix: str = "") -> str:
    """
    Download a file from Supabase Storage to a local temp file.
    Returns the local file path. Caller is responsible for cleanup.
    """
    url = _public_url(bucket, path)
    print(f"[STORAGE] download {bucket}/{path}")
    r = httpx.get(url, follow_redirects=True, timeout=120)
    if r.status_code in (400, 404):
        raise FileNotFoundError(f"Storage file not found: {bucket}/{path} (status {r.status_code})")
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name


def download_slide_to_tmp(script_id: str, chunk_number: int, suffix: str = ".png") -> str:
    """
    Download a slide image, trying both 0- and 1-indexed naming conventions.
    """
    last_err: Exception | None = None
    for candidate in (chunk_number, chunk_number + 1):
        path = f"{script_id}/slide_{candidate:03d}.png"
        try:
            return download_to_tmp(SLIDES_BUCKET, path, suffix)
        except FileNotFoundError as e:
            last_err = e
            continue
    raise last_err or FileNotFoundError(f"No slide for {script_id}/chunk {chunk_number}")


def slide_path(script_id: str, chunk_number: int) -> str:
    return f"{script_id}/slide_{chunk_number:03d}.png"


def audio_path(script_id: str, chunk_number: int) -> str:
    return f"{script_id}/audio_{chunk_number}.mp3"


def clip_storage_path(script_id: str, chunk_number: int, slide_source: str) -> str:
    return f"{script_id}/clip_{chunk_number:03d}_{slide_source}.mp4"


def upload_clip(local_path: str, storage_path: str) -> str:
    """
    Upload a rendered MP4 to Supabase Storage video-clips bucket.
    Creates the bucket if it doesn't exist.
    Returns the public URL.
    """
    sb = get_supabase()

    # Ensure bucket exists
    try:
        buckets = sb.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if VIDEO_CLIPS_BUCKET not in bucket_names:
            sb.storage.create_bucket(VIDEO_CLIPS_BUCKET, options={"public": True})
            print(f"[STORAGE] Created bucket: {VIDEO_CLIPS_BUCKET}")
    except Exception as e:
        print(f"[STORAGE] Bucket check warning: {e}")

    with open(local_path, "rb") as f:
        data = f.read()

    print(f"[STORAGE] upload {VIDEO_CLIPS_BUCKET}/{storage_path} ({len(data)//1024}KB)")
    sb.storage.from_(VIDEO_CLIPS_BUCKET).upload(
        storage_path,
        data,
        {"content-type": "video/mp4", "upsert": "true"},
    )
    return _public_url(VIDEO_CLIPS_BUCKET, storage_path)


def cleanup(*paths: str) -> None:
    """Delete temp files silently."""
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except Exception:
                pass
