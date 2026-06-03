"""
main.py — FastAPI application for sky-annotations-worker.
DEPLOY_MARKER: 2026-06-03-v4 (Support for dictionary-based bboxes and robust scaling)

All annotation pipeline routes:
  GET  /health
  POST /ocr/run
  POST /ocr/run-all
  POST /timestamps/run
  POST /timestamps/run-all
  POST /clips/render
  POST /clips/render-all
  GET  /slide-preview
  GET  /clips/file/{filename}
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lib.config import config
from lib.supabase_client import get_supabase
from lib.storage import (
    download_slide_to_tmp,
    SLIDES_BUCKET, AUDIO_BUCKET, VIDEO_CLIPS_BUCKET,
    slide_path, audio_path, clip_storage_path,
    download_to_tmp, upload_clip, cleanup,
)
from workers.ocr import run_ocr
from workers.timestamps import get_timestamps
# (run_ai_annotations import removed)

from workers.render import render_clip, get_audio_duration

import httpx



# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("/tmp/render").mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] sky-annotations-worker ready on port {config.PORT}")
    yield

app = FastAPI(title="sky-annotations-worker", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────

class OcrRunReq(BaseModel):
    script_id:    str
    chunk_id:     str
    chunk_number: int
    slide_source: str = "gamma"

class OcrRunAllReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"

class TsRunReq(BaseModel):
    script_id:    str
    chunk_id:     str
    chunk_number: int

class TsRunAllReq(BaseModel):
    script_id: str


class RenderReq(BaseModel):
    script_id:    str
    chunk_id:     str
    chunk_number: int
    slide_source: str = "gamma"

class RenderAllReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"

# (AiRunReq and AiRunAllReq removed as they are now handled by Supabase Edge Functions)




# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_chunks(script_id: str) -> list[dict]:
    res = get_supabase().table("script_chunks").select("*").eq("script_id", script_id).execute()
    return res.data or []

def _upsert_ocr(script_id: str, chunk_id: str, chunk_number: int,
                slide_source: str, words: list[dict]) -> None:
    get_supabase().table("ocr_results").upsert(
        {
            "script_id":    script_id,
            "chunk_id":     chunk_id,
            "chunk_number": chunk_number,
            "slide_source": slide_source,
            "words":        json.dumps(words),
        },
        on_conflict="script_id,chunk_id,slide_source",
    ).execute()

def _upsert_timestamps(script_id: str, chunk_id: str, chunk_number: int,
                       words: list[dict]) -> None:
    get_supabase().table("audio_timestamps").upsert(
        {
            "script_id":    script_id,
            "chunk_id":     chunk_id,
            "chunk_number": chunk_number,
            "words":        json.dumps(words),
        },
        on_conflict="script_id,chunk_id",
    ).execute()


# (_upsert_ai removed - handled by edge function)




def _upsert_clip(script_id: str, chunk_id: str, chunk_number: int,
                 slide_source: str, status: str = "rendering",
                 file_name: str | None = None, file_url: str | None = None,
                 duration: float | None = None, error_msg: str | None = None) -> dict:
    payload: dict = {
        "script_id":    script_id,
        "chunk_id":     chunk_id,
        "chunk_number": chunk_number,
        "slide_source": slide_source,
        "status":       status,
    }
    if file_name  is not None: payload["file_name"]  = file_name
    if file_url   is not None: payload["file_url"]   = file_url
    if duration   is not None: payload["duration"]   = duration
    if error_msg  is not None: payload["error_msg"]  = error_msg
    res = get_supabase().table("video_clips").upsert(
        payload, on_conflict="script_id,chunk_id,slide_source"
    ).execute()
    return res.data[0] if res.data else payload

def _get_ocr(chunk_id: str, slide_source: str) -> dict | None:
    res = (get_supabase().table("ocr_results").select("*")
           .eq("chunk_id", chunk_id).eq("slide_source", slide_source)
           .limit(1).execute())
    return res.data[0] if res.data else None

def _get_ts(chunk_id: str) -> dict | None:
    res = (get_supabase().table("audio_timestamps").select("*")
           .eq("chunk_id", chunk_id).limit(1).execute())
    return res.data[0] if res.data else None

def _get_ai(chunk_id: str, slide_source: str) -> dict | None:
    res = (get_supabase().table("clip_annotations").select("*")
           .eq("chunk_id", chunk_id).eq("slide_source", slide_source)
           .limit(1).execute())
    return res.data[0] if res.data else None

def _update_clip_status(script_id: str, chunk_id: str, slide_source: str,
                        **kwargs) -> None:
    get_supabase().table("video_clips").update(kwargs).eq(
        "chunk_id", chunk_id
    ).eq("slide_source", slide_source).execute()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

WORKER_VERSION = "2026-06-03.elevenlabs-forced-alignment-017-target-timing-repair"

@app.get("/health")
def health():
    return {"ok": True, "service": "sky-annotations-worker", "version": WORKER_VERSION}

@app.get("/version")
def version():
    return {"version": WORKER_VERSION}


# ══════════════════════════════════════════════════════════════════════════════
# OCR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr/run")
def ocr_run(req: OcrRunReq):
    tmp = None
    try:
        print(f"[OCR/run] chunk {req.chunk_number} ({req.chunk_id})")
        tmp = download_slide_to_tmp(req.script_id, req.chunk_number, ".png")
        words = run_ocr(tmp)
        _upsert_ocr(req.script_id, req.chunk_id, req.chunk_number, req.slide_source, words)
        return {"ok": True, "word_count": len(words)}
    except Exception as e:
        print(f"[OCR/run] ERROR chunk {req.chunk_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


def _ocr_all_job(script_id: str, slide_source: str) -> None:
    chunks = _get_chunks(script_id)
    print(f"[OCR/run-all] processing {len(chunks)} chunks")
    for chunk in chunks:
        tmp = None
        try:
            chunk_id     = chunk["id"]
            chunk_number = chunk["chunk_index"]
            tmp = download_slide_to_tmp(script_id, chunk_number, ".png")
            words = run_ocr(tmp)
            _upsert_ocr(script_id, chunk_id, chunk_number, slide_source, words)
            print(f"[OCR/run-all] chunk {chunk_number} done — {len(words)} words")
        except Exception as e:
            print(f"[OCR/run-all] chunk {chunk.get('chunk_index')} FAILED: {e}")
        finally:
            cleanup(tmp)

@app.post("/ocr/run-all")
def ocr_run_all(req: OcrRunAllReq, bg: BackgroundTasks):
    try:
        chunks = _get_chunks(req.script_id)
        bg.add_task(_ocr_all_job, req.script_id, req.slide_source)
        return {"ok": True, "queued": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TIMESTAMPS ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_audio_key(script_id: str, chunk_id: str, chunk_number: int) -> str:
    """Resolve the audio storage key from chunks.audio_url, with legacy fallback."""
    try:
        row = (get_supabase().table("script_chunks")
               .select("audio_url").eq("id", chunk_id).limit(1).execute())
        url = (row.data or [{}])[0].get("audio_url") or ""
        marker = f"/{AUDIO_BUCKET}/"
        if marker in url:
            return url.split(marker, 1)[1].split("?", 1)[0]
    except Exception as e:
        print(f"[AUDIO] resolve from DB failed: {e}")
    return audio_path(script_id, chunk_number)


def _get_chunk_text(chunk_id: str) -> str:
    try:
        row = (get_supabase().table("script_chunks")
               .select("content").eq("id", chunk_id).limit(1).execute())
        return ((row.data or [{}])[0].get("content") or "").strip()
    except Exception as e:
        print(f"[TS] failed to load chunk content for {chunk_id}: {e}")
        return ""


@app.post("/timestamps/run")
def timestamps_run(req: TsRunReq):
    tmp = None
    try:
        print(f"[TS/run] chunk {req.chunk_number} ({req.chunk_id})")
        key = _resolve_audio_key(req.script_id, req.chunk_id, req.chunk_number)
        tmp = download_to_tmp(AUDIO_BUCKET, key, ".mp3")
        script_text = _get_chunk_text(req.chunk_id)
        words, duration = get_timestamps(tmp, script_text=script_text)
        _upsert_timestamps(req.script_id, req.chunk_id, req.chunk_number, words)
        return {"ok": True, "word_count": len(words), "duration": duration}
    except Exception as e:
        print(f"[TS/run] ERROR chunk {req.chunk_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(tmp)


def _ts_all_job(script_id: str) -> None:
    chunks = _get_chunks(script_id)
    print(f"[TS/run-all] processing {len(chunks)} chunks")
    for chunk in chunks:
        tmp = None
        try:
            chunk_id     = chunk["id"]
            chunk_number = chunk["chunk_index"]
            key = _resolve_audio_key(script_id, chunk_id, chunk_number)
            tmp = download_to_tmp(AUDIO_BUCKET, key, ".mp3")
            script_text = (chunk.get("content") or "").strip() or _get_chunk_text(chunk_id)
            words, _ = get_timestamps(tmp, script_text=script_text)
            _upsert_timestamps(script_id, chunk_id, chunk_number, words)
            print(f"[TS/run-all] chunk {chunk_number} done — {len(words)} words")
        except Exception as e:
            print(f"[TS/run-all] chunk {chunk.get('chunk_index')} FAILED: {e}")
        finally:
            cleanup(tmp)


@app.post("/timestamps/run-all")
def timestamps_run_all(req: TsRunAllReq, bg: BackgroundTasks):
    try:
        chunks = _get_chunks(req.script_id)
        bg.add_task(_ts_all_job, req.script_id)
        return {"ok": True, "queued": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# (AI ANNOTATIONS ROUTES removed - now handled by Supabase Edge Functions)




# ══════════════════════════════════════════════════════════════════════════════
# RENDER ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def _render_job(script_id: str, chunk_id: str, chunk_number: int, slide_source: str) -> None:
    tmp_slide = tmp_audio = tmp_out = None
    try:
        print(f"[RENDER] start chunk {chunk_number} ({chunk_id})")

        ai_row = _get_ai(chunk_id, slide_source)
        if not ai_row:
            raise ValueError("Annotations not found — run AI step first")

        tmp_slide = download_slide_to_tmp(script_id, chunk_number, ".png")
        tmp_audio = download_to_tmp(AUDIO_BUCKET, _resolve_audio_key(script_id, chunk_id, chunk_number), ".mp3")

        os.makedirs("/tmp/render", exist_ok=True)
        _fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir="/tmp/render")
        os.close(_fd)

        annotations = json.loads(ai_row["annotations"])

        duration = render_clip(tmp_slide, tmp_audio, annotations, tmp_out)

        storage_key = clip_storage_path(script_id, chunk_number, slide_source)
        file_url    = upload_clip(tmp_out, storage_key)

        _update_clip_status(
            script_id, chunk_id, slide_source,
            status="done",
            file_name=storage_key.split("/")[-1],
            file_url=file_url,
            duration=duration,
            error_msg=None,
        )
        print(f"[RENDER] chunk {chunk_number} done — {duration:.2f}s → {file_url}")

    except Exception as e:
        print(f"[RENDER] chunk {chunk_number} ERROR: {e}")
        _update_clip_status(script_id, chunk_id, slide_source,
                            status="error", error_msg=str(e)[:500])
    finally:
        cleanup(tmp_slide, tmp_audio, tmp_out)


@app.post("/clips/render")
def clips_render(req: RenderReq, bg: BackgroundTasks):
    try:
        ai_row = _get_ai(req.chunk_id, req.slide_source)
        if not ai_row:
            raise HTTPException(status_code=400, detail="Annotations not generated yet — run AI step first")

        _upsert_clip(req.script_id, req.chunk_id, req.chunk_number, req.slide_source, status="rendering")
        bg.add_task(_render_job, req.script_id, req.chunk_id, req.chunk_number, req.slide_source)
        return {"ok": True, "status": "rendering"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _render_all_job(script_id: str, slide_source: str) -> None:
    chunks = _get_chunks(script_id)
    print(f"[RENDER/all] processing {len(chunks)} chunks")
    for chunk in chunks:
        chunk_id     = chunk["id"]
        chunk_number = chunk["chunk_index"]
        ai_row = _get_ai(chunk_id, slide_source)
        if not ai_row:
            print(f"[RENDER/all] chunk {chunk_number} skipped — no annotations")
            continue
        _upsert_clip(script_id, chunk_id, chunk_number, slide_source, status="rendering")
        _render_job(script_id, chunk_id, chunk_number, slide_source)

@app.post("/clips/render-all")
def clips_render_all(req: RenderAllReq, bg: BackgroundTasks):
    try:
        chunks = _get_chunks(req.script_id)
        bg.add_task(_render_all_job, req.script_id, req.slide_source)
        return {"ok": True, "queued": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/slide-preview")
async def slide_preview(
    script_id:    str = Query(...),
    chunk_id:     str = Query(...),
    chunk_number: int = Query(...),
    source:       str = Query("gamma"),
):
    try:
        url = (
            f"{config.SUPABASE_URL}/storage/v1/object/public/"
            f"{SLIDES_BUCKET}/{slide_path(script_id, chunk_number)}"
        )
        async with httpx.AsyncClient() as client:
            r = await client.get(url, follow_redirects=True, timeout=30)
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail="Slide not found")
            r.raise_for_status()
            content = r.content

        return StreamingResponse(
            iter([content]),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clips/file/{filename}")
async def stream_clip(filename: str):
    try:
        # filename is just the last segment; derive script_id from DB lookup
        # For simplicity, we redirect to Supabase public URL
        # The caller can also access the file_url directly from video_clips table
        res = (get_supabase().table("video_clips")
               .select("file_url").eq("file_name", filename).limit(1).execute())
        if not res.data or not res.data[0].get("file_url"):
            raise HTTPException(status_code=404, detail="Clip not found")

        file_url = res.data[0]["file_url"]

        async def _stream():
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", file_url, follow_redirects=True) as resp:
                    async for chunk in resp.aiter_bytes(65536):
                        yield chunk

        return StreamingResponse(_stream(), media_type="video/mp4")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MERGE — combine all rendered clips into a single mega video
# ══════════════════════════════════════════════════════════════════════════════

import subprocess

class MergeReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"

def _set_merge_status(script_id: str, value: dict) -> None:
    key = f"merge:{script_id}"
    get_supabase().table("app_metadata").upsert(
        {"key": key, "value": value, "updated_at": "now()"},
        on_conflict="key",
    ).execute()

def _merge_job(script_id: str, slide_source: str) -> None:
    work_dir = None
    try:
        _set_merge_status(script_id, {"status": "running", "url": None, "error": None})
        # fetch ordered clips for this script + slide_source
        res = (get_supabase().table("video_clips").select("*")
               .eq("script_id", script_id).eq("slide_source", slide_source)
               .eq("status", "done").order("chunk_number").execute())
        clips = res.data or []
        if not clips:
            raise ValueError("No rendered clips found for this script + slide source")

        work_dir = tempfile.mkdtemp(prefix="merge_", dir="/tmp/render")
        list_file = os.path.join(work_dir, "list.txt")
        local_files: list[str] = []

        # download each clip locally
        for i, clip in enumerate(clips):
            file_url = clip.get("file_url")
            if not file_url:
                continue
            local = os.path.join(work_dir, f"clip_{i:03d}.mp4")
            with httpx.stream("GET", file_url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for ch in r.iter_bytes(65536):
                        f.write(ch)
            local_files.append(local)

        if not local_files:
            raise ValueError("All clips have empty file_url")

        # write ffmpeg concat list
        with open(list_file, "w") as f:
            for p in local_files:
                f.write(f"file '{p}'\n")

        out_path = os.path.join(work_dir, "mega.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            out_path,
        ]
        print(f"[MERGE] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # fallback: re-encode (handles slightly different streams)
            print(f"[MERGE] copy concat failed, retrying with re-encode\n{proc.stderr[-2000:]}")
            cmd2 = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                out_path,
            ]
            proc2 = subprocess.run(cmd2, capture_output=True, text=True)
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg merge failed: {proc2.stderr[-1000:]}")

        storage_key = f"{script_id}/mega_{slide_source}.mp4"
        public_url = upload_clip(out_path, storage_key)

        _set_merge_status(script_id, {
            "status": "done",
            "url": public_url,
            "error": None,
            "clip_count": len(local_files),
            "slide_source": slide_source,
        })
        print(f"[MERGE] done — {public_url}")
    except Exception as e:
        print(f"[MERGE] ERROR: {e}")
        _set_merge_status(script_id, {"status": "error", "url": None, "error": str(e)[:500]})
    finally:
        if work_dir:
            try:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


@app.post("/merge/run")
def merge_run(req: MergeReq, bg: BackgroundTasks):
    try:
        _set_merge_status(req.script_id, {"status": "queued", "url": None, "error": None})
        bg.add_task(_merge_job, req.script_id, req.slide_source)
        return {"ok": True, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/merge/status")
def merge_status(script_id: str = Query(...)):
    res = (get_supabase().table("app_metadata").select("*")
           .eq("key", f"merge:{script_id}").limit(1).execute())
    if not res.data:
        return {"status": "idle", "url": None}
    return res.data[0].get("value") or {"status": "idle", "url": None}
