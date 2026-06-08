# Updated: 2026-06-03 19:15 - Removed AI Annotation routes
"""
main.py — FastAPI application for sky-annotations-worker.

Pipeline routes:
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
import threading

# Global render lock — ensures only ONE clip renders at a time on the worker.
# Each render holds ~1500 raw RGBA frames in memory (≈5GB); running multiple
# in parallel OOM-kills the container. BackgroundTasks queue up behind this.
RENDER_LOCK = threading.Lock()
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lib.config import config
from lib.supabase_client import get_supabase
from lib.storage import (
    SLIDES_BUCKET, AUDIO_BUCKET, VIDEO_CLIPS_BUCKET,
    slide_path, audio_path, clip_storage_path,
    download_to_tmp, download_audio_to_tmp, upload_clip, cleanup,
)
from workers.ocr import run_ocr
from workers.timestamps import get_timestamps
from workers.render import render_clip, get_audio_duration
from workers.merger import merge_script_clips
from workers.editor import (
    apply_cuts as editor_apply_cuts,
    _meta_key as editor_meta_key,
    export_video as editor_export_video,
    _export_meta_key as editor_export_meta_key,
)

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

class MergeRunReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"


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


def _update_clip_status(script_id: str, chunk_id: str, slide_source: str,
                        **kwargs) -> None:
    get_supabase().table("video_clips").update(kwargs).eq(
        "chunk_id", chunk_id
    ).eq("slide_source", slide_source).execute()


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"ok": True, "service": "sky-annotations-worker", "version": "v1.0.7-bbox-fix"}


# ══════════════════════════════════════════════════════════════════════════════
# OCR ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr/run")
def ocr_run(req: OcrRunReq):
    tmp = None
    try:
        print(f"[OCR/run] chunk {req.chunk_number} ({req.chunk_id})")
        tmp = download_to_tmp(SLIDES_BUCKET, slide_path(req.script_id, req.chunk_number), ".png")
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
            tmp = download_to_tmp(SLIDES_BUCKET, slide_path(script_id, chunk_number), ".png")
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

@app.post("/timestamps/run")
def timestamps_run(req: TsRunReq):
    tmp = None
    try:
        print(f"[TS/run] chunk {req.chunk_number} ({req.chunk_id})")
        tmp = download_audio_to_tmp(req.script_id, req.chunk_number)
        words, duration = get_timestamps(tmp)
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
            tmp = download_audio_to_tmp(script_id, chunk_number)
            words, _ = get_timestamps(tmp)
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




# ══════════════════════════════════════════════════════════════════════════════
# RENDER ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def _render_job(script_id: str, chunk_id: str, chunk_number: int, slide_source: str) -> None:
    # Serialize all renders globally to avoid OOM (each render is memory-heavy).
    print(f"[RENDER] queued chunk {chunk_number} ({chunk_id}) — waiting for render lock")
    with RENDER_LOCK:
        tmp_slide = tmp_audio = tmp_out = None
        try:
            print(f"[RENDER] start chunk {chunk_number} ({chunk_id})")

            # Fetch annotations from DB (now generated by Supabase Edge Function)
            res = (get_supabase().table("clip_annotations").select("annotations")
                   .eq("chunk_id", chunk_id).eq("slide_source", slide_source)
                   .limit(1).execute())
            ai_row = res.data[0] if res.data else None
            if not ai_row:
                raise ValueError("Annotations not found in DB — run AI step first")

            tmp_slide = download_to_tmp(SLIDES_BUCKET, slide_path(script_id, chunk_number), ".png")
            tmp_audio = download_audio_to_tmp(script_id, chunk_number)

            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp/render").name
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
        res = (get_supabase().table("clip_annotations").select("annotations")
               .eq("chunk_id", req.chunk_id).eq("slide_source", req.slide_source)
               .limit(1).execute())
        ai_row = res.data[0] if res.data else None
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
        res = (get_supabase().table("clip_annotations").select("annotations")
               .eq("chunk_id", chunk_id).eq("slide_source", slide_source)
               .limit(1).execute())
        ai_row = res.data[0] if res.data else None
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
# MERGE ROUTE
# ══════════════════════════════════════════════════════════════════════════════

def _merge_job(script_id: str, slide_source: str) -> None:
    try:
        merge_script_clips(script_id, slide_source)
    except Exception as e:
        print(f"[MERGE] background job ERROR: {e}")

@app.post("/merge/run")
def merge_run(req: MergeRunReq, bg: BackgroundTasks):
    try:
        # mark queued immediately so UI poll picks it up
        get_supabase().table("app_metadata").upsert({
            "key": f"merge:{req.script_id}",
            "value": {"status": "queued", "slide_source": req.slide_source},
        }, on_conflict="key").execute()
        bg.add_task(_merge_job, req.script_id, req.slide_source)
        return {"ok": True, "status": "queued"}
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
# EDITOR ROUTES (apply user cuts → overwrite original video in storage)
# Isolated from OCR / timestamps / render / merge — does not touch their tables.
# ══════════════════════════════════════════════════════════════════════════════

class EditorApplyCutsReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"
    bucket:       str = "video-clips"
    path:         str            # e.g. "{script_id}/mega_gamma.mp4"
    cuts:         list[list[float]]  # [[start_sec, end_sec], ...] ranges to REMOVE


def _editor_job(script_id: str, slide_source: str, bucket: str, path: str,
                cuts: list[list[float]]) -> None:
    try:
        editor_apply_cuts(script_id, slide_source, bucket, path, cuts)
    except Exception as e:
        print(f"[EDITOR] background job ERROR: {e}")


@app.post("/editor/apply-cuts")
def editor_apply_cuts_route(req: EditorApplyCutsReq, bg: BackgroundTasks):
    try:
        key = editor_meta_key(req.script_id, req.slide_source)
        get_supabase().table("app_metadata").upsert({
            "key":   key,
            "value": {
                "status": "queued",
                "bucket": req.bucket,
                "path":   req.path,
                "slide_source": req.slide_source,
                "error":  None,
                "url":    None,
            },
        }, on_conflict="key").execute()
        bg.add_task(_editor_job, req.script_id, req.slide_source,
                    req.bucket, req.path, req.cuts)
        return {"ok": True, "status": "queued", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/editor/status")
def editor_status(script_id: str = Query(...), slide_source: str = Query("gamma")):
    try:
        key = editor_meta_key(script_id, slide_source)
        res = (get_supabase().table("app_metadata").select("value")
               .eq("key", key).limit(1).execute())
        if not res.data:
            return {"ok": True, "status": "unknown", "value": None}
        return {"ok": True, "value": res.data[0]["value"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── EDITOR EXPORT ─────────────────────────────────────────────────────────────

class EditorExportReq(BaseModel):
    script_id:    str
    slide_source: str = "gamma"
    bucket:       str = "video-clips"
    path:         str
    quality:      str = "medium"  # "low" | "medium" | "high"


def _editor_export_job(script_id: str, slide_source: str, bucket: str,
                       path: str, quality: str) -> None:
    try:
        editor_export_video(script_id, slide_source, bucket, path, quality)
    except Exception as e:
        print(f"[EDITOR/EXPORT] background job ERROR: {e}")


@app.post("/editor/export")
def editor_export_route(req: EditorExportReq, bg: BackgroundTasks):
    try:
        key = editor_export_meta_key(req.script_id, req.slide_source)
        get_supabase().table("app_metadata").upsert({
            "key":   key,
            "value": {
                "status": "queued", "quality": req.quality,
                "bucket": req.bucket, "path": req.path,
                "url": None, "size_bytes": None, "error": None,
            },
        }, on_conflict="key").execute()
        bg.add_task(_editor_export_job, req.script_id, req.slide_source,
                    req.bucket, req.path, req.quality)
        return {"ok": True, "status": "queued", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/editor/export-status")
def editor_export_status(script_id: str = Query(...), slide_source: str = Query("gamma")):
    try:
        key = editor_export_meta_key(script_id, slide_source)
        res = (get_supabase().table("app_metadata").select("value")
               .eq("key", key).limit(1).execute())
        if not res.data:
            return {"ok": True, "status": "unknown", "value": None}
        return {"ok": True, "value": res.data[0]["value"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
