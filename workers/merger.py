import os
import tempfile
import subprocess
import json
from pathlib import Path
from lib.supabase_client import get_supabase
from lib.storage import VIDEO_CLIPS_BUCKET, download_to_tmp, upload_clip, cleanup

def merge_script_clips(script_id: str, slide_source: str) -> dict:
    """
    1. Fetch all 'done' clips for script_id + slide_source
    2. Sort by chunk_number
    3. Download them
    4. Concat with ffmpeg
    5. Upload mega video
    6. Update app_metadata
    """
    sb = get_supabase()
    meta_key = f"merge:{script_id}"
    
    def _update_meta(status: str, url: str = None, error: str = None, clip_count: int = 0):
        val = {
            "status": status,
            "url": url,
            "error": error,
            "clip_count": clip_count,
            "slide_source": slide_source
        }
        sb.table("app_metadata").upsert({
            "key": meta_key,
            "value": val
        }, on_conflict="key").execute()

    try:
        print(f"[MERGE] Starting merge for script {script_id} ({slide_source})")
        _update_meta("running")

        # 1. Fetch clips
        res = (sb.table("video_clips").select("*")
               .eq("script_id", script_id)
               .eq("slide_source", slide_source)
               .eq("status", "done")
               .order("chunk_number")
               .execute())
        
        clips = res.data or []
        if not clips:
            raise ValueError("No rendered clips found for this script. Render chunks first.")

        # 2. Download clips
        tmp_paths = []
        try:
            for c in clips:
                # Use the storage path from clip_storage_path in storage.py
                # Path is script_id/clip_NNN_source.mp4
                fname = f"clip_{c['chunk_number']:03d}_{slide_source}.mp4"
                spath = f"{script_id}/{fname}"
                print(f"[MERGE] Downloading chunk {c['chunk_number']} from {spath}")
                local = download_to_tmp(VIDEO_CLIPS_BUCKET, spath, ".mp4")
                tmp_paths.append(local)

            # 3. Create concat file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                for p in tmp_paths:
                    f.write(f"file '{p}'\n")
                concat_list_path = f.name

            # 4. FFmpeg concat
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            # We use 'copy' codec because all chunks should have identical params (H, W, FPS) from render.py
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                out_path
            ]
            print(f"[MERGE] Running ffmpeg: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[MERGE] FFmpeg ERROR: {proc.stderr}")
                raise RuntimeError(f"FFmpeg failed: {proc.stderr}")

            # 5. Upload
            storage_key = f"{script_id}/mega_{slide_source}.mp4"
            final_url = upload_clip(out_path, storage_key)
            
            # 6. Success
            _update_meta("done", url=final_url, clip_count=len(clips))
            print(f"[MERGE] Success! Mega video: {final_url}")
            return {"ok": True, "url": final_url, "clips": len(clips)}

        finally:
            # Cleanup all locals
            cleanup(*tmp_paths)
            if 'concat_list_path' in locals(): cleanup(concat_list_path)
            if 'out_path' in locals(): cleanup(out_path)

    except Exception as e:
        print(f"[MERGE] ERROR: {e}")
        _update_meta("error", error=str(e))
        raise
