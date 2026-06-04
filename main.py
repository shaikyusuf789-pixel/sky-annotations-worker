"""
railway-worker/main.py — Entry point for the Railway worker.
Version: v1.1.1-render-fix
"""

import os
import time
import traceback
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from lib.storage import upload_to_r2
from workers.render import render_clip

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("[ERROR] DATABASE_URL is not set")
    exit(1)

def get_db_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def _update_clip_status(clip_id, status, video_url=None, error_msg=None):
    try:
        conn = get_db_conn()
        with conn:
            with conn.cursor() as cur:
                if video_url:
                    cur.execute(
                        "UPDATE video_clips SET status = %s, video_url = %s, updated_at = NOW() WHERE id = %s",
                        (status, video_url, clip_id)
                    )
                elif error_msg:
                    cur.execute(
                        "UPDATE video_clips SET status = %s, error_msg = %s, updated_at = NOW() WHERE id = %s",
                        (status, error_msg, clip_id)
                    )
                else:
                    cur.execute(
                        "UPDATE video_clips SET status = %s, updated_at = NOW() WHERE id = %s",
                        (status, clip_id)
                    )
    except Exception as e:
        print(f"[ERROR] Failed to update clip status: {e}")

def _render_job(clip):
    clip_id = clip["id"]
    print(f"[JOB] Starting render for clip {clip_id}")
    _update_clip_status(clip_id, "processing")
    
    tmp_output = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        # annotations might be a string (JSON) or a list depending on pg driver / schema
        anns = clip["annotations"]
        if isinstance(anns, str):
            import json
            anns = json.loads(anns)
            
        render_clip(
            slide_path=clip["slide_url"], # Assuming these are local paths or reachable URLs
            audio_path=clip["audio_url"],
            annotations=anns,
            output_path=tmp_output
        )
        
        # Upload to R2
        r2_url = upload_to_r2(tmp_output, f"clips/{clip_id}.mp4")
        if not r2_url:
            raise RuntimeError("R2 upload failed")
            
        _update_clip_status(clip_id, "completed", video_url=r2_url)
        print(f"[JOB] Completed clip {clip_id}")
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[ERROR] Render failed for {clip_id}:\n{error_detail}")
        _update_clip_status(clip_id, "failed", error_msg=str(e))
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)

def main():
    print("--- Railway Worker Starting (v1.1.1-render-fix) ---")
    while True:
        try:
            conn = get_db_conn()
            with conn:
                with conn.cursor() as cur:
                    # Simple polling for pending clips
                    cur.execute(
                        "SELECT * FROM video_clips WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
                    )
                    clip = cur.fetchone()
                    
                    if clip:
                        _render_job(clip)
                    else:
                        time.sleep(5)
            conn.close()
        except Exception as e:
            print(f"[ERROR] Main loop exception: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
