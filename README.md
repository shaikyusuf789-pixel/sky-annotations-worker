# sky-annotations-worker

FastAPI backend for the SKY Academy video annotation pipeline.
Deployed to Railway, called from the Lovable frontend.

---

## What it does

For each script chunk, runs 4 sequential steps:

| Step | Endpoint | What it does |
|------|----------|-------------|
| 1. OCR | `POST /ocr/run` | Tesseract reads English text + bboxes from a slide PNG |
| 2. Timestamps | `POST /timestamps/run` | OpenAI Whisper returns per-word start/end times from MP3 |
| 3. AI Annotations | `POST /ai/run` | GPT-4o picks 3–8 key words to highlight and when |
| 4. Render Clip | `POST /clips/render` | FFmpeg composites slide + audio + animated SVG strokes → MP4 |

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL (Settings → API) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key — full DB + storage access |
| `OPENAI_API_KEY` | OpenAI API key (Whisper + GPT-4o) |
| `PORT` | Injected automatically by Railway |

---

## Supabase setup

### Storage buckets (create if missing)

| Bucket | Access | File paths |
|--------|--------|-----------|
| `slides` | public | `{script_id}/slide_{NNN}.png` |
| `audio-files` | public | `{script_id}/audio_{N}.mp3` |
| `video-clips` | public | created automatically by the worker |

### Database tables (already exist)

- `script_chunks` — `id`, `script_id`, `chunk_index`, `content`, `slide_url`, `audio_url`
- `ocr_results` — upsert on `(script_id, chunk_id, slide_source)`
- `audio_timestamps` — upsert on `(script_id, chunk_id)`
- `clip_annotations` — upsert on `(script_id, chunk_id, slide_source)`
- `video_clips` — upsert on `(script_id, chunk_id, slide_source)`

---

## Deploy to Railway

### Option A — GitHub (recommended)

1. Push this folder as the **root** of a new GitHub repo:
   ```bash
   git init sky-annotations-worker
   cp -r . sky-annotations-worker/
   cd sky-annotations-worker
   git add . && git commit -m "init"
   git remote add origin https://github.com/YOU/sky-annotations-worker.git
   git push -u origin main
   ```

2. Go to **railway.app → New Project → Deploy from GitHub repo** → select the repo.

3. Railway auto-detects the `Dockerfile`. First build takes ~4 min (apt installs ffmpeg + tesseract).

4. In the Railway dashboard → **Variables**, add:
   ```
   SUPABASE_URL=https://xxx.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=eyJ...
   OPENAI_API_KEY=sk-...
   ```

5. After deploy, your public URL is shown in Railway → **Settings → Networking**.

### Option B — Railway CLI

```bash
npm install -g @railway/cli
railway login
railway init
railway up
railway variables set SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... OPENAI_API_KEY=...
```

---

## Run locally

```bash
# 1. Install Python 3.11 + system deps (macOS)
brew install ffmpeg tesseract

# 2. Create virtualenv
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Set env vars
cp .env.example .env
# Edit .env with your real values

# 5. Start server
uvicorn main:app --reload --port 8000
```

---

## Test with curl

```bash
BASE=http://localhost:8000

# Health check
curl $BASE/health

# Run OCR for one chunk
curl -X POST $BASE/ocr/run \
  -H "Content-Type: application/json" \
  -d '{"script_id":"<uuid>","chunk_id":"<uuid>","chunk_number":1,"slide_source":"gamma"}'

# Run Whisper timestamps
curl -X POST $BASE/timestamps/run \
  -H "Content-Type: application/json" \
  -d '{"script_id":"<uuid>","chunk_id":"<uuid>","chunk_number":1}'

# Generate AI annotations
curl -X POST $BASE/ai/run \
  -H "Content-Type: application/json" \
  -d '{"script_id":"<uuid>","chunk_id":"<uuid>","chunk_number":1,"slide_source":"gamma"}'

# Render clip (returns immediately, renders in background)
curl -X POST $BASE/clips/render \
  -H "Content-Type: application/json" \
  -d '{"script_id":"<uuid>","chunk_id":"<uuid>","chunk_number":1,"slide_source":"gamma"}'

# Run all-at-once (background jobs)
curl -X POST $BASE/ocr/run-all      -H "Content-Type: application/json" -d '{"script_id":"<uuid>","slide_source":"gamma"}'
curl -X POST $BASE/timestamps/run-all -H "Content-Type: application/json" -d '{"script_id":"<uuid>"}'
curl -X POST $BASE/ai/run-all       -H "Content-Type: application/json" -d '{"script_id":"<uuid>","slide_source":"gamma"}'
curl -X POST $BASE/clips/render-all -H "Content-Type: application/json" -d '{"script_id":"<uuid>","slide_source":"gamma"}'
```

## Run smoke tests

```bash
python test_endpoints.py                          # local
python test_endpoints.py https://your.railway.app # production
```

---

## File structure

```
sky-annotations-worker/
├── main.py                  # FastAPI app + all routes
├── workers/
│   ├── ocr.py               # Tesseract OCR
│   ├── timestamps.py        # OpenAI Whisper
│   ├── ai_annotations.py    # GPT-4o annotation generation
│   └── render.py            # FFmpeg + Pillow + CairoSVG renderer
├── lib/
│   ├── config.py            # Env var loader (fail-fast)
│   ├── supabase_client.py   # Singleton Supabase admin client
│   └── storage.py           # Download/upload helpers
├── requirements.txt
├── Dockerfile
├── railway.toml
├── .env.example
├── .gitignore
└── README.md                # (this file)
```
\n# Redeploy trigger: Wed Jun  3 21:24:29 UTC 2026
\n# Triggering Railway Redeploy: Wed Jun  3 21:24:41 UTC 2026
