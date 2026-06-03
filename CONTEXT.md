# SKY Studio: Complete Project Context (Autobiography)

## 1. Project Overview
SKY Studio is an autonomous AI-driven video production ecosystem designed for educational content creators, specifically targeting the Indian government exam preparation market (SSC, UPSC, RRB, Banking). It automates the entire journey from competitor monitoring and idea generation to script writing, audio synthesis, slide creation, and final video rendering with dynamic AI annotations.

### Key Mission
To allow a single operator to manage a high-frequency YouTube channel by offloading research, scripting, and post-production to a coordinated fleet of AI agents and specialized workers.

### Technical Stack
- **Frontend**: React (19.2), TanStack Router, TanStack Query, Tailwind CSS 4, Lucide Icons, Recharts.
- **Backend (API/Edge)**: Supabase (Auth, DB, Functions, Storage), Lovable (Vite/TanStack Start).
- **Railway Worker**: Python (3.11) FastAPI service handling heavy compute (FFmpeg, OCR, Whisper, CairoSVG).
- **AI Models**:
  - **Scripting/Analysis**: GPT-4o, Gemini 2.5 Pro.
  - **Vision**: Google Cloud Vision, Tesseract OCR.
  - **Audio**: ElevenLabs (V3), Google TTS (Gemini), Cartesia (Sonic 3.5).
  - **Transcripts**: Apify (YouTube Transcript Summary Actor).
- **Infrastructure**: Supabase Cloud, Railway (Compute Worker), GitHub (Source Control), Lovable Cloud.

---

## 2. Full Project Structure

### Repository Layout
- `/src`: Frontend React application.
  - `/routes`: TanStack Router file-based routes (the "Pages" of the app).
  - `/lib`: API clients, configuration, and shared business logic.
  - `/components`: UI components, including specialized views like `IdeaCardView` and `WatchdogControl`.
  - `/integrations/supabase`: Auto-generated Supabase client and types.
- `/railway-worker`: Independent Python service for media processing.
  - `main.py`: FastAPI entry point and route definitions.
  - `/workers`: Specialized processing modules (OCR, Timestamps, AI Annotations, Render).
  - `/lib`: Internal helpers for Supabase connection, storage, and config.
  - `Dockerfile` & `railway.toml`: Deployment configuration for Railway.
- `/supabase`: Backend configuration.
  - `/functions`: Deno Edge Functions (Audio gen, Script gen, Queue processing).
  - `/migrations`: PostgreSQL schema and RLS policy history.

### Entry Points
- **Web App**: `src/main.tsx` → `src/router.tsx`.
- **Worker**: `railway-worker/main.py` (FastAPI).
- **Edge**: `supabase/functions/process-queue/index.ts` (The "Brain" that orchestrates background jobs).

---

## 3. Full Feature Map

### 1. Ideas Engine (Competitor Watchdog)
- **Purpose**: Automatically discovers trending topics from competitor YouTube channels.
- **Implementation**: `src/lib/engine.functions.ts` (`runIdeaEngine`).
- **Input**: YouTube Channel URLs (configured in `Sources Master`).
- **Process**: RSS scraping → Apify transcription → Gemini analysis/proposal generation.
- **Output**: Entries in `raw_content` table with "Pending" status.

### 2. Script Generator
- **Purpose**: Generates full-length educational scripts from transcripts, PDFs, or topics.
- **Implementation**: `src/routes/_dashboard.script-generator.tsx` & `supabase/functions/generate-script-stream`.
- **Features**: Streaming AI generation, integrated Fact-Checker (wrong statement detection), word count targeting.
- **Input**: Transcript text, PDF files, or Idea Card context.
- **Output**: Saved record in `scripts` table.

### 3. Audio Engine
- **Purpose**: Synthesizes human-like voiceovers for script segments.
- **Implementation**: `src/routes/_dashboard.audio.tsx` & `supabase/functions/generate-audio`.
- **Providers**: ElevenLabs (High quality), Google (Fast), Cartesia (Ultra-low latency).
- **Output**: MP3 files stored in `audio-files` bucket; URLs saved in `script_chunks`.

### 4. Slide Maker
- **Purpose**: Creates visual slides using Gamma.app or DALL-E based on script context.
- **Implementation**: `src/routes/_dashboard.slides.tsx` & `supabase/functions/generate-slides`.
- **Process**: GPT-4o generates a slide outline → Gamma API creates the slide deck.
- **Output**: PNG images stored in `slides` bucket.

### 5. Annotation & Render Pipeline (The "Worker")
- **Purpose**: Adds dynamic visual cues (underlines, circles) to slides and renders video clips.
- **Implementation**: `railway-worker/` (Python/FFmpeg).
- **Process**:
  1. **OCR**: Detects word positions on slides (Tesseract).
  2. **Timestamps**: Aligns words to audio (Whisper).
  3. **AI Logic**: GPT-4o chooses what to emphasize (Strictly: 60% Underlines, 40% Circles).
  4. **Render**: FFmpeg composites SVG strokes onto PNG frames + MP3 audio.
- **Output**: MP4 clips in `video-clips` bucket.

---

## 4. Full App/Screen Map

### Navigation Sidebar (`src/routes/_dashboard.tsx`)
- Contains the main navigation groups: PIPELINE and UTILITIES.
- Shows storage usage and "DNA Active" status.

### Pages & Routes
- **Dashboard** (`/dashboard`): System health, production velocity charts, and database status.
- **Ideas Engine** (`/ideas-engine`): Configure YouTube sources and trigger manual "Scrapes".
- **Idea Cards** (`/idea-cards`): The "Inbox" for new ideas.
  - Tab: [Pending] → [Approve] → Moves to [Approved].
  - Tab: [Approved] → [Generate] → Navigates to Scripting.
- **Scripting** (`/script-generator`): The primary editor for content generation.
  - Button: [Generate Script] → Triggers AI Stream.
  - Button: [Fact Check] → Highlights errors in red.
  - Button: [Shift to Script Done] → Saves and advances pipeline.
- **Chunks** (`/chunks`): Splits long scripts into ~180-word segments for processing.
- **Audio** (`/audio`): Voice selection and batch generation for all chunks.
- **Slides** (`/slides`): Theme selection and automated Gamma slide creation.
- **Annotations** (`/annotations`): The multi-step pipeline for OCR, Timestamps, and AI logic.
- **Master Video** (`/master-video`): Library of all completed full-length video renders.
- **YouTube** (`/youtube`): SEO pack generator (Titles, Tags, Description) and Thumbnail AI.

---

## 5. System Details

### Data Schema (PostgreSQL)
- `sources_master`: YouTube channels to monitor.
- `raw_content`: Raw ideas scraped from competitors.
- `scripts`: Full generated script text and metadata.
- `script_chunks`: Segments of scripts with associated audio/slide URLs.
- `ocr_results`: JSON of word bounding boxes on slides.
- `audio_timestamps`: Word-level timing for audio files.
- `clip_annotations`: GPT-generated instructions for drawing (Underline/Circle).
- `video_clips`: Metadata for rendered MP4 segments.
- `app_settings`: Global config (Watchdog intervals, default models).

### Storage Buckets
- `slides`: PNG slide images.
- `audio-files`: MP3 voiceovers.
- `video-clips`: MP4 rendered segments.
- `raw-scripts`: (Optional) PDF/Document uploads.

---

## 6. Pipeline/Workflow Step-by-Step

1. **Watchdog**: Scrapes YouTube → Creates **Idea Cards**.
2. **User**: Approves Idea Card → Moves to **Scripting**.
3. **AI**: Generates **Script** → User **Fact Checks** → User saves as **Script Done**.
4. **User**: **Chunks** script → Script is split into `script_chunks`.
5. **AI**: Generates **Audio** for each chunk (ElevenLabs/Google).
6. **AI**: Generates **Slides** for each chunk (Gamma).
7. **Worker (Railway)**:
   - **OCR**: Detects word coords on slide.
   - **Timestamps**: Matches coords to audio timing.
   - **AI Annotations**: Decides where to draw.
   - **Render**: Combines all into a `.mp4` clip.
8. **Worker**: **Merges** all chunk clips into one **Master Video**.
9. **AI**: Generates **SEO Pack** & **Thumbnail** for YouTube upload.

---

## 7. Screenshot/Output Reference Index

1. **Dashboard Overview**: Main stats, production velocity chart.
2. **Sources Master**: Table of monitored YouTube channels.
3. **Idea Card (Pending)**: Thumbnail from YouTube + AI proposal.
4. **Idea Card (Processing)**: Shows checkmarks as transcript/analysis finish.
5. **Script Editor (Empty)**: Input fields for topic/instructions.
6. **Script Editor (Generating)**: Streaming text appearing in real-time.
7. **Fact Check View**: Red highlights on disputed statements + corrections.
8. **Chunking Engine**: Slider for word count + segmented cards.
9. **Audio Settings**: Provider selection (ElevenLabs vs Cartesia).
10. **Audio Progress**: Circular indicators on each chunk.
11. **Slide Maker (Outline)**: AI-generated slide contents for Gamma.
12. **Gamma Preview**: Thumbnail of the generated slide.
13. **Annotation Pipeline**: Step-by-step buttons (OCR → TS → AI → Render).
14. **Annotation JSON**: `{"type": "underline", "start_time": 4.5, ...}`.
15. **FFmpeg Logs**: Terminal output from the Railway worker.
16. **Master Video Library**: Gallery of completed full videos.
17. **YouTube SEO**: Grid of 5 title variations + long description.
18. **Thumbnail Generator**: DALL-E generated background + text overlays.
19. **Watchdog Control**: Toggle for auto-run + frequency settings.
20. **Storage Browser**: List of generated assets in Supabase.

---

## 8. How to Rebuild This Project From Scratch

### 1. Setup Environment
- Initialize a **TanStack Start** (Vite) project.
- Connect to **Supabase** for DB and Auth.
- Setup a **Railway** service with the provided `Dockerfile` for the Python worker.

### 2. Infrastructure
- Create the PostgreSQL tables using the provided migrations.
- Enable RLS policies.
- Create Supabase Storage buckets: `slides`, `audio-files`, `video-clips` (Set to Public).

### 3. Service Integration
- Add API Keys to Supabase/Railway secrets:
  - `OPENAI_API_KEY`
  - `GOOGLE_API_KEY` (Gemini & Vision)
  - `ELEVEN_LABS_API_KEY`
  - `APIFY_API_TOKEN`
  - `GAMMA_API_KEY`
  - `CARTESIA_API_KEY`

### 4. Build Order
1. **The Brain**: Deploy `process-queue` edge function to orchestrate status changes.
2. **Core App**: Build the Dashboard and Idea Engine.
3. **The Worker**: Deploy the Railway FastAPI service; verify `/health`.
4. **The Editor**: Build the Script Generator with streaming support.
5. **The Pipeline**: Connect Chunks → Audio → Slides → Annotations in sequence.

### 5. Testing Checklist
- [ ] Watchdog successfully scrapes a YouTube channel and creates a card.
- [ ] AI Script generator streams at least 600 words.
- [ ] Audio synthesizes and saves to storage.
- [ ] Railway worker successfully returns OCR bounding boxes.
- [ ] FFmpeg renders a 1080p clip with moving blue underlines.
- [ ] Master Video merge correctly combines 3+ clips.
