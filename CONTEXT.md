# SKY Studio: Complete Project Context (Autobiography v6.0)

## 1. Project Overview
SKY Studio is an autonomous AI-driven video production ecosystem designed for educational content creators, specifically targeting the Indian government exam preparation market (SSC, UPSC, RRB, Banking). It automates the entire journey from competitor monitoring and idea generation to script writing, audio synthesis, slide creation, and final video rendering with dynamic AI annotations.

### Key Mission
To allow a single operator to manage a high-frequency YouTube channel by offloading research, scripting, and post-production to a coordinated fleet of AI agents and specialized workers.

### Technical Stack
- **Frontend**: React (19.2), TanStack Router, TanStack Query, Tailwind CSS 4, Lucide Icons, Recharts.
- **Backend (API/Edge)**: Supabase (Auth, DB, Functions, Storage), Lovable (Vite/TanStack Start).
- **Railway Worker**: Python (3.11) FastAPI service handling heavy compute (FFmpeg, OCR, Whisper, CairoSVG).
- **AI Models**:
  - **Scripting/Analysis**: GPT-4o, Gemini 3.1 Pro.
  - **Vision**: Google Cloud Vision, Tesseract OCR.
  - **Audio**: ElevenLabs (V3), Google TTS (Gemini), Cartesia (Sonic 3.5).
  - **Transcripts**: Apify (YouTube Transcript Summary Actor).
- **Infrastructure**: Supabase Cloud, Railway (Compute Worker), GitHub (Source Control), Lovable Cloud.

---

## 2. Full Project Structure

### Repository Layout
- `/src`: Frontend React application.
- `/railway-worker`: Independent Python service for media processing.
- `/supabase`: Backend configuration.
- `/docs`: Project documentation and screenshots.
  - `UI_SPEC.md`: Canonical description of the UI (v6.0 updated).
  - `/screenshots/v6`: Latest visual captures of the system.

---

## 3. Full Feature Map

### 1. Ideas Engine (Competitor Watchdog)
- **Purpose**: Automatically discovers trending topics from competitor YouTube channels.
- **Implementation**: `src/lib/engine.functions.ts`.
- **Process**: RSS scraping → Apify transcription → Gemini analysis/proposal generation.

### 2. Script Generator
- **Purpose**: Generates full-length educational scripts from transcripts, PDFs, or topics.
- **Features**: Streaming AI generation, integrated Fact-Checker, word count targeting.

### 3. Audio Engine
- **Purpose**: Synthesizes human-like voiceovers for script segments.
- **Providers**: ElevenLabs (v3), Google AI Studio (Gemini 2.5), Cartesia (Sonic 3.5).

### 4. Slide Maker
- **Purpose**: Creates visual slides using Gamma.app or DALL-E.

### 5. Annotation & Render Pipeline (The "Worker")
- **Purpose**: Adds dynamic visual cues (underlines, circles) and renders clips.
- **Latest Fix**: Enforced strict schema in `ai_annotations.py` to ensure only `underline` and `circle` types are used.

---

## 4. Full App/Screen Map

### Navigation Sidebar (`src/routes/_dashboard.tsx`)
- Sidebar groups: PIPELINE and UTILITIES.
- Shows storage usage and "DNA Active" status.

### Pages & Routes
- **Dashboard** (`/dashboard`): System health and velocity.
- **Ideas Engine** (`/ideas-engine`): Source configuration.
- **Idea Cards** (`/idea-cards`): Triage inbox.
- **Scripting** (`/script-generator`): Script editor.
- **Audio** (`/audio`): TTS management.
- **Slides** (`/slides`): Visual creation.
- **Annotations** (`/annotations`): The multi-step production pipeline.
- **Master Video** (`/master-video`): Final assembly viewer.
- **YouTube** (`/youtube`): SEO and distribution tools.
- **History** (`/history`): Assistant memory log.
- **Storage** (`/storage`): Asset manager.
- **Settings** (`/settings`): API and notification config.
- **Tables** (`/tables`): Raw data inspector.
- **Uploads** (`/uploads`): Reference file manager.
- **Pipeline Engine** (`/pipeline`): System status watchdog.
- **Hook Engine** (`/hook-generator`): Viral hook utility.

---

## 5. Deployment & Infrastructure
- **Railway**: Deploys the Python worker. The `railway.toml` defines the start command and memory limits.
- **Supabase**: Handles auth, database, storage, and edge functions.
- **GitHub**: Source of truth for the codebase, synced via GitHub API with PAT.
