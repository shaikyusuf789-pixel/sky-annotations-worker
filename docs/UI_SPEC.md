# Sky Studio — Full UI & System Biography (v6.0)

> **Purpose.** Canonical, pixel-and-byte description of Sky Studio. Hand this single file + `docs/backup/` SQL to another AI builder (Lovable, Replit, Bolt, Cursor, v0) and they MUST be able to rebuild a **1:1 replica**: same routes, same layout, same colors, same wiring, same secrets contract, same database.
>
> **Last full refresh:** 2026-06-03. Screenshots live in `docs/screenshots/v6/`.
>
> **What changed since v5.1:**
> - 🚀 **Audio Engine Upgrade:** Integrated Cartesia Sonic 3.5 and Google Gemini TTS models.
> - ⚡ **Strict Annotation Schema:** `ai_annotations.py` now enforces a strict JSON schema for `underline` and `circle` types only, eliminating "phantom arrows".
> - 🆕 **Unified Pipeline Watchdog:** Automated competitor monitoring and content generation (Ideas -> Scripts -> Audio -> Slides).
> - 🆕 **Hook Engine:** New utility for generating viral hooks for content.
> - 🆕 **ElevenLabs Forced Alignment:** Fully operational as a TanStack server function.

---

## 0. Quick Migration Checklist (read first)

1. **Stack.** TanStack Start v1 + Vite 7 + React 19 + Tailwind CSS v4 + Lucide + shadcn/ui (Radix). TypeScript strict.
2. **Routing.** File-based under `src/routes/`. Root `__root.tsx`, dashboard shell `_dashboard.tsx`, leaves `_dashboard.<name>.tsx`.
3. **Database cluster.** Direct Supabase project. Full schema in `supabase/migrations/`.
4. **Auth.** Single-user / permissive RLS for now. All Supabase access uses publishable + service-role keys.
5. **Secrets.** Stored in Supabase Edge Function secrets. Never in the client bundle.
6. **Server logic.** TanStack `createServerFn` for OCR / Timestamps / Script / Audio / Slides / Engine. 
7. **External worker.** A separate Railway Python worker handles AI annotations, ffmpeg clip rendering, and mega-merging. Repo: `shaikyusuf789-pixel/sky-annotations-worker`.
8. **Storage buckets.** `slides`, `audio-files`, `user-uploads`, `video-clips` — all public.
9. **Theme.** Light mode only. Primary blue `oklch(0.55 0.22 257)`. 
10. **Mobile-first.** Every page is responsive and usable at 390px.

---

## 1. Identity, Brand, Theme

- **Product name:** Sky Studio (sidebar header reads `SKY Studio / AI VIDEO BOT V4.2`).
- **Worker Version:** `2026-06-03.elevenlabs-forced-alignment-018-strict-schema`.
- **Tagline:** "Your AI-driven content command center."
- **Logo mark:** Rounded-square gradient tile, text `SKY` in white 700-weight.
- **Color tokens (`src/styles.css`):**
  | Token | Light value | Use |
  |---|---|---|
  | `--primary` | `oklch(0.55 0.22 257)` | CTAs, active nav |
  | `--accent` | `oklch(0.7 0.15 220)` | Highlights |
  | `--background` | `oklch(0.99 0 0)` | App bg |

---

## 2. Global Layout

### 2.1 Root (`src/routes/__root.tsx`)
- Wraps everything in `QueryClientProvider` (TanStack Query).
- Provides `<Toaster />` (sonner) at root.

### 2.2 Dashboard layout (`src/routes/_dashboard.tsx`)
Two-column shell:
- **Left sidebar** (fixed 256px on ≥ md, hidden on mobile):
  - Header: logo tile + `SKY Studio / AI VIDEO BOT V4.2`.
  - **PIPELINE**: Dashboard · Ideas Engine · Idea Cards · 1 Scripting · 2 Chunks · 3 Audio · 4 Slides · 5 Annotations · 6 Master Video · 7 YouTube.
  - **UTILITIES**: History · Storage · Settings.
  - Footer: "STORAGE USAGE 65%" progress bar.
- **Main column**:
  - Top header: hamburger (mobile) · centered search · notification bell.
  - Page content with 24-32 px padding.

### 2.3 Mobile menu (Sheet)
Hamburger → slides in from left → identical nav contents.
![Mobile menu](screenshots/v6/23-mobile-menu.png)

---

## 3. Page Catalogue

### 3.1 Dashboard — `/dashboard`
![Dashboard desktop](screenshots/v6/01-dashboard.png)
![Dashboard mobile](screenshots/v6/02-dashboard-mobile.png)

- **Purpose:** Command center showing pipeline health.
- **Sections:**
  - Hero: "Sky Studio" H1, **Full Backup** button.
  - **Status Tiles**: Total Ideas, Pending Approval, Pending Priority, Pending Scripting, Pending Audio, Pending Slides.
  - **Production Velocity** area chart.
  - **Status Distribution** bar chart.
  - **Active Database Clusters** grid: links to `/tables`.
  - **Production Workflow** card.
- **Wiring:** Uses `getDashboardCounts` and `runFullBackup`.

### 3.2 Ideas Engine — `/ideas-engine`
![Ideas Engine](screenshots/v6/03-ideas-engine.png)

- **Purpose:** Configure YouTube channels to monitor.
- **AI Engine Card:** Channel Name + YouTube URL → **Sync to Source Master**.
- **Manual Entry:** Register single sources.
- **Bottom:** Source Master Table.
- **Wiring:** `addSource`, `runIdeaEngine`.

### 3.3 Idea Cards — `/idea-cards`
![Idea Cards](screenshots/v6/04-idea-cards.png)
![Idea Cards Populated](screenshots/v6/24-idea-cards-populated.png)

- **Purpose:** Triage scraped ideas. Tabs: **Pending · Approved · Priority**.
- **Actions:** Approve / Reject / Priority / Generate Script.
- **Wiring:** Updates `raw_content.status`.

### 3.4 Script Generator — `/script-generator`
![Script Generator](screenshots/v6/05-script-generator.png)
![Script Generator Dropdown](screenshots/v6/20-script-generator-dropdown.png)
![Script Generator Mobile](screenshots/v6/33-script-generator-mobile.png)

- **Phase 1 — Scripting DNA.**
- **Settings:**
  - Provider: Sky Studio Gemini (default), OpenAI.
  - Model: Gemini 3.1 Pro, GPT-4o.
  - Video Type: Subjective vs General.
  - Input Mode: Priority List, Topic Name, Competitor Transcripts, Book/PDF.
  - Approximate Total Script Words slider.
- **Wiring:** `generateScript` server function.

### 3.5 Chunks — `/chunks`
![Chunks](screenshots/v6/06-chunks.png)

- **Phase 2 — Segmentation.**
- **Settings:** Select Script, Words per chunk slider (80-300).
- **Wiring:** `chunkScript` → splits `scripts` into `script_chunks`.

### 3.6 Audio — `/audio`
![Audio](screenshots/v6/07-audio.png)

- **Purpose:** Per-chunk TTS voiceovers.
- **Settings:**
  - Provider: Google AI Studio, Cartesia, ElevenLabs.
  - TTS Model: gemini-2.5-pro-preview-tts, sonic-3-latest, eleven_v3.
  - Voice: Extensive list (Zephyr, Puck, etc.).
- **Actions:** Generate All, Merge & Download.
- **Wiring:** `generateAudio`, `mergeAudio`.

### 3.7 Slides — `/slides`
![Slides](screenshots/v6/08-slides.png)

- **Phase 2 — Slide Maker.**
- **Settings:** Select Script, Model (Oasis, etc.).
- **Actions:** Generate All Outlines, Generate All Slides.
- **Wiring:** `generateSlideOutline`, `generateGammaSlide`.

### 3.8 Annotations — `/annotations`
![Annotations Empty](screenshots/v6/28-annotations-empty.png)
![Annotations Dropdown](screenshots/v6/21-annotations-dropdown.png)
![Annotations Populated](screenshots/v6/22-annotations-populated.png)
![Annotations Expanded](screenshots/v6/29-annotations-chunk-expanded.png)
![Annotations Mobile](screenshots/v6/32-annotations-mobile.png)

- **Phase 6 — Annotation Pipeline.**
- **Progress bar:** OCR → Timestamps → AI Annotations → Render Clips → Merge Mega Video.
- **Bulk Actions:** All OCR, All Timestamps, All AI, Render All (and Skip & Run versions).
- **Per-chunk View:**
  - Original Script snippet.
  - Slide preview.
  - OCR Output (Google Vision).
  - Timestamps (ElevenLabs FA).
  - AI Annotations (Strict JSON).
  - Final Clip preview.
- **Mega Video:** Merged 4K MP4 assembly.
- **Wiring:** Calls `runOcr`, `runTimestamps`, Railway `/ai/run`, Railway `/clips/render`, Railway `/merge/run`.

### 3.9 Master Video — `/master-video`
![Master Video](screenshots/v6/10-master-video.png)

- **Phase 6 — Assembly.** Large preview of final assembly.
- **Wiring:** Displays list of mega videos from `master_videos` table.

### 3.10 YouTube Studio — `/youtube`
![YouTube](screenshots/v6/11-youtube.png)

- **Phase 4 — Distribution.** SEO tools for metadata generation.
- **Wiring:** Writes to `youtube_seo` table.

### 3.11 History — `/history`
![History](screenshots/v6/12-history.png)

- **Purpose:** Persistent log of the AI assistant's memory.
- **Wiring:** Reads/writes `ai_chat_memory`.

### 3.12 Storage — `/storage`
![Storage](screenshots/v6/13-storage.png)
![Storage Detail](screenshots/v6/30-storage-biography-details.png)

- **Purpose:** Unified asset manager for buckets: `slides`, `audio-files`, `user-uploads`, `video-clips`.
- **Wiring:** Direct Supabase storage interaction.

### 3.13 Settings — `/settings`
![Settings](screenshots/v6/14-settings.png)
![Settings Full](screenshots/v6/31-settings-full.png)

- **API Credentials:** Masked fields for API keys (managed via Supabase Secrets).
- **Notifications:** Email Alerts, Telegram Bot.

### 3.14 Tables — `/tables`
![Tables](screenshots/v6/15-tables.png)
![Tables Full](screenshots/v6/25-tables-full.png)

- **Purpose:** Raw data inspector for all 9+ public tables.
- **Wiring:** CRUD operations on Supabase tables.

### 3.15 Uploads — `/uploads`
![Uploads](screenshots/v6/16-uploads.png)

- **Purpose:** Grounding documents (PDFs, transcripts).

### 3.16 Pipeline Engine — `/pipeline`
![Pipeline](screenshots/v6/17-pipeline.png)

- **Purpose:** Live status of the entire chain.
- **Wiring:** Monitors `app_settings` and `last_run` timestamps.

### 3.17 Content Preview — `/content-preview`
![Content Preview](screenshots/v6/18-content-preview.png)

- **Purpose:** Flat tabular overview of all ideas.

### 3.18 Hook Engine — `/hook-generator`
![Hook Engine](screenshots/v6/19-hook-generator.png)

- **Purpose:** Generate viral hooks for scripts.

---

## 4. System Architecture

### 4.1 Frontend
- **Framework:** TanStack Start v1.
- **State:** TanStack Query + Supabase Realtime.
- **Styling:** Tailwind CSS v4 + Lucide icons.

### 4.2 Backend (Supabase)
- **Database:** PostgreSQL.
- **Edge Functions:** `process-queue`, `process-idea`, `auth-email-hook`.
- **Storage:** S3-compatible buckets.

### 4.3 Backend (Railway Worker)
- **Language:** Python (FastAPI).
- **Core Jobs:** FFmpeg composition, Pillow image manipulation, Annotation generation.
- **Optimizations:** Memory diet (half-res compositing, aggressive GC).

---

## 5. Secret Contract (Supabase Secrets)

| Key | Use |
|---|---|
| `OPENAI_API_KEY` | GPT-4o scripting, summaries |
| `ELEVEN_LABS_API_KEY` | TTS, Forced Alignment |
| `GOOGLE_API_KEY` | Gemini Pro scripting, Gemini TTS, Google Vision OCR |
| `GAMMA_API_KEY` | Slide generation |
| `APIFY_API_TOKEN` | YouTube scraping |
| `CARTESIA_API_KEY` | High-speed TTS |

---

## 6. Testing Checklist

- [ ] Sidebar collapses/expands correctly.
- [ ] Idea triage updates status in realtime.
- [ ] Script generation writes full text to DB.
- [ ] Audio generation queues jobs in Supabase.
- [ ] Annotation pipeline runs OCR/TS/AI/Render in sequence.
- [ ] Master video merge combines clips into 4K MP4.
- [ ] Storage bucket uploads/downloads work.
