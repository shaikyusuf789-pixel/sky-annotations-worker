"""
workers/render.py — FFmpeg clip renderer with per-frame Pillow + CairoSVG compositing.

Pipeline:
  1. Resize slide PNG to 1920×1080 (letterboxed, black bars)
  2. For each frame at 30fps, composite active annotation SVG strokes onto the slide
  3. Pipe raw RGBA frames into FFmpeg → MP4 (libx264 + AAC)

Annotation colors:
  underline        → #3b82f6
  double_underline → #0ea5e9
  circle           → #f59e0b
  box              → #10b981
  arrow            → #8b5cf6
"""

import io
import math
import random
import subprocess
import tempfile
import os
from typing import Any

import cairosvg
from PIL import Image

W, H, FPS = 1920, 1080, 30
# Annotations start drawing this many seconds BEFORE the spoken word so the
# visual lands in sync with the voice (compensates for ElevenLabs alignment
# bias + human perception lag). Override via env if needed.
ANNOTATION_LEAD = float(os.environ.get("ANNOTATION_LEAD_SECONDS", "0.8"))
# Minimum gap between the END of one annotation's draw animation and the
# START of the next, so the tutor visually "lifts the pen" before the next
# stroke. Prevents two circles/underlines being drawn simultaneously.
ANNOTATION_PEN_LIFT = float(os.environ.get("ANNOTATION_PEN_LIFT", "0.08"))


# Single-pen mode: ONE color per clip, chosen from slide background brightness.
# Dark slide → light pen, light slide → dark pen. Set at render time.
PEN_DARK  = "#111111"   # near-black ink for light slides
PEN_LIGHT = "#f5f5f5"   # near-white ink for dark slides

DRAW_SECONDS = {
    "underline":         0.8,
    "double_underline":  0.8,
    "circle":            1.1,
    "box":               1.2,
    "arrow":             0.9,
}


def _pick_pen(slide_rgba: "Image.Image") -> str:
    """Sample the slide, average brightness → pick black or white pen."""
    small = slide_rgba.convert("RGB").resize((40, 24), Image.BILINEAR)
    px = list(small.getdata())
    # Perceived luminance
    avg = sum(0.299 * r + 0.587 * g + 0.114 * b for (r, g, b) in px) / len(px)
    return PEN_LIGHT if avg < 110 else PEN_DARK



# ── Coordinate transform (OCR source → 1920×1080) ───────────────────────────

def _scale_bbox(
    bbox: list[int],
    src_w: int,
    src_h: int,
) -> tuple[int, int, int, int]:
    """Scale OCR bbox from source image dimensions to 1920×1080 letterboxed space."""
    scale   = min(W / src_w, H / src_h)
    off_x   = (W - src_w * scale) / 2
    off_y   = (H - src_h * scale) / 2
    x, y, w, h = bbox
    return (
        round(x * scale + off_x),
        round(y * scale + off_y),
        round(w * scale),
        round(h * scale),
    )


# ── SVG path builders ────────────────────────────────────────────────────────

def _eased(t: float) -> float:
    return t * t * (3 - 2 * t)


def _pts_to_path(pts: list[tuple[float, float]]) -> str:
    if not pts:
        return ""
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    # Quadratic-smoothed path so the hand wobble looks like ink, not zig-zags
    for i in range(1, len(pts)):
        x, y = pts[i]
        d += f" L{x:.1f},{y:.1f}"
    return d


def _rng(seed_key: str) -> random.Random:
    return random.Random(hash(seed_key) & 0xFFFFFFFF)


def _underline_pts(x: int, y_bottom: int, w: int, seed: str = "u") -> list[tuple[float, float]]:
    """Wavy, slightly sloped underline — tutor's marker stroke. Inset on both ends
    so it stays *under* the actual word run, never overshooting the text."""
    r = _rng(seed)
    # Inset ~6% (min 8px, max 28px) so the line never overruns text or crosses boxes
    inset = max(8, min(28, int(w * 0.06)))
    x_start = x + inset
    x_end = max(x_start + 20, x + w - inset)
    eff_w = x_end - x_start
    steps = max(30, eff_w // 8)
    slope = r.uniform(-0.008, 0.008)
    base_y = y_bottom + r.uniform(2, 5)
    amp = r.uniform(1.0, 2.0)
    freq = r.uniform(0.020, 0.040)
    phase = r.uniform(0, math.tau)
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        px = x_start + eff_w * i / steps
        wave = math.sin(phase + (px - x_start) * freq) * amp
        jitter = r.uniform(-0.6, 0.6)
        py = base_y + slope * (px - x_start) + wave + jitter
        pts.append((px, py))
    return pts


def _double_underline_pts(x: int, y_bottom: int, w: int, seed: str = "d") -> list[list[tuple[float, float]]]:
    line1 = _underline_pts(x, y_bottom, w, seed + "1")
    line2 = _underline_pts(x, y_bottom + 7, w, seed + "2")
    return [line1, line2]


def _circle_pts(cx: float, cy: float, rx: float, ry: float, seed: str = "c") -> list[tuple[float, float]]:
    """Hand-drawn loop — neat tutor pen circle: minimal tilt, small overshoot,
    light wobble, and enough padding so the loop sits OUTSIDE the text."""
    r = _rng(seed)
    n = 130
    start = -math.pi / 2 + r.uniform(-0.25, 0.25)
    sweep = 2 * math.pi + r.uniform(0.10, 0.25)  # gentle overshoot (~6–14°)
    tilt  = r.uniform(-0.05, 0.05)               # ~±3° — barely tilted
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    wob_amp_x = r.uniform(0.02, 0.04)
    wob_amp_y = r.uniform(0.02, 0.04)
    wob_freq_x = r.uniform(1.5, 2.5)
    wob_freq_y = r.uniform(1.5, 2.5)
    wob_phase_x = r.uniform(0, math.tau)
    wob_phase_y = r.uniform(0, math.tau)
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        a = start - sweep * t
        wx = 1 + wob_amp_x * math.sin(wob_phase_x + t * math.tau * wob_freq_x) + r.uniform(-0.012, 0.012)
        wy = 1 + wob_amp_y * math.sin(wob_phase_y + t * math.tau * wob_freq_y) + r.uniform(-0.012, 0.012)
        ex = rx * wx * math.cos(a)
        ey = ry * wy * math.sin(a)
        px = cx + ex * cos_t - ey * sin_t + r.uniform(-0.5, 0.5)
        py = cy + ex * sin_t + ey * cos_t + r.uniform(-0.5, 0.5)
        pts.append((px, py))
    return pts


def _box_pts(x: int, y: int, w: int, h: int, seed: str = "b") -> list[tuple[float, float]]:
    r = _rng(seed)
    def side(x1, y1, x2, y2, n=24):
        out = []
        for i in range(n + 1):
            t = i / n
            px = x1 + (x2 - x1) * t + r.uniform(-1.4, 1.4)
            py = y1 + (y2 - y1) * t + r.uniform(-1.4, 1.4)
            out.append((px, py))
        return out
    # Start a bit before top-left, close a bit past it (sketchy overshoot)
    sx = x - r.uniform(2, 6); sy = y - r.uniform(2, 6)
    return (side(sx, sy, x+w, y)
            + side(x+w, y, x+w, y+h)
            + side(x+w, y+h, x, y+h)
            + side(x, y+h, sx, sy))


def _arrow_pts(x: int, y_mid: int, seed: str = "a") -> list[list[tuple[float, float]]]:
    """Hand-drawn arrow with a slightly curved shaft + asymmetric arrowhead.
    Shaft length is clamped so the tail can't shoot across the slide into the
    opposite column."""
    r = _rng(seed)
    tip_x = x - 10 + r.uniform(-3, 3)
    tip_y = y_mid + r.uniform(-4, 4)
    length = r.uniform(70, 95)                  # tighter than before
    angle  = math.pi + r.uniform(-0.18, 0.18)   # mostly leftward, gentle tilt
    tail_x = tip_x + math.cos(angle) * length
    tail_y = tip_y + math.sin(angle) * length
    mid_x = (tip_x + tail_x) / 2 + r.uniform(-8, 8)
    mid_y = (tip_y + tail_y) / 2 + r.uniform(-10, 10)
    shaft = []
    steps = 48
    for i in range(steps + 1):
        t = i / steps
        # Quadratic Bezier: tail → mid → tip
        bx = (1 - t) ** 2 * tail_x + 2 * (1 - t) * t * mid_x + t ** 2 * tip_x
        by = (1 - t) ** 2 * tail_y + 2 * (1 - t) * t * mid_y + t ** 2 * tip_y
        shaft.append((bx + r.uniform(-0.6, 0.6), by + r.uniform(-0.6, 0.6)))
    # Arrowhead pointing along incoming shaft direction
    ang_in = math.atan2(tip_y - mid_y, tip_x - mid_x)
    head_len = r.uniform(18, 24)
    spread1 = r.uniform(0.45, 0.65)
    spread2 = r.uniform(0.45, 0.65)
    h1_end = (tip_x - head_len * math.cos(ang_in - spread1),
              tip_y - head_len * math.sin(ang_in - spread1))
    h2_end = (tip_x - head_len * math.cos(ang_in + spread2),
              tip_y - head_len * math.sin(ang_in + spread2))
    head1 = [(tip_x, tip_y), h1_end]
    head2 = [(tip_x, tip_y), h2_end]
    return [shaft, head1, head2]



# ── Frame SVG builder ────────────────────────────────────────────────────────

def _build_frame_svg(
    annotations: list[dict],
    progress_map: dict[int, float],  # ann_index → 0.0–1.0
    src_w: int,
    src_h: int,
    pen: str,
) -> str:
    paths_svg = []

    def _stroke(d: str, color: str, sw: float, opacity: float = 0.92) -> str:
        return (
            f'<path d="{d}" stroke="{color}" stroke-width="{sw:.1f}" '
            f'fill="none" stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="{opacity:.2f}"/>'
        )

    for idx, ann in enumerate(annotations):
        prog = progress_map.get(idx)
        if prog is None or prog <= 0:
            continue

        ann_type = ann["type"]
        # Single-pen mode: every annotation uses the same color for the whole clip.
        color    = pen
        x, y, w, h = _scale_bbox(ann["bbox"], src_w, src_h)
        seed = f"{idx}-{ann.get('target_text','')[:24]}"

        # Double underline removed by user request → render as a single underline.
        if ann_type in ("underline", "double_underline"):
            pts = _underline_pts(x, y + h, w, seed=seed)
            n   = max(2, round(len(pts) * prog))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 4.0, 0.85))

        elif ann_type == "circle":
            cx, cy = x + w / 2, y + h / 2
            rx, ry = w / 2 + 18, h / 2 + 14
            pts = _circle_pts(cx, cy, rx, ry, seed=seed)
            n   = max(2, round(len(pts) * prog))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 4.0, 0.9))

        elif ann_type == "box":
            pts = _box_pts(x - 6, y - 4, w + 12, h + 8, seed=seed)
            n   = max(2, round(len(pts) * prog))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 4.0, 0.85))

        elif ann_type == "arrow":
            shaft, h1, h2 = _arrow_pts(x, y + h // 2, seed=seed)
            n = max(2, round(len(shaft) * prog))
            paths_svg.append(_stroke(_pts_to_path(shaft[:n]), color, 4.0, 0.9))
            if prog > 0.7:
                head_prog = min(1.0, (prog - 0.7) / 0.3)
                # Draw arrowhead progressively too
                paths_svg.append(_stroke(_pts_to_path(h1), color, 4.0, 0.9 * head_prog))
                paths_svg.append(_stroke(_pts_to_path(h2), color, 4.0, 0.9 * head_prog))

    if not paths_svg:
        return ""

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
        + "".join(paths_svg)
        + "</svg>"
    )



# ── Audio duration ────────────────────────────────────────────────────────────

def get_audio_duration(path: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 30.0


# ── Main render function ──────────────────────────────────────────────────────

def render_clip(
    slide_path: str,
    audio_path: str,
    annotations: list[dict[str, Any]],
    output_path: str,
    ocr_src_w: int = 2400,
    ocr_src_h: int = 1350,
) -> float:
    """
    Render an annotated MP4 clip.

    Returns the clip duration in seconds.

    Parameters
    ----------
    slide_path   : local PNG path (any resolution — will be letterboxed to 1920×1080)
    audio_path   : local MP3/AAC/WAV path
    annotations  : list of annotation dicts (type, start_time, bbox, …)
    output_path  : where to write the MP4
    ocr_src_w/h  : resolution of the image OCR ran on (for correct bbox scaling)
    """
    audio_dur    = get_audio_duration(audio_path)
    total_frames = math.ceil(audio_dur * FPS)
    print(f"[RENDER] {total_frames} frames @ {FPS}fps, duration={audio_dur:.2f}s")

    # Resize slide to 1920×1080 letterboxed (black bars)
    raw_slide = Image.open(slide_path).convert("RGBA")
    bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    sw, sh = raw_slide.size
    scale  = min(W / sw, H / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    resized = raw_slide.resize((nw, nh), Image.LANCZOS)
    bg.paste(resized, ((W - nw) // 2, (H - nh) // 2))
    slide_rgba = bg
    slide_bytes = slide_rgba.tobytes()  # reuse for blank frames
    pen = _pick_pen(slide_rgba)
    print(f"[RENDER] pen color = {pen} (single-pen mode)")

    # Per-annotation draw duration
    draw_durations = [
        max(0.5, float(ann.get("draw_duration") or DRAW_SECONDS.get(ann["type"], 1.5)))
        for ann in annotations
    ]

    # ── Serialize annotation visual starts (pen-lift) ──────────────────────
    # A real tutor only draws one stroke at a time. We pre-compute the
    # effective on-screen start for each annotation by chronological order:
    # if its scheduled start would overlap the previous stroke's animation,
    # push it to (prev_effective_start + prev_draw_duration + PEN_LIFT).
    # Only the visual draw is serialized — original ordering / spoken-word
    # anchors are otherwise preserved.
    order = sorted(
        range(len(annotations)),
        key=lambda i: float(annotations[i].get("start_time") or 0),
    )
    effective_start: list[float] = [0.0] * len(annotations)
    prev_end = -1e9
    for i in order:
        scheduled = float(annotations[i].get("start_time") or 0) - ANNOTATION_LEAD
        if scheduled < 0:
            scheduled = 0.0
        start = max(scheduled, prev_end + ANNOTATION_PEN_LIFT)
        effective_start[i] = start
        prev_end = start + draw_durations[i]


    # Start FFmpeg in a 2-pass-friendly way: write raw frames to a temp file
    # first (avoids pipe-buffer/encoder back-pressure issues that have been
    # causing "flush of closed file" errors), then encode in one shot.
    import threading

    ff_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-y", output_path,
    ]
    print(f"[RENDER] ffmpeg: {' '.join(ff_cmd)}")

    ff = subprocess.Popen(
        ff_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Drain stderr in background so the pipe never blocks ffmpeg.
    stderr_chunks: list[bytes] = []
    def _pump_stderr() -> None:
        try:
            while True:
                chunk = ff.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        except Exception:
            pass
    t_err = threading.Thread(target=_pump_stderr, daemon=True)
    t_err.start()

    pipe_error: Exception | None = None
    last_frame = 0
    # Cache composited frame bytes keyed by a quantized progress signature.
    # Most frames are either "no annotations active" or "all active annotations
    # fully drawn" — both produce identical pixels for long stretches. Caching
    # avoids re-rasterizing SVG + alpha_composite every frame, which was the
    # source of OOM-kill (rc=-9, BrokenPipe) on Railway.
    frame_cache: dict[tuple, bytes] = {}
    MAX_CACHE = 48

    def _sig(pm: dict[int, float]) -> tuple:
        # Quantize to 2% steps so near-identical progress reuses cache
        return tuple(sorted((i, round(p * 50) / 50) for i, p in pm.items()))

    try:
        for f_idx in range(total_frames):
            last_frame = f_idx
            t = f_idx / FPS
            prog_map: dict[int, float] = {}
            for i, ann in enumerate(annotations):
                start = effective_start[i]
                dur   = draw_durations[i]
                if t < start:
                    continue
                elapsed = t - start
                prog_map[i] = 1.0 if elapsed >= dur else _eased(elapsed / dur)

            if not prog_map:
                ff.stdin.write(slide_bytes)
                continue

            key = _sig(prog_map)
            cached = frame_cache.get(key)
            if cached is not None:
                ff.stdin.write(cached)
                continue

            svg = _build_frame_svg(annotations, prog_map, ocr_src_w, ocr_src_h, pen)
            if not svg:
                frame_bytes = slide_bytes
            else:
                png_bytes = cairosvg.svg2png(
                    bytestring=svg.encode(),
                    output_width=W,
                    output_height=H,
                )
                overlay = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                frame = Image.alpha_composite(slide_rgba, overlay)
                frame_bytes = frame.tobytes()
                overlay.close()
                frame.close()
                del overlay, frame, png_bytes

            if len(frame_cache) < MAX_CACHE:
                frame_cache[key] = frame_bytes
            ff.stdin.write(frame_bytes)
    except (BrokenPipeError, ValueError, OSError) as pipe_err:
        pipe_error = pipe_err
        print(f"[RENDER] pipe write failed at frame {last_frame}/{total_frames}: {pipe_err!r}")

    try:
        ff.stdin.close()
    except Exception:
        pass

    try:
        rc = ff.wait(timeout=300)
    except subprocess.TimeoutExpired:
        ff.kill()
        rc = ff.wait()
        stderr_chunks.append(b"\n<ffmpeg killed: wait timed out>")

    t_err.join(timeout=5)
    ff_err = b"".join(stderr_chunks)[-2000:].decode(errors="replace")

    if pipe_error is not None or rc != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={rc}, frame={last_frame}/{total_frames}, "
            f"pipe_error={pipe_error!r}); stderr: {ff_err or '<empty>'}"
        )


    print(f"[RENDER] done → {output_path}")
    return audio_dur
