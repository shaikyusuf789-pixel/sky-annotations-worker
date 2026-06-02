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

# Warmer "marker" palette — feels more like a tutor's highlighter than a UI accent
STROKE_COLORS = {
    "underline":         "#ffd54a",  # marker yellow
    "double_underline":  "#ff7043",  # orange-red (heading)
    "circle":            "#ff5252",  # red ink
    "box":               "#26c6da",  # cyan ink
    "arrow":             "#ab47bc",  # purple ink
}

DRAW_SECONDS = {
    "underline":         0.9,
    "double_underline":  1.4,
    "circle":            1.6,
    "box":               1.8,
    "arrow":             1.0,
}



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
    """Wavy, slightly sloped underline — looks like a tutor's marker stroke."""
    r = _rng(seed)
    steps = max(40, w // 6)
    # Random tiny slope and vertical offset
    slope = r.uniform(-0.012, 0.012)
    base_y = y_bottom + r.uniform(3, 8)
    amp = r.uniform(1.6, 3.2)
    freq = r.uniform(0.018, 0.035)
    phase = r.uniform(0, math.tau)
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        px = x + w * i / steps
        wave = math.sin(phase + (px - x) * freq) * amp
        jitter = r.uniform(-0.9, 0.9)
        py = base_y + slope * (px - x) + wave + jitter
        pts.append((px, py))
    return pts


def _double_underline_pts(x: int, y_bottom: int, w: int, seed: str = "d") -> list[list[tuple[float, float]]]:
    line1 = _underline_pts(x, y_bottom, w, seed + "1")
    line2 = _underline_pts(x, y_bottom + 9, w, seed + "2")
    return [line1, line2]


def _circle_pts(cx: float, cy: float, rx: float, ry: float, seed: str = "c") -> list[tuple[float, float]]:
    """Hand-drawn loop — slightly overshoots, varies radius."""
    r = _rng(seed)
    n = 110
    start = -math.pi / 2 + r.uniform(-0.2, 0.2)
    sweep = 2 * math.pi + r.uniform(0.1, 0.45)  # overshoot
    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        a = start - sweep * t
        # Wobble radius
        wob = 1 + 0.04 * math.sin(t * 6.0 + r.random() * 2) + r.uniform(-0.015, 0.015)
        px = cx + rx * wob * math.cos(a)
        py = cy + ry * wob * math.sin(a)
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
    r = _rng(seed)
    tip_x = x - 8 + r.uniform(-2, 2)
    tip_y = y_mid + r.uniform(-3, 3)
    tail_x = tip_x - 80
    tail_y = tip_y + r.uniform(-6, 6)
    shaft = []
    for i in range(40):
        t = i / 39
        px = tail_x + (tip_x - tail_x) * t + r.uniform(-1.0, 1.0)
        py = tail_y + (tip_y - tail_y) * t + math.sin(t * 4) * 1.2
        shaft.append((px, py))
    head1 = [(tip_x, tip_y), (tip_x - 18 + r.uniform(-2,2), tip_y - 12 + r.uniform(-2,2))]
    head2 = [(tip_x, tip_y), (tip_x - 18 + r.uniform(-2,2), tip_y + 12 + r.uniform(-2,2))]
    return [shaft, head1, head2]



# ── Frame SVG builder ────────────────────────────────────────────────────────

def _build_frame_svg(
    annotations: list[dict],
    progress_map: dict[int, float],  # ann_index → 0.0–1.0
    src_w: int,
    src_h: int,
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
        color    = STROKE_COLORS.get(ann_type, "#ffd54a")
        x, y, w, h = _scale_bbox(ann["bbox"], src_w, src_h)
        seed = f"{idx}-{ann.get('target_text','')[:24]}"

        if ann_type == "underline":
            pts = _underline_pts(x, y + h, w, seed=seed)
            n   = max(2, round(len(pts) * prog))
            # Two slightly offset passes → marker ink texture
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 7.0, 0.85))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 3.5, 0.55))

        elif ann_type == "double_underline":
            for li, line_pts in enumerate(_double_underline_pts(x, y + h, w, seed=seed)):
                n = max(2, round(len(line_pts) * prog))
                paths_svg.append(_stroke(_pts_to_path(line_pts[:n]), color, 6.0, 0.9))
                paths_svg.append(_stroke(_pts_to_path(line_pts[:n]), color, 2.8, 0.5))

        elif ann_type == "circle":
            cx, cy = x + w / 2, y + h / 2
            rx, ry = w / 2 + 18, h / 2 + 14
            pts = _circle_pts(cx, cy, rx, ry, seed=seed)
            n   = max(2, round(len(pts) * prog))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 6.5, 0.9))

        elif ann_type == "box":
            pts = _box_pts(x - 8, y - 6, w + 16, h + 12, seed=seed)
            n   = max(2, round(len(pts) * prog))
            paths_svg.append(_stroke(_pts_to_path(pts[:n]), color, 6.0, 0.9))

        elif ann_type == "arrow":
            shaft, h1, h2 = _arrow_pts(x, y + h // 2, seed=seed)
            n = max(2, round(len(shaft) * prog))
            paths_svg.append(_stroke(_pts_to_path(shaft[:n]), color, 6.0, 0.9))
            if prog > 0.85:
                paths_svg.append(_stroke(_pts_to_path(h1), color, 6.0, 0.9))
                paths_svg.append(_stroke(_pts_to_path(h2), color, 6.0, 0.9))

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

    # Per-annotation draw duration
    draw_durations = [
        max(0.5, float(ann.get("draw_duration") or DRAW_SECONDS.get(ann["type"], 1.5)))
        for ann in annotations
    ]

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
    try:
        for f_idx in range(total_frames):
            last_frame = f_idx
            t = f_idx / FPS
            prog_map: dict[int, float] = {}
            for i, ann in enumerate(annotations):
                start = float(ann.get("start_time") or 0)
                dur   = draw_durations[i]
                if t < start:
                    continue
                elapsed = t - start
                prog_map[i] = 1.0 if elapsed >= dur else _eased(elapsed / dur)

            if not prog_map:
                ff.stdin.write(slide_bytes)
            else:
                svg = _build_frame_svg(annotations, prog_map, ocr_src_w, ocr_src_h)
                if svg:
                    png_bytes = cairosvg.svg2png(
                        bytestring=svg.encode(),
                        output_width=W,
                        output_height=H,
                    )
                    overlay = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                    frame   = Image.alpha_composite(slide_rgba, overlay)
                    ff.stdin.write(frame.tobytes())
                else:
                    ff.stdin.write(slide_bytes)
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
