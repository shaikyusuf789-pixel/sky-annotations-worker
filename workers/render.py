"""
workers/render.py — FFmpeg clip renderer with per-frame Pillow + CairoSVG compositing.

Based on the original Replit render.py, hardened for long videos:
  • Frame-state caching (quantized progress) avoids re-rasterizing identical frames
  • All PIL Image objects are explicitly closed (no "Too many open files")
  • Robust ffmpeg pipe handling (BrokenPipe → surfaces stderr)
  • Supports bbox as list [x,y,w,h] OR dict {x,y,w,h}

Pipeline:
  1. Resize slide PNG to 1920×1080 (letterboxed, black bars)
  2. For each frame at 30fps, composite active annotation SVG strokes onto the slide
  3. Pipe raw RGBA frames into FFmpeg → MP4 (libx264 + AAC)
"""

import io
import math
import subprocess
from typing import Any

import cairosvg
from PIL import Image

W, H, FPS = 1280, 720, 24

STROKE_COLORS = {
    "underline":         "#ffffff",
    "double_underline":  "#ffffff",
    "circle":            "#ffffff",
    "box":               "#ffffff",
    "arrow":             "#ffffff",
}

# Base drawing durations
DRAW_SECONDS = {
    "underline":         1.2,
    "double_underline":  1.6,
    "circle":            1.8,
    "box":               2.0,
    "arrow":             1.2,
}

import random

def _add_human_jitter(pts: list[tuple[int, int]], intensity: float = 1.5) -> list[tuple[int, int]]:
    """Adds small random offsets to mimic shaky human hand."""
    if not pts: return pts
    return [
        (round(x + random.uniform(-intensity, intensity)), 
         round(y + random.uniform(-intensity, intensity)))
        for x, y in pts
    ]

def _get_image_brightness(img: Image.Image) -> str:
    """Returns 'light' or 'dark' based on perceived brightness."""
    # Convert to grayscale and get average
    grayscale = img.convert("L")
    stat = grayscale.getdata()
    avg = sum(stat) / len(stat)
    return "dark" if avg < 128 else "light"


# Quantize per-annotation progress to this many steps so frames with
# the same visual state can be cached and reused.
PROG_STEPS = 40


# ── Coordinate transform (OCR source → 1920×1080) ───────────────────────────

def _scale_bbox(bbox: Any, src_w: int, src_h: int) -> tuple[int, int, int, int]:
    scale = min(W / src_w, H / src_h)
    off_x = (W - src_w * scale) / 2
    off_y = (H - src_h * scale) / 2

    if isinstance(bbox, dict):
        bx = bbox.get("x", 0); by = bbox.get("y", 0)
        bw = bbox.get("w", bbox.get("width", 0))
        bh = bbox.get("h", bbox.get("height", 0))
    else:
        bx, by, bw, bh = bbox

    return (
        round(bx * scale + off_x),
        round(by * scale + off_y),
        round(bw * scale),
        round(bh * scale),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _eased(t: float) -> float:
    return t * t * (3 - 2 * t)


def _pts_to_path(pts: list[tuple[int, int]]) -> str:
    if not pts:
        return ""
    d = f"M{pts[0][0]},{pts[0][1]}"
    for p in pts[1:]:
        d += f" L{p[0]},{p[1]}"
    return d


def _underline_pts(x, y_bottom, w):
    steps = max(20, w // 4)
    pts = [(x + w * i // steps, y_bottom + 4) for i in range(steps + 1)]
    return _add_human_jitter(pts)


def _double_underline_pts(x, y_bottom, w):
    steps = max(20, w // 4)
    line1 = [(x + w * i // steps, y_bottom + 4) for i in range(steps + 1)]
    line2 = [(x + w * i // steps, y_bottom + 11) for i in range(steps + 1)]
    return [_add_human_jitter(line1), _add_human_jitter(line2)]


def _circle_pts(cx, cy, rx, ry):
    # Humans don't draw perfect circles; start/end mismatch slightly
    pts = []
    # 95 steps to overdraw slightly
    for i in range(95):
        angle = -math.pi / 2 - (2.05 * math.pi * i / 89)
        px = cx + rx * math.cos(angle)
        py = cy + ry * math.sin(angle)
        pts.append((round(px), round(py)))
    return _add_human_jitter(pts, intensity=2.0)


def _box_pts(x, y, w, h):
    n = 20
    def side(x1, y1, x2, y2):
        return [(x1 + (x2 - x1) * i // n, y1 + (y2 - y1) * i // n) for i in range(n)]
    
    # Human box: corners don't always meet perfectly
    pts = (side(x, y, x+w, y) + side(x+w, y, x+w, y+h)
            + side(x+w, y+h, x, y+h) + side(x, y+h, x, y))
    # Close it imperfectly
    pts.append((x + random.randint(-3, 3), y + random.randint(-3, 3)))
    return _add_human_jitter(pts)


def _arrow_pts(x, y_mid):
    tip = x - 10
    stem = [(tip - 40 + i * 2, y_mid) for i in range(20)]
    head1 = [(tip - i * 8, y_mid - i * 8) for i in range(5)]
    head2 = [(tip - 40 + i * 8, y_mid - 40 + i * 8) for i in range(5)]
    return _add_human_jitter(stem + head1 + head2)



# ── SVG frame builder ────────────────────────────────────────────────────────

def _build_frame_svg(annotations, progress_map, src_w, src_h, default_color="#ffffff") -> str:
    paths_svg = []
    for idx, ann in enumerate(annotations):
        prog = progress_map.get(idx)
        if prog is None or prog <= 0:
            continue
        ann_type = ann["type"]
        # User requested white pen always
        color = "#ffffff"

        # Reduced stroke width to 70% of previous (8->5.6, 6->4.2)
        sw = "5.6" if ann_type in ("circle", "box") else "4.2"
        x, y, w, h = _scale_bbox(ann["bbox"], src_w, src_h)

        def add(pts, stroke_w=sw):
            n = max(2, round(len(pts) * prog))
            d = _pts_to_path(pts[:n])
            paths_svg.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{stroke_w}" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        if ann_type == "underline":
            add(_underline_pts(x, y + h, w))
        elif ann_type == "double_underline":
            for line in _double_underline_pts(x, y + h, w):
                add(line, "6")
        elif ann_type == "circle":
            add(_circle_pts(x + w / 2, y + h / 2, w / 2 + 14, h / 2 + 12))
        elif ann_type == "box":
            add(_box_pts(x - 6, y - 4, w + 12, h + 8))
        elif ann_type == "arrow":
            add(_arrow_pts(x, y + h // 2))

    if not paths_svg:
        return ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
        + "".join(paths_svg) + "</svg>"
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


# ── Main render ───────────────────────────────────────────────────────────────

def render_clip(
    slide_path: str,
    audio_path: str,
    annotations: list[dict[str, Any]],
    output_path: str,
    ocr_src_w: int = 2400,
    ocr_src_h: int = 1350,
) -> float:
    audio_dur = get_audio_duration(audio_path)
    total_frames = math.ceil(audio_dur * FPS)
    print(f"[RENDER] {total_frames} frames @ {FPS}fps, dur={audio_dur:.2f}s, anns={len(annotations)}", flush=True)

    # Load + letterbox slide once
    with Image.open(slide_path) as img:
        raw_slide = img.convert("RGBA")
    try:
        bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        sw, sh = raw_slide.size
        scale = min(W / sw, H / sh)
        nw, nh = round(sw * scale), round(sh * scale)
        resized = raw_slide.resize((nw, nh), Image.LANCZOS)
        try:
            bg.paste(resized, ((W - nw) // 2, (H - nh) // 2))
        finally:
            resized.close()
    finally:
        raw_slide.close()
    slide_rgba = bg
    slide_bytes = slide_rgba.tobytes()
    
    # Pen color is now always white as per user request
    default_pen_color = "#ffffff"
    print(f"[RENDER] using white pen (always)", flush=True)

    # Calculate durations and ENFORCE SEQUENTIAL (One hand rule)
    draw_durations = []
    start_times = []
    
    last_end_time = 0.0
    # Sort annotations by their intended start time to ensure we process them in order
    sorted_anns = sorted(enumerate(annotations), key=lambda x: float(x[1].get("start_time") or 0))
    
    # We'll store the adjusted start times in a map linked to original index
    adjusted_starts = {}
    
    for orig_idx, ann in sorted_anns:
        orig_start = float(ann.get("start_time") or 0)
        dur = max(0.5, float(ann.get("draw_duration") or DRAW_SECONDS.get(ann["type"], 1.5)))
        
        # If this starts before the previous one finished, push it forward
        if orig_start < last_end_time:
            new_start = last_end_time + 0.1 # 0.1s gap between strokes
        else:
            new_start = orig_start
            
        adjusted_starts[orig_idx] = new_start
        draw_durations.append(dur) # These will be used by index i below, so order matters
        last_end_time = new_start + dur

    # Re-align start_times list to match original annotation order for the frame loop
    start_times = [adjusted_starts[i] for i in range(len(annotations))]
    draw_durations = [max(0.5, float(ann.get("draw_duration") or DRAW_SECONDS.get(ann["type"], 1.5))) for ann in annotations]


    # Start ffmpeg
    ff = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-y", output_path,
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # Frame cache: quantized progress tuple → rendered bytes
    frame_cache: dict[tuple, bytes] = {}
    MAX_CACHE = 256

    try:
        for f_idx in range(total_frames):
            if ff.poll() is not None:
                break   # ffmpeg died early — stop writing frames

            t = f_idx / FPS

            # Quantized progress signature per annotation
            sig_list = []
            prog_map: dict[int, float] = {}
            any_active = False
            for i in range(len(annotations)):
                start = start_times[i]
                if t < start:
                    sig_list.append(0)
                    continue
                elapsed = t - start
                dur = draw_durations[i]
                if elapsed >= dur:
                    p = 1.0
                else:
                    p = _eased(elapsed / dur)
                q = round(p * PROG_STEPS)
                sig_list.append(q)
                if q > 0:
                    prog_map[i] = q / PROG_STEPS
                    any_active = True

            if not any_active:
                ff.stdin.write(slide_bytes)
                continue

            sig = tuple(sig_list)
            cached = frame_cache.get(sig)
            if cached is not None:
                ff.stdin.write(cached)
                continue

            svg = _build_frame_svg(annotations, prog_map, ocr_src_w, ocr_src_h, default_pen_color)
            if not svg:
                frame_bytes = slide_bytes
            else:
                png_bytes = cairosvg.svg2png(
                    bytestring=svg.encode(),
                    output_width=W, output_height=H,
                )
                with Image.open(io.BytesIO(png_bytes)) as overlay_img:
                    overlay = overlay_img.convert("RGBA")
                try:
                    composed = Image.alpha_composite(slide_rgba, overlay)
                    try:
                        frame_bytes = composed.tobytes()
                    finally:
                        composed.close()
                finally:
                    overlay.close()

            if len(frame_cache) < MAX_CACHE:
                frame_cache[sig] = frame_bytes
            ff.stdin.write(frame_bytes)

    except (BrokenPipeError, ValueError, OSError):
        pass
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.stdin = None          # ← keep this line — prevents communicate() re-flushing closed pipe
        slide_rgba.close()

    _, stderr_data = ff.communicate()
    if ff.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {ff.returncode}): "
            + stderr_data[-2000:].decode(errors="replace")
        )

    print(f"[RENDER] done → {output_path}", flush=True)
    return audio_dur
