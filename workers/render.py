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
import subprocess
import tempfile
import os
from typing import Any

import cairosvg
from PIL import Image

W, H, FPS = 1920, 1080, 30

STROKE_COLORS = {
    "underline":         "#3b82f6",
    "double_underline":  "#0ea5e9",
    "circle":            "#f59e0b",
    "box":               "#10b981",
    "arrow":             "#8b5cf6",
}

DRAW_SECONDS = {
    "underline":         1.2,
    "double_underline":  1.6,
    "circle":            1.8,
    "box":               2.0,
    "arrow":             1.2,
}


# ── Coordinate transform (OCR source → 1920×1080) ───────────────────────────

def _scale_bbox(
    bbox: Any,
    src_w: int,
    src_h: int,
) -> tuple[int, int, int, int]:
    """Scale OCR bbox from source image dimensions to 1920×1080 letterboxed space."""
    scale   = min(W / src_w, H / src_h)
    off_x   = (W - src_w * scale) / 2
    off_y   = (H - src_h * scale) / 2

    if isinstance(bbox, dict):
        bx, by, bw, bh = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
    else:
        bx, by, bw, bh = bbox

    return (
        round(bx * scale + off_x),
        round(by * scale + off_y),
        round(bw * scale),
        round(bh * scale),
    )


# ── SVG path builders ────────────────────────────────────────────────────────

def _eased(t: float) -> float:
    return t * t * (3 - 2 * t)


def _pts_to_path(pts: list[tuple[int, int]]) -> str:
    if not pts:
        return ""
    d = f"M{pts[0][0]},{pts[0][1]}"
    for p in pts[1:]:
        d += f" L{p[0]},{p[1]}"
    return d


def _underline_pts(x: int, y_bottom: int, w: int) -> list[tuple[int, int]]:
    steps = max(20, w // 4)
    return [(x + w * i // steps, y_bottom + 4) for i in range(steps + 1)]


def _double_underline_pts(x: int, y_bottom: int, w: int) -> list[list[tuple[int, int]]]:
    steps = max(20, w // 4)
    line1 = [(x + w * i // steps, y_bottom + 4) for i in range(steps + 1)]
    line2 = [(x + w * i // steps, y_bottom + 11) for i in range(steps + 1)]
    return [line1, line2]


def _circle_pts(cx: float, cy: float, rx: float, ry: float) -> list[tuple[int, int]]:
    return [
        (
            round(cx + rx * math.cos(-math.pi / 2 - 2.1 * math.pi * i / 89)),
            round(cy + ry * math.sin(-math.pi / 2 - 2.1 * math.pi * i / 89)),
        )
        for i in range(90)
    ]


def _box_pts(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    n = 20
    def side(x1, y1, x2, y2):
        return [(x1 + (x2 - x1) * i // n, y1 + (y2 - y1) * i // n) for i in range(n)]
    return side(x, y, x+w, y) + side(x+w, y, x+w, y+h) + side(x+w, y+h, x, y+h) + side(x, y+h, x, y) + [(x, y)]


def _arrow_pts(x: int, y_mid: int) -> list[tuple[int, int]]:
    tip = x - 10
    return (
        [(tip - 40 + i * 2, y_mid) for i in range(20)] +
        [(tip - i * 8, y_mid - i * 8) for i in range(5)] +
        [(tip - 40 + i * 8, y_mid - 40 + i * 8) for i in range(5)]
    )


# ── Frame SVG builder ────────────────────────────────────────────────────────

def _build_frame_svg(
    annotations: list[dict],
    progress_map: dict[int, float],  # ann_index → 0.0–1.0
    src_w: int,
    src_h: int,
) -> str:
    paths_svg = []

    for idx, ann in enumerate(annotations):
        prog = progress_map.get(idx)
        if prog is None or prog <= 0:
            continue

        ann_type = ann["type"]
        color    = STROKE_COLORS.get(ann_type, "#ffffff")
        sw       = "8" if ann_type in ("circle", "box") else "6"
        x, y, w, h = _scale_bbox(ann["bbox"], src_w, src_h)

        if ann_type == "underline":
            pts  = _underline_pts(x, y + h, w)
            n    = max(2, round(len(pts) * prog))
            d    = _pts_to_path(pts[:n])
            paths_svg.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{sw}" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        elif ann_type == "double_underline":
            for line_pts in _double_underline_pts(x, y + h, w):
                n  = max(2, round(len(line_pts) * prog))
                d  = _pts_to_path(line_pts[:n])
                paths_svg.append(
                    f'<path d="{d}" stroke="{color}" stroke-width="6" '
                    f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
                )

        elif ann_type == "circle":
            cx, cy = x + w / 2, y + h / 2
            rx, ry = w / 2 + 14, h / 2 + 12
            pts  = _circle_pts(cx, cy, rx, ry)
            n    = max(2, round(len(pts) * prog))
            d    = _pts_to_path(pts[:n])
            paths_svg.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{sw}" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        elif ann_type == "box":
            pts = _box_pts(x - 6, y - 4, w + 12, h + 8)
            n   = max(2, round(len(pts) * prog))
            d   = _pts_to_path(pts[:n])
            paths_svg.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{sw}" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

        elif ann_type == "arrow":
            pts = _arrow_pts(x, y + h // 2)
            n   = max(2, round(len(pts) * prog))
            d   = _pts_to_path(pts[:n])
            paths_svg.append(
                f'<path d="{d}" stroke="{color}" stroke-width="{sw}" '
                f'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            )

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

    # Start FFmpeg
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-y", output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    for f_idx in range(total_frames):
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

    ff.stdin.close()
    _, stderr_data = ff.communicate()
    if ff.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {ff.returncode}: "
            + stderr_data[-2000:].decode(errors="replace")
        )

    print(f"[RENDER] done → {output_path}")
    return audio_dur
