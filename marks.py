"""Visible corner labels used by other image generators.

These marks are a logo or a short line of text blended over a known corner.
They are light on dark pictures and dark on light ones, so both are searched.
The search window and the usual width come from that layout. A pixel is part
of the mark only when it is a compact band of neutral lift. The original color
is solved from that blend. A plain photo, and the Gemini sparkle, are left for
the sparkle engine.

Auto only accepts a see-through label that hugs the corner the way generators
place them, so a solid signature or a caption further in is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (id, title, x, y, width, height as fractions, usual text width / image width)
_SPECS = (
    ("grok", "Grok", 0.70, 0.86, 0.30, 0.14, 0.11),
    ("doubao", "Doubao", 0.55, 0.86, 0.45, 0.14, 0.15),
    ("doubao-chat", "Doubao chat", 0.00, 0.00, 0.22, 0.08, 0.078),
    ("jimeng", "Jimeng", 0.55, 0.86, 0.45, 0.14, 0.18),
    ("tongyi", "Tongyi", 0.50, 0.86, 0.50, 0.14, 0.20),
    ("qwen", "Qwen", 0.55, 0.86, 0.45, 0.14, 0.165),
    ("kling", "Kling", 0.60, 0.86, 0.40, 0.14, 0.137),
    ("wenxin", "Wenxin", 0.55, 0.86, 0.45, 0.14, 0.199),
    ("veo", "Veo", 0.80, 0.88, 0.20, 0.12, 0.045),
)


@dataclass(frozen=True)
class MarkSpec:
    id: str
    title: str
    x: float
    y: float
    width: float
    height: float
    text_width: float


MARKS: tuple[MarkSpec, ...] = tuple(MarkSpec(*item) for item in _SPECS)
_BY_ID = {item.id: item for item in MARKS}
MARK_IDS = ("auto", "gemini") + tuple(item.id for item in MARKS)


# Auto: how far a label may sit from its corner, as a share of the picture.
HUG_X = 0.07
HUG_Y = 0.06
# Generator labels are see-through. Solid text is part of the picture.
MAX_LABEL_OPACITY = 0.82
LIGHT_CORNER = 110.0
# Fewest pixels a label can cover. A small export still shows a 120px mark.
MIN_LABEL_PIXELS = 60


@dataclass(frozen=True)
class LabelHit:
    image: np.ndarray
    detail: str
    mark_id: str = ""
    box: tuple[int, int, int, int] | None = None
    mask: np.ndarray | None = None
    dark: bool = False


def mark_menu() -> list[dict[str, str]]:
    rows = [{"id": "auto", "title": "Auto"}, {"id": "gemini", "title": "Gemini"}]
    rows.extend({"id": item.id, "title": item.title} for item in MARKS)
    return rows


def clear_labels(rgb: np.ndarray, mark_id: str | None) -> LabelHit | None:
    """Remove one visible corner label. None means the corner was left alone."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return None
    height, width = rgb.shape[:2]
    if height < 64 or width < 64:
        return None
    if mark_id is None:
        specs = MARKS
        loose = False
    else:
        spec = _BY_ID.get(mark_id)
        if spec is None:
            return None
        specs = (spec,)
        loose = True

    best: tuple[float, LabelHit] | None = None
    tried: dict[tuple[tuple[int, int, int, int], bool], tuple | None] = {}
    for spec in specs:
        box = _box(width, height, spec)
        region = _region(rgb, box)
        # A dark mark only appears on a light corner. On a dark one the gaps
        # between light letters would read as a dark mark.
        light_corner = region is not None and float(np.median(region)) >= LIGHT_CORNER
        for dark in (False, True):
            if dark and not light_corner:
                continue
            key = (box, dark)
            if key not in tried:
                tried[key] = _restore_box(rgb, box, dark=dark)
            found = tried[key]
            if found is None:
                continue
            restored, span, mask, opacity = found
            text_width = span / width
            floor = 0.045 if loose else spec.text_width * 0.62
            if text_width < floor:
                continue
            error = abs(text_width / spec.text_width - 1.0)
            limit = 0.75 if loose else 0.45
            if error > limit:
                continue
            if not loose and opacity > MAX_LABEL_OPACITY:
                continue
            if not loose and not _hugs_corner(mask, box, width, height, spec):
                continue
            if best is None or error < best[0]:
                hit = LabelHit(restored, f"{spec.title} mark", spec.id, box, mask, dark)
                best = (error, hit)
    return None if best is None else best[1]


def _hugs_corner(mask: np.ndarray, box: tuple[int, int, int, int], width: int, height: int, spec: MarkSpec) -> bool:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return False
    x0, y0 = box[:2]
    left = x0 + int(xs.min())
    right = x0 + int(xs.max())
    top = y0 + int(ys.min())
    bottom = y0 + int(ys.max())
    tall = bottom - top + 1
    if tall > max(16, 0.05 * height):
        return False
    gap_x = left / width if spec.x < 0.5 else (width - 1 - right) / width
    gap_y = top / height if spec.y < 0.5 else (height - 1 - bottom) / height
    return gap_x <= HUG_X and gap_y <= HUG_Y


def _region(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, box_w, box_h = box
    region = rgb[y : y + box_h, x : x + box_w]
    return region if region.shape[0] >= 8 and region.shape[1] >= 8 else None


def apply_mask(rgb: np.ndarray, hit: LabelHit, mask: np.ndarray | None) -> np.ndarray:
    """Clean a known label with a fixed mask. A frame without it barely changes."""
    if hit.box is None or mask is None:
        return rgb
    region = _region(rgb, hit.box)
    if region is None or region.shape[:2] != mask.shape:
        return rgb
    plane = 255 - region if hit.dark else region
    restored = _unblend(plane, mask, strict=False)
    if restored is None:
        return rgb
    x, y, box_w, box_h = hit.box
    out = rgb.copy()
    out[y : y + box_h, x : x + box_w] = 255 - restored if hit.dark else restored
    return out


def _frame_mask(rgb: np.ndarray, hit: LabelHit) -> np.ndarray | None:
    region = _region(rgb, hit.box) if hit.box is not None else None
    if region is None:
        return None
    plane = 255 - region if hit.dark else region
    mask, _span = _text_mask(plane, relaxed=True)
    return mask


def mask_share(frames: list[np.ndarray], hit: LabelHit) -> float:
    """Share of frames that show the same label in the same pixels."""
    if hit.mask is None or not frames:
        return 0.0
    same = 0
    for frame in frames:
        mask = _frame_mask(frame, hit)
        if mask is None or mask.shape != hit.mask.shape:
            continue
        overlap = int((mask & hit.mask).sum())
        if overlap >= 0.5 * int(hit.mask.sum()):
            same += 1
    return same / len(frames)


def stable_mask(frames: list[np.ndarray], hit: LabelHit) -> np.ndarray | None:
    """Pixels the label covers on most frames. Passing detail drops out."""
    masks = [mask for mask in (_frame_mask(frame, hit) for frame in frames) if mask is not None]
    masks = [mask for mask in masks if hit.mask is not None and mask.shape == hit.mask.shape]
    if len(masks) < 3:
        return hit.mask
    votes = np.stack(masks).mean(axis=0) >= 0.5
    return votes if int(votes.sum()) >= MIN_LABEL_PIXELS else hit.mask


def _box(width: int, height: int, spec: MarkSpec) -> tuple[int, int, int, int]:
    x = int(spec.x * width)
    y = int(spec.y * height)
    box_w = max(12, int(spec.width * width))
    box_h = max(12, int(spec.height * height))
    x = min(x, width - 8)
    y = min(y, height - 8)
    box_w = min(box_w, width - x)
    box_h = min(box_h, height - y)
    return x, y, box_w, box_h


def _restore_box(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    relaxed: bool = False,
    dark: bool = False,
) -> tuple[np.ndarray, int, np.ndarray, float] | None:
    x, y, box_w, box_h = box
    if x < 0 or y < 0 or box_w < 8 or box_h < 8:
        return None
    region = _region(rgb, box)
    if region is None:
        return None
    # A dark mark is a light one on the inverted picture.
    plane = 255 - region if dark else region
    mask, span = _text_mask(plane, relaxed=relaxed)
    if mask is None or span is None:
        return None
    restored = _unblend(plane, mask)
    if restored is None:
        return None
    out = rgb.copy()
    out[y : y + region.shape[0], x : x + region.shape[1]] = 255 - restored if dark else restored
    under = restored.astype(np.float32).mean(axis=2)[mask]
    over = plane.astype(np.float32).mean(axis=2)[mask]
    # Stroke centers, not their soft edges, show how solid the text is.
    opacity = float(np.percentile((over - under) / np.maximum(255.0 - under, 1.0), 75))
    return out, span, mask, opacity


def _text_mask(region: np.ndarray, relaxed: bool = False) -> tuple[np.ndarray, int] | tuple[None, None]:
    height, width = region.shape[:2]
    if height < 8 or width < 16:
        return None, None
    plane = region.astype(np.float32)
    # Horizontal only. A square minimum sits inside a tall letter and reports no lift.
    # Opening, not a plain minimum: a soft gradient is not a mark.
    radius = int(np.clip(width / 28, 10, 22))
    background = np.empty_like(plane)
    for channel in range(3):
        background[:, :, channel] = _open_h(plane[:, :, channel], radius)
    lift = plane - background
    spread = lift.max(axis=2) - lift.min(axis=2)
    amount = lift.mean(axis=2)
    amount_min = 10.0 if relaxed else 16.0
    spread_max = 28.0 if relaxed else 22.0
    lift_min = 4.0 if relaxed else 6.0
    # A blend over a colored picture lifts each channel by its own share.
    mask = (amount >= amount_min) & (spread <= np.maximum(spread_max, 0.32 * amount)) & (lift.min(axis=2) >= lift_min)
    mask = _keep_core(mask)
    if int(mask.sum()) < MIN_LABEL_PIXELS:
        return None, None

    rows = mask.sum(axis=1) >= max(4, int(width * 0.012))
    band = _run(rows, gap=3)
    if band is None:
        return None, None
    top, bottom = band
    if bottom - top + 1 > height * 0.75:
        return None, None
    if top <= 1 and not relaxed:
        return None, None
    cols = mask[top : bottom + 1].sum(axis=0) >= 2
    # Letter spacing is a gap, not a new mark. Join strokes that sit on one line.
    run = _run(cols, gap=max(8, (bottom - top) // 2))
    if run is None:
        return None, None
    left, right = run
    span = right - left + 1
    if span < 12 or span > width * 0.92:
        return None, None
    if not relaxed and span / max(1, bottom - top + 1) < 2.0:
        return None, None
    crop = mask[top : bottom + 1, left : right + 1]
    density = float(crop.mean())
    density_lo, density_hi = (0.05, 0.88) if relaxed else (0.08, 0.8)
    if density < density_lo or density > density_hi:
        return None, None
    if crop.size < 48:
        return None, None

    kept = np.zeros_like(mask)
    kept[top : bottom + 1, left : right + 1] = crop
    covered = int(kept.sum())
    if covered < MIN_LABEL_PIXELS or covered / float(mask.size) > (0.40 if relaxed else 0.28):
        return None, None
    return kept, span


def _unblend(region: np.ndarray, mask: np.ndarray, strict: bool = True) -> np.ndarray | None:
    """Rebuild the pixels under a label from the pixels around it.

    A label's own color and opacity are unknown, and solving for them leaves a
    colored smear. Its strokes are thin, so the surroundings say what was there.
    """
    from engine import _fill_hole, _grow

    hole = _grow(_whole_label(region, mask), 2)
    ys, xs = np.nonzero(hole)
    if xs.size == 0:
        return None
    pad = 6
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(hole.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(hole.shape[1], int(xs.max()) + pad + 1)
    plane = region.astype(np.float32)
    out = plane.copy()
    piece = hole[y0:y1, x0:x1]
    if bool(piece.all()):
        return None
    for channel in range(3):
        out[y0:y1, x0:x1, channel] = _fill_hole(plane[y0:y1, x0:x1, channel], piece)
    changed = np.abs(out - plane)[mask]
    if changed.size == 0:
        return None
    if strict and (float(changed.mean()) < 8.0 or float(changed.mean()) > 170.0):
        return None
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _whole_label(region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Add the soft edges around a label's strokes.

    Detection keeps only the strong stroke centers. Filling just those leaves a
    ghost of the letters, so the fainter pixels inside the label's box join in.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask
    height, width = mask.shape
    y0, y1 = max(0, int(ys.min()) - 3), min(height, int(ys.max()) + 4)
    x0, x1 = max(0, int(xs.min()) - 3), min(width, int(xs.max()) + 4)
    gray = region.astype(np.float32).mean(axis=2)
    lift = gray - _open_h(gray, int(np.clip(width / 28, 10, 22)))
    floor = max(5.0, 0.18 * float(np.median(lift[mask])))
    soft = np.zeros_like(mask)
    soft[y0:y1, x0:x1] = lift[y0:y1, x0:x1] >= floor
    return mask | soft


def _min_filter_h(channel: np.ndarray, radius: int) -> np.ndarray:
    """Sliding minimum across rows. The span doubles, so a wide window stays cheap."""
    window = radius * 2 + 1
    out = np.pad(channel, ((0, 0), (radius, radius)), mode="edge")
    span = 1
    while span * 2 <= window:
        out = np.minimum(out[:, :-span], out[:, span:])
        span *= 2
    rest = window - span
    if rest:
        out = np.minimum(out[:, :-rest], out[:, rest:])
    return out


def _open_h(channel: np.ndarray, radius: int) -> np.ndarray:
    """Remove bright strokes narrower than the window and keep smooth ramps."""
    return -_min_filter_h(-_min_filter_h(channel, radius), radius)


def _keep_core(mask: np.ndarray) -> np.ndarray:
    """Drop isolated pixels. A letter stroke has neighbors."""
    neighbors = np.zeros(mask.shape, dtype=np.uint8)
    neighbors[1:] += mask[:-1]
    neighbors[:-1] += mask[1:]
    neighbors[:, 1:] += mask[:, :-1]
    neighbors[:, :-1] += mask[:, 1:]
    return mask & (neighbors >= 2)


def _run(active: np.ndarray, gap: int) -> tuple[int, int] | None:
    index = np.flatnonzero(active)
    if index.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(index) > gap + 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, index.size - 1]
    best = max(range(starts.size), key=lambda item: int(index[ends[item]] - index[starts[item]]))
    return int(index[starts[best]]), int(index[ends[best]])
