"""Remove the visible Gemini sparkle by reversing its alpha blend.

Gemini composites a white sparkle over the bottom-right corner:

    watermarked = alpha * 255 + (1 - alpha) * original

The sparkle's alpha is known, so the original pixels are:

    original = (watermarked - alpha * 255) / (1 - alpha)

The mark has shipped at more than one size, inset, and opacity. Recent 2K
exports use a fainter 96px sparkle set 192px in from the corner. That one
needs its own alpha capture; scaling the older capture leaves the star
behind. Auto mode tries the known layouts and keeps a match only when
removing it wipes out the sparkle's own outline.

The stored captures only give the star's shape. Its opacity is measured from
the picture, because a small opacity error is amplified on dark pixels. A
video gets its own star, estimated from many frames of that clip, and each
frame is checked so a frame without the mark is left alone.
"""

from __future__ import annotations

import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MODES = ("auto", "standard", "current")

ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99
LOGO_VALUE = 255.0
PATCH_MARGIN = 3

# Removal has to erase the sparkle outline. Above this, the outline survived
# or the subtraction invented edges, which is what happens on a clean photo.
MAX_SURVIVAL = 0.80
# Leftover outline energy, compared with the same shape measured on nearby
# patches of the same photo. Content that merely sits in the corner fails here.
MAX_CONTROL_RATIO = 3.0
# A smooth or busy-dark corner makes that ratio useless. Accept those only
# when the patch is the sparkle: its shape matches the template and the
# bright pixels sit on the star. Survival may be higher than on textured
# photos because foam, cloth, or JPEG noise keeps some outline energy.
MIN_SHAPE_NCC = 0.75
MIN_SHAPE_LIFT = 14.0
SHAPE_ALPHA_HIGH = 0.20
SHAPE_ALPHA_LOW = 0.05

# Full-strength captures, then the fainter opacities used on recent exports.
SCALES = (1.0, 0.70, 0.85, 1.15, 0.60, 1.30, 0.55, 0.45, 0.40, 0.35)
# bg_96_20260520.png is already the fainter star. Do not scale it down again.
SCALES_2026 = (1.0, 0.85, 1.15, 0.70, 0.55)
# The corner search. Peaks under this correlation are not worth scoring.
MIN_PEAK_NCC = 0.42
MAX_PEAKS = 3
# Capture noise around the star. Real edge pixels sit inside the grown core.
STAR_CORE = 0.06
STAR_GROW = 2
# A clip's estimated star may differ from the capture by this much at most.
MAX_ALPHA_DRIFT = 0.12
# (size, inset from the right and bottom). Current 48px / 96px layout first.
ANCHORS = (
    (48, 96),
    (48, 32),
    (48, 64),
    (96, 64),
    (96, 96),
    (96, 192),
    (36, 96),
    (36, 64),
)


def v2_layouts(width: int, height: int) -> tuple[tuple[int, int, int], ...]:
    """Gemini 3.5+ corners: (logo size, margin from the right, margin from the bottom).

    Large exports stay at 96px / 192px. Smaller exports inherit a scaled margin
    from the 2752/2816/2848 source, so a fixed 48px / 32px corner misses them.
    720p video diamonds are included because they do not use that image formula.
    """
    found: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(size: int, right: int, bottom: int) -> None:
        size = int(size)
        right = int(right)
        bottom = int(bottom)
        if size < 28 or right < 8 or bottom < 8:
            return
        if right + size + PATCH_MARGIN > width or bottom + size + PATCH_MARGIN > height:
            return
        key = (size, right, bottom)
        if key in seen:
            return
        seen.add(key)
        found.append(key)

    if width > 1024 and height > 1024:
        add(96, 192, 192)
    else:
        long_side = max(width, height)
        short_side = min(width, height)
        if long_side > 1100:
            doubled = 2.0 * long_side
            source = 2752.0
            for cand in (2816.0, 2848.0):
                if abs(doubled - cand) < abs(doubled - source):
                    source = cand
        elif short_side >= 566:
            source = 2752.0
        elif short_side >= 550:
            source = 2816.0
        else:
            source = 2848.0
        scale = long_side / source
        margin = int(round(192.0 * scale))
        ideal = int(round(96.0 * scale))
        logo = 36 if ideal <= 40 else ideal
        add(logo, margin, margin)
        for delta in (-2, 2):
            add(logo + delta, margin, margin)
        for delta in (-4, -2, 2, 4):
            add(logo, margin + delta, margin + delta)

    pair = (width, height)
    if pair in {(1280, 720), (720, 1280)}:
        add(48, 72, 72)
        add(44, 29, 40)
        add(44, 40, 29)
    if pair in {(1920, 1080), (1080, 1920)}:
        add(96, 192, 192)
        add(72, 108, 108)
    if height > width and min(width, height) <= 1280:
        add(48, 96, 128)
        add(44, 64, 96)
    return tuple(found)


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def asset_dir() -> Path:
    return app_dir() / "assets"


@dataclass
class Prepared:
    alpha: np.ndarray
    weight: np.ndarray
    scale: float
    size: int
    variant: str


@dataclass
class Match:
    x: int
    y: int
    inset: int
    prepared: Prepared
    survival: float
    control_ratio: float
    residual_lift: float = 0.0
    clip: float = 0.0
    inset_x: int = -1
    inset_y: int = -1

    @property
    def size(self) -> int:
        return self.prepared.size

    @property
    def right(self) -> int:
        return self.inset if self.inset_x < 0 else self.inset_x

    @property
    def bottom(self) -> int:
        return self.inset if self.inset_y < 0 else self.inset_y


@dataclass
class CleanInfo:
    status: str
    detail: str
    size: int | None = None
    inset: int | None = None
    survival: float | None = None


class Stopped(Exception):
    """The caller cancelled a long upscale between tiles."""


def rgb_array(image: Image.Image) -> np.ndarray:
    """Copy a PIL image to a contiguous RGB uint8 array. Views dangle after close."""
    if image.mode.startswith("I") or image.mode == "F":
        # Pillow clips 16-bit values at 255, which turns the picture white.
        deep = np.array(image, dtype=np.float64, copy=True)
        top = 65535.0 if float(deep.max(initial=0.0)) > 255.0 else 255.0
        gray = np.clip(np.rint(deep * 255.0 / top), 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=2)
    return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)


def gray_array(image: Image.Image) -> np.ndarray:
    """Copy a PIL image to a contiguous uint8 array."""
    return np.array(image, dtype=np.uint8, copy=True)


def _grow(mask: np.ndarray, steps: int) -> np.ndarray:
    """Dilate a boolean mask by a plus-shaped step, without wrapping at the edges."""
    out = mask.copy()
    for _ in range(int(steps)):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _star_only(alpha: np.ndarray) -> np.ndarray:
    """Drop the capture noise around the star.

    The captures carry about 0.02 alpha across their whole square. Reversing
    that darkens a visible box, so everything away from the star becomes zero.
    """
    core = alpha >= STAR_CORE
    if not core.any():
        return alpha
    return np.where(_grow(core, STAR_GROW), alpha, 0.0).astype(np.float32)


def _load_alpha_png(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    channels = np.array(image, dtype=np.float32, copy=True)
    return _star_only(channels.max(axis=2) / 255.0)


def _scale_alpha(alpha: np.ndarray, size: int) -> np.ndarray:
    if alpha.shape == (size, size):
        return alpha
    plate = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    plate = plate.resize((size, size), Image.Resampling.BILINEAR)
    return np.array(plate, dtype=np.float32, copy=True) / 255.0


def _silhouette_weight(alpha: np.ndarray) -> np.ndarray:
    """How fast alpha changes. The sparkle's outline lives on this ring."""
    levels = alpha * 255.0
    gx = np.zeros_like(levels)
    gy = np.zeros_like(levels)
    gx[:, 1:-1] = levels[:, 2:] - levels[:, :-2]
    gy[1:-1, :] = levels[2:, :] - levels[:-2, :]
    weight = np.hypot(gx, gy)
    # Faint captures have weaker gradients. 5 keeps the star ring and drops noise.
    weight[weight < 5.0] = 0.0
    weight[0, :] = 0.0
    weight[-1, :] = 0.0
    weight[:, 0] = 0.0
    weight[:, -1] = 0.0
    return weight


def _edge_energy(patch: np.ndarray, weight: np.ndarray, margin: int = PATCH_MARGIN) -> float:
    """Gradient energy along the sparkle outline, using pixels just outside it."""
    total = float(weight.sum())
    if total <= 0.0:
        return 0.0
    height, width = weight.shape
    pixels = patch.astype(np.float32, copy=False)
    y0 = margin
    x0 = margin
    gx = pixels[y0 : y0 + height, x0 + 1 : x0 + width + 1] - pixels[y0 : y0 + height, x0 - 1 : x0 + width - 1]
    gy = pixels[y0 + 1 : y0 + height + 1, x0 : x0 + width] - pixels[y0 - 1 : y0 + height - 1, x0 : x0 + width]
    energy = np.sum(gx * gx, axis=2) + np.sum(gy * gy, axis=2)
    return float((energy * weight).sum() / total)


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64, copy=False).ravel()
    b = right.astype(np.float64, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 1e-8:
        return 0.0
    return float((a * b).sum() / denom)


def _residual_lift(region: np.ndarray, alpha: np.ndarray) -> float:
    """Signed leftover: bright star is positive, a punched hole is negative."""
    if region.shape[:2] != alpha.shape:
        return 1e6
    gray = region.astype(np.float32).mean(axis=2)
    high = alpha > SHAPE_ALPHA_HIGH
    low = alpha < SHAPE_ALPHA_LOW
    if int(high.sum()) < 8 or int(low.sum()) < 8:
        return 1e6
    return float(gray[high].mean() - gray[low].mean())


def _matches_sparkle(region: np.ndarray, alpha: np.ndarray) -> bool:
    """True when the patch is brighter on the star and shaped like the template."""
    if region.shape[:2] != alpha.shape:
        return False
    gray = region.astype(np.float32).mean(axis=2)
    if _ncc(gray, alpha) < MIN_SHAPE_NCC:
        return False
    high = alpha > SHAPE_ALPHA_HIGH
    low = alpha < SHAPE_ALPHA_LOW
    if int(high.sum()) < 8 or int(low.sum()) < 8:
        return False
    lift = float(gray[high].mean() - gray[low].mean())
    return lift >= MIN_SHAPE_LIFT


def _measure(rgb: np.ndarray, x: int, y: int, prepared: Prepared) -> tuple[float, float] | None:
    """Return (outline survival, leftover energy) for one candidate."""
    height, width = rgb.shape[:2]
    size = prepared.size
    if not _region_fits(height, width, x, y, size):
        return None
    if float(prepared.weight.sum()) <= 0.0:
        return None
    margin = PATCH_MARGIN
    patch = rgb[y - margin : y + size + margin, x - margin : x + size + margin]
    before = _edge_energy(patch, prepared.weight, margin)
    if before < 1.0:
        return None
    cleaned = patch.copy()
    cleaned[margin : margin + size, margin : margin + size] = _reverse_region(
        patch[margin : margin + size, margin : margin + size],
        prepared.alpha,
        fill=False,
    )
    after = _edge_energy(cleaned, prepared.weight, margin)
    return after / before, after


def _clip_fraction(region: np.ndarray, alpha: np.ndarray) -> float:
    """Share of sparkle pixels the reverse-blend sends below 0. Over-strong opacity."""
    blended = region.astype(np.float32)
    clamped = np.minimum(alpha, MAX_ALPHA)
    restored = (blended - clamped[..., None] * LOGO_VALUE) / (1.0 - clamped[..., None])
    keep = alpha >= ALPHA_THRESHOLD
    total = int(keep.sum())
    if total < 8:
        return 0.0
    hole = keep & np.any(restored < -1.0, axis=2)
    return float(hole.sum()) / float(total)


def _fill_clipped(plane: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Replace reverse-blend undershoot with the nearest surviving pixel on the row."""
    out = plane.copy()
    height = hole.shape[0]
    for row in range(height):
        need = hole[row]
        if not need.any():
            continue
        clear = np.flatnonzero(~need)
        if clear.size == 0:
            continue
        xs = np.flatnonzero(need)
        place = np.searchsorted(clear, xs)
        left = np.clip(place - 1, 0, clear.size - 1)
        right = np.clip(place, 0, clear.size - 1)
        choose_left = np.abs(clear[left] - xs) <= np.abs(clear[right] - xs)
        chosen = np.where(choose_left, clear[left], clear[right])
        out[row, xs] = plane[row, chosen]
    return out


def _reverse_region(region: np.ndarray, alpha: np.ndarray, fill: bool = True) -> np.ndarray:
    """Undo a white alpha blend. region is uint8 HxWx3, alpha is float HxW.

    Scoring passes fill=False. Holes only matter on the pixels that get saved.
    """
    blended = region.astype(np.float32)
    clamped = np.minimum(alpha, MAX_ALPHA)
    restored = (blended - clamped[..., None] * LOGO_VALUE) / (1.0 - clamped[..., None])
    keep = alpha >= ALPHA_THRESHOLD
    mixed = np.where(keep[..., None], restored, blended)
    # Dark JPEG pixels can sit below alpha*255, so the formula goes negative
    # and clips to a black crater. Fill those from nearby restored pixels.
    hole = keep & np.any(restored < -1.0, axis=2)
    if fill and hole.any() and np.any(~hole):
        mixed = np.where(hole[..., None], _fill_clipped(mixed, hole), mixed)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _fill_hole(plane: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Smooth guess at what sits under a hole, grown in from its edge."""
    out = plane.astype(np.float32, copy=True)
    clear = ~hole
    if not clear.any() or not hole.any():
        return out
    out[hole] = float(out[clear].mean())
    for step in (8, 4, 2, 1):
        for _ in range(8 if step > 1 else 14):
            mean = (
                np.roll(out, step, axis=0)
                + np.roll(out, -step, axis=0)
                + np.roll(out, step, axis=1)
                + np.roll(out, -step, axis=1)
            ) * 0.25
            out[hole] = mean[hole]
    return out


def _fit_opacity(gray: np.ndarray, alpha: np.ndarray, hole: np.ndarray) -> tuple[float, float, np.ndarray]:
    """How strongly this alpha is blended into the patch, with the fit's residual.

    Solves observed - background = k * alpha * (255 - background). Textured
    pixels that disagree with the star are weighted down.
    """
    back = _fill_hole(gray, hole)
    x = alpha * np.maximum(255.0 - back, 1.0)
    y = gray - back
    use = alpha > 0.08
    if int(use.sum()) < 8:
        return 0.0, 1e6, back
    xs = x[use]
    ys = y[use]
    weight = np.ones_like(xs)
    k = 0.0
    for _ in range(3):
        denom = float((weight * xs * xs).sum())
        if denom <= 1e-6:
            return 0.0, 1e6, back
        k = float((weight * xs * ys).sum()) / denom
        rest = ys - k * xs
        spread = 1.4826 * float(np.median(np.abs(rest))) + 1.0
        weight = 1.0 / (1.0 + (rest / (2.0 * spread)) ** 2)
    rest = ys - k * xs
    return k, float(np.sqrt((weight * rest * rest).sum() / max(float(weight.sum()), 1.0))), back


def _fit_rest(gray: np.ndarray, alpha: np.ndarray, hole: np.ndarray) -> float:
    """How far off this exact alpha is, with no freedom to rescale it."""
    back = _fill_hole(gray, hole)
    use = alpha > 0.08
    if int(use.sum()) < 8:
        return 1e6
    rest = (gray - back - alpha * np.maximum(255.0 - back, 1.0))[use]
    spread = 1.4826 * float(np.median(np.abs(rest))) + 1.0
    weight = 1.0 / (1.0 + (rest / (2.0 * spread)) ** 2)
    return float(np.sqrt((weight * rest * rest).sum() / max(float(weight.sum()), 1.0)))


def _region_fits(height: int, width: int, x: int, y: int, size: int, margin: int = PATCH_MARGIN) -> bool:
    return (
        x - margin >= 0
        and y - margin >= 0
        and x + size + margin <= width
        and y + size + margin <= height
    )


class Engine:
    def __init__(self, assets: Path | None = None):
        folder = assets or asset_dir()
        small = folder / "bg_48.png"
        large = folder / "bg_96.png"
        faint = folder / "bg_96_20260520.png"
        missing = [path.name for path in (small, large, faint) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Sparkle templates are missing from {folder}: {', '.join(missing)}")
        self._alpha48 = _load_alpha_png(small)
        self._alpha96 = _load_alpha_png(large)
        self._alpha96_2026 = _load_alpha_png(faint)
        self._cache: dict[tuple[int, float, str], Prepared] = {}
        self._lock = threading.Lock()

    def _base(self, size: int, variant: str) -> np.ndarray:
        if variant == "2026":
            return self._alpha96_2026
        if size == 48:
            return self._alpha48
        return self._alpha96

    def _prepare(self, size: int, scale: float, variant: str = "classic") -> Prepared:
        key = (size, round(float(scale), 3), variant)
        with self._lock:
            prepared = self._cache.get(key)
            if prepared is None:
                alpha = np.clip(_scale_alpha(self._base(size, variant), size) * np.float32(scale), 0.0, 1.0)
                alpha = np.ascontiguousarray(alpha, dtype=np.float32)
                prepared = Prepared(alpha, _silhouette_weight(alpha), float(scale), size, variant)
                self._cache[key] = prepared
            return prepared

    def _variant_scales(self, size: int) -> tuple[tuple[str, tuple[float, ...]], ...]:
        # Both captures can appear at any export size once the chat app or an
        # editor has resized the picture. The 2026 star keeps its shape when
        # scaled down, so do not restrict it to the 96px anchor.
        return (("classic", SCALES), ("2026", SCALES_2026))

    def _score(self, rgb: np.ndarray, x: int, y: int, prepared: Prepared) -> tuple[float, float] | None:
        measured = _measure(rgb, x, y, prepared)
        if measured is None:
            return None
        survival, after = measured
        if survival > MAX_SURVIVAL:
            return None
        control = self._control_energy(rgb, x, y, prepared)
        size = prepared.size
        region = rgb[y : y + size, x : x + size]
        template = self._prepare(size, 1.0, prepared.variant).alpha
        shaped = _matches_sparkle(region, template)

        # Flat, dark, or near-uniform corners provide no reliable control edges.
        # On near-white backgrounds, lift is small (<14) so shaped fails, but
        # outline collapse is very clean (survival <= 0.10) and clip is 0.
        if control < 1.0:
            clip = _clip_fraction(region, prepared.alpha)
            if shaped or (survival <= 0.10 and clip < 0.02):
                return survival, 0.0
            return None

        ratio = after / control
        if ratio <= MAX_CONTROL_RATIO:
            clip = _clip_fraction(region, prepared.alpha)
            # On textured images where background details mask the shape, outline collapse
            # must not produce heavy clipping. Unshaped noise artifacts clip heavily.
            if (survival <= 0.55 and clip < 0.10) or shaped:
                return survival, ratio
            return None
        # Smooth cloth, dark water, and JPEG noise make the neighborhood
        # comparison useless. The patch still has to be the sparkle.
        if not shaped:
            return None
        return survival, ratio

    def _control_energy(self, rgb: np.ndarray, x: int, y: int, prepared: Prepared) -> float:
        size = prepared.size
        near = size + 8
        offsets = (
            (-near, 0),
            (0, -near),
            (-near, -near),
            (-2 * size, 0),
            (0, -2 * size),
            (-2 * size, -2 * size),
        )
        height, width = rgb.shape[:2]
        samples: list[float] = []
        margin = PATCH_MARGIN
        for ox, oy in offsets:
            cx = x + ox
            cy = y + oy
            if not _region_fits(height, width, cx, cy, size, margin):
                continue
            patch = rgb[cy - margin : cy + size + margin, cx - margin : cx + size + margin]
            samples.append(_edge_energy(patch, prepared.weight, margin))
        if not samples:
            return 0.0
        samples.sort()
        return samples[len(samples) // 2]

    def _consider(
        self,
        rgb: np.ndarray,
        size: int,
        inset: int,
        scale: float,
        best: Match | None,
        variant: str = "classic",
        inset_y: int | None = None,
    ) -> Match | None:
        height, width = rgb.shape[:2]
        bottom = inset if inset_y is None else int(inset_y)
        x = width - inset - size
        y = height - bottom - size
        prepared = self._prepare(size, scale, variant)
        scored = self._score(rgb, x, y, prepared)
        if scored is None:
            return best
        survival, ratio = scored
        region = rgb[y : y + size, x : x + size]
        cleaned = _reverse_region(region, prepared.alpha)
        template = self._prepare(size, 1.0, variant).alpha
        lift = _residual_lift(cleaned, template)
        clip = _clip_fraction(region, prepared.alpha)
        candidate = Match(x, y, inset, prepared, survival, ratio, lift, clip, inset, bottom)
        if best is None or _prefer(candidate, best):
            return candidate
        return best

    def _locate(self, rgb: np.ndarray) -> list[tuple[float, int, int, int]]:
        """The likeliest stars in the corner: (outline left, size, right, bottom).

        Sliding correlation over the whole corner costs one transform per
        size, so it finds a star at any margin and rules out clean pictures
        quickly. It uses the shape alone, which works at any opacity.
        """
        height, width = rgb.shape[:2]
        short = min(height, width)
        side = min(short, max(320, min(560, int(round(0.36 * short)))))
        if side < 48:
            return []
        gray = rgb[height - side :, width - side :].astype(np.float32).mean(axis=2)
        spectrum = np.fft.rfft2(gray)
        sums = np.zeros((side + 1, side + 1), dtype=np.float64)
        squares = np.zeros((side + 1, side + 1), dtype=np.float64)
        sums[1:, 1:] = gray.astype(np.float64).cumsum(axis=0).cumsum(axis=1)
        squares[1:, 1:] = (gray.astype(np.float64) ** 2).cumsum(axis=0).cumsum(axis=1)

        def box(table: np.ndarray, size: int) -> np.ndarray:
            return table[size:, size:] - table[:-size, size:] - table[size:, :-size] + table[:-size, :-size]

        limit = max(40, min(120, int(round(0.13 * short))))
        peaks: list[tuple[float, int, int, int]] = []
        for size in range(24, min(limit, side - 8) + 1, 4):
            template = self._prepare(size, 1.0).alpha
            zero = template - float(template.mean())
            norm = float(np.sqrt((zero * zero).sum()))
            if norm <= 0.0:
                continue
            filled = np.fft.irfft2(spectrum * np.conj(np.fft.rfft2(zero, s=gray.shape)), s=gray.shape)
            count = float(size * size)
            total = box(sums, size)
            variance = box(squares, size) - total * total / count
            top = filled[: side - size + 1, : side - size + 1]
            # A flat patch has no star on it, and its correlation is only noise.
            score = np.where(variance > count * 1.5, top / (norm * np.sqrt(np.maximum(variance, 1e-6))), -1.0)
            spot = int(score.argmax())
            y, x = np.unravel_index(spot, score.shape)
            value = float(score[y, x])
            if value < MIN_PEAK_NCC:
                continue
            right = int(side - x - size)
            bottom = int(side - y - size)
            # A small star fits any blob. Removing it is what tells them apart.
            trial = [
                _measure(rgb, width - right - size, height - bottom - size, self._prepare(size, scale))
                for scale in (1.0, 0.6)
            ]
            left = [item[0] for item in trial if item is not None]
            if left:
                peaks.append((min(left), size, right, bottom))
        peaks.sort()
        return peaks[:MAX_PEAKS]

    def _corner_search(self, rgb: np.ndarray, best: Match | None) -> Match | None:
        height, width = rgb.shape[:2]
        for _value, size0, right0, bottom0 in self._locate(rgb):
            x0 = width - right0 - size0
            y0 = height - bottom0 - size0
            fit = self._fit_shape(rgb, x0, y0, size0)
            if fit is None:
                continue
            strength, _rest, variant = fit
            scale = round(strength, 2)
            # Settle the exact spot on the outline alone, then score that one.
            spot: tuple[float, int, int, int] | None = None
            for size in range(max(20, size0 - 2), size0 + 3):
                prepared = self._prepare(size, scale, variant)
                for right in (right0 - 1, right0, right0 + 1):
                    for bottom in (bottom0 - 1, bottom0, bottom0 + 1):
                        if right < 0 or bottom < 0:
                            continue
                        measured = _measure(rgb, width - right - size, height - bottom - size, prepared)
                        if measured is not None and (spot is None or measured[0] < spot[0]):
                            spot = (measured[0], size, right, bottom)
            if spot is None:
                continue
            _survival, size, right, bottom = spot
            if best is not None:
                if (best.size, best.right, best.bottom) == (size, right, bottom):
                    continue  # The listed opacities already covered this layout.
                held = self._fit_shape(rgb, best.x, best.y, best.size)
                moved = self._fit_shape(rgb, width - right - size, height - bottom - size, size)
                # A known layout keeps its place unless this one clearly fits better.
                if held is not None and (moved is None or moved[1] > 0.8 * held[1]):
                    continue
            best = self._consider(rgb, size, right, scale, best, variant, bottom)
        return best

    def _fit_shape(self, rgb: np.ndarray, x: int, y: int, size: int) -> tuple[float, float, str] | None:
        """The capture and opacity that explain this patch best."""
        pad = 6
        height, width = rgb.shape[:2]
        if x - pad < 0 or y - pad < 0 or x + size + pad > width or y + size + pad > height:
            return None
        gray = rgb[y - pad : y + size + pad, x - pad : x + size + pad].astype(np.float32).mean(axis=2)
        best: tuple[float, float, str] | None = None
        for variant in ("classic", "2026"):
            alpha = np.pad(self._prepare(size, 1.0, variant).alpha, pad)
            k, rest, _back = _fit_opacity(gray, alpha, _grow(alpha > 0.0, 1))
            if k < 0.15 or k > 1.6:
                continue
            if best is None or rest < best[1]:
                best = (k, rest, variant)
        return best

    def refine(self, rgb: np.ndarray, match: Match) -> Match:
        """Replace a listed opacity with the one measured from the pixels."""
        size = match.size
        # The outline is already gone, so a measured value could only add texture.
        # Clipping fakes that result: too strong a star on black also reads as flat.
        if match.survival <= 0.05 and match.clip <= 0.02:
            return match
        fit = self._fit_shape(rgb, match.x, match.y, size)
        if fit is None:
            return match
        strength, rest, variant = fit
        pad = 6
        gray = rgb[match.y - pad : match.y + size + pad, match.x - pad : match.x + size + pad]
        unit = np.pad(match.prepared.alpha, pad)
        held = _fit_rest(gray.astype(np.float32).mean(axis=2), unit, _grow(unit > 0.0, 1))
        # Texture can always be fitted a little. Only a clearly better fit counts.
        if rest > 0.8 * held:
            return match
        strength = 1.0 if abs(strength - 1.0) < 0.04 else round(strength, 2)
        if variant == match.prepared.variant and abs(strength - match.prepared.scale) < 0.015:
            return match
        prepared = self._prepare(size, strength, variant)
        region = rgb[match.y : match.y + size, match.x : match.x + size]
        template = self._prepare(size, 1.0, variant).alpha
        lift = _residual_lift(_reverse_region(region, prepared.alpha), template)
        measured = _measure(rgb, match.x, match.y, prepared)
        survival = match.survival if measured is None else measured[0]
        # The fit reads texture as opacity too, so it has to show a clear gain.
        flatter = abs(lift) + 2.0 < abs(match.residual_lift)
        sharper = survival + 0.03 < match.survival and abs(lift) <= abs(match.residual_lift) + 1.0
        if not (flatter or sharper) or survival > match.survival + 0.05:
            return match
        return Match(
            match.x, match.y, match.inset, prepared, survival, match.control_ratio, lift,
            _clip_fraction(region, prepared.alpha), match.inset_x, match.inset_y,
        )

    def find_sparkle(self, rgb: np.ndarray) -> Match | None:
        best: Match | None = None
        height, width = rgb.shape[:2]
        for size, right, bottom in v2_layouts(width, height):
            for scale in SCALES_2026:
                best = self._consider(rgb, size, right, scale, best, "2026", bottom)
            for scale in (1.0, 0.70, 0.40):
                best = self._consider(rgb, size, right, scale, best, "classic", bottom)
        for size, inset in ANCHORS:
            for variant, scales in self._variant_scales(size):
                for scale in scales:
                    best = self._consider(rgb, size, inset, scale, best, variant)

        # A clean, exact match needs no wider search. A weak or clipped hit
        # may be the wrong anchor size, so check the rest of the corner too.
        if best is None or best.survival > 0.30 or best.clip > 0.05:
            best = self._corner_search(rgb, best)
        return None if best is None else self.refine(rgb, best)

    def clean_with_match(self, rgb: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, CleanInfo, Match | None]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")

        if mode == "auto":
            match = self.find_sparkle(rgb)
            if match is None:
                return rgb, CleanInfo("skipped", "No sparkle found"), None
        else:
            match = self._forced_match(rgb, mode)
            if match is None:
                return rgb, CleanInfo("failed", "Image is too small for this watermark"), None

        detail = f"{match.size}px sparkle, {match.inset}px from the corner"
        return apply_sparkle(rgb, match), CleanInfo(
            "removed",
            detail,
            size=match.size,
            inset=match.inset,
            survival=match.survival,
        ), match

    def clean_rgb(self, rgb: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, CleanInfo]:
        image, info, _match = self.clean_with_match(rgb, mode)
        return image, info

    def _forced_match(self, rgb: np.ndarray, mode: str) -> Match | None:
        height, width = rgb.shape[:2]
        if mode == "standard":
            size, inset = (96, 64) if width > 1024 and height > 1024 else (48, 32)
        else:
            size, inset = 48, 96
        x = width - inset - size
        y = height - inset - size
        prepared = self._prepare(size, 1.0)
        if x < 0 or y < 0 or x + size > width or y + size > height:
            return None
        # Forced modes do not have to pass detection. Survival is still recorded
        # when the patch has room for the outline measurement.
        scored = None
        if _region_fits(height, width, x, y, size):
            scored = self._score(rgb, x, y, prepared)
        survival = 1.0 if scored is None else scored[0]
        ratio = 0.0 if scored is None else scored[1]
        return Match(x, y, inset, prepared, survival, ratio)

    def match_layout(self, rgb: np.ndarray, mode: str) -> Match | None:
        """Score the Standard or Current corner. None if it is not the sparkle."""
        match = self._forced_match(rgb, mode)
        if match is None:
            return None
        scored = self._score(rgb, match.x, match.y, match.prepared)
        if scored is None:
            return None
        survival, ratio = scored
        size = match.size
        region = rgb[match.y : match.y + size, match.x : match.x + size]
        cleaned = _reverse_region(region, match.prepared.alpha)
        template = self._prepare(size, 1.0, match.prepared.variant).alpha
        lift = _residual_lift(cleaned, template)
        return Match(match.x, match.y, match.inset, match.prepared, survival, ratio, lift)


def _prefer(candidate: Match, incumbent: Match) -> bool:
    """Lower outline survival wins.

    Two captures of the same star can land close on that test while one of
    them still leaves the star bright. On the same layout, take the removal
    that actually flattens it. Neighboring opacities of one capture break a
    remaining tie toward the captured strength.
    """
    same_layout = (candidate.size, candidate.right, candidate.bottom) == (
        incumbent.size,
        incumbent.right,
        incumbent.bottom,
    )

    if same_layout:
        surv_diff = abs(candidate.survival - incumbent.survival)

        # On dark backgrounds, an over-strong opacity clips and craters.
        # Prefer the lower-clip candidate only if it actually removes the watermark
        # (does not leave a bright star with lift >= 6.0 or terrible survival).
        if candidate.clip >= 0.20 and incumbent.clip < 0.12 and incumbent.residual_lift < 10.0:
            return False
        if incumbent.clip >= 0.20 and candidate.clip < 0.12 and candidate.residual_lift < 10.0:
            return True
        if (
            candidate.clip + 0.03 < incumbent.clip
            and candidate.residual_lift < 6.0
            and candidate.survival <= incumbent.survival + 0.03
        ):
            return True
        if (
            incumbent.clip + 0.03 < candidate.clip
            and incumbent.residual_lift < 6.0
            and incumbent.survival <= candidate.survival + 0.03
        ):
            return False

        different_capture = candidate.prepared.variant != incumbent.prepared.variant
        if (
            different_capture
            and surv_diff <= 0.06
            and abs(abs(candidate.residual_lift) - abs(incumbent.residual_lift)) >= 4.0
        ):
            return abs(candidate.residual_lift) < abs(incumbent.residual_lift)

        same_capture = candidate.prepared.variant == incumbent.prepared.variant
        close_opacity = abs(candidate.prepared.scale - incumbent.prepared.scale) <= 0.16
        if same_capture and close_opacity and surv_diff <= 0.03:
            # Only the opacity differs here, so the outline test is nearly
            # blind and a too-strong alpha can score slightly better while it
            # punches a dark hole. A near-flat corner is the clean removal.
            # Textured backgrounds shift both numbers, so only override when
            # exactly one candidate is flat; ranking two uneven values against
            # each other is what over-removed exact marks before.
            flat = 8.0
            candidate_flat = abs(candidate.residual_lift) <= flat
            incumbent_flat = abs(incumbent.residual_lift) <= flat
            if candidate_flat != incumbent_flat:
                return candidate_flat
            if surv_diff <= 0.002:
                return abs(candidate.prepared.scale - 1.0) < abs(incumbent.prepared.scale - 1.0)

        return candidate.survival < incumbent.survival

    # Different layouts: reject severe clipping candidates (>0.40) competing with clean candidates
    if candidate.clip > 0.40 and incumbent.clip < 0.20:
        return False
    if incumbent.clip > 0.40 and candidate.clip < 0.20:
        return True

    # Near-equal outlines mean both layouts fit the same corner. In that case
    # a candidate that leaves a bright star (positive lift) or punches a hole
    # (negative lift) should lose to one that restores the corner cleanly.
    surv_diff = abs(candidate.survival - incumbent.survival)
    if surv_diff <= 0.10:
        cand_lift = abs(candidate.residual_lift)
        inc_lift = abs(incumbent.residual_lift)
        if abs(cand_lift - inc_lift) >= 4.0:
            return cand_lift < inc_lift

    return candidate.survival < incumbent.survival


def _soften_dark(region: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Calm the noise reversal amplifies on dark video.

    Dividing by 1 - alpha doubles compression noise under the star. On a dark
    frame that noise is the star's ghost, and there is little detail to lose.
    A bright frame keeps every pixel.
    """
    pixels = region.astype(np.float32)
    blurred = pixels.copy()
    for axis in (0, 1):
        total = blurred.copy()
        for step in (1, 2):
            total = total + np.roll(blurred, step, axis) + np.roll(blurred, -step, axis)
        blurred = total / 5.0
    dark = np.clip((95.0 - blurred.mean(axis=2)) / 55.0, 0.0, 1.0)
    weight = (0.85 * dark * np.clip(alpha / 0.08, 0.0, 1.0))[..., None]
    return np.clip(np.rint(pixels * (1.0 - weight) + blurred * weight), 0, 255).astype(np.uint8)


def apply_sparkle(rgb: np.ndarray, match: Match) -> np.ndarray:
    """Reverse one known sparkle. Later video frames reuse the first match."""
    size = match.size
    height, width = rgb.shape[:2]
    x = width - match.right - size
    y = height - match.bottom - size
    if x < 0 or y < 0 or x + size > width or y + size > height:
        return rgb
    cleaned = _reverse_region(rgb[y : y + size, x : x + size], match.prepared.alpha)
    out = rgb.copy()
    out[y : y + size, x : x + size] = cleaned
    return out


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def collect_images(folder: Path, recursive: bool, exclude: Path | None = None) -> list[Path]:
    """Images and videos. Upscale skips videos later."""
    folder = folder.resolve()
    excluded = exclude.resolve() if exclude is not None else None
    pattern = "**/*" if recursive else "*"
    found: list[Path] = []
    for path in folder.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if excluded is not None:
            try:
                path.resolve().relative_to(excluded)
            except ValueError:
                pass
            else:
                continue
        found.append(path)
    found.sort()
    return found


def _save_image(
    dest: Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    jpeg_quality: int,
    exif: bytes | None,
    icc: bytes | None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if alpha is not None:
        image = image.convert("RGBA")
        image.putalpha(Image.fromarray(alpha, mode="L"))
    extension = dest.suffix.lower()
    options: dict = {}
    if icc:
        options["icc_profile"] = icc
    if extension in {".jpg", ".jpeg"}:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if exif:
            options["exif"] = exif
        image.save(dest, quality=jpeg_quality, subsampling=0, **options)
    elif extension == ".png":
        image.save(dest, compress_level=3, **options)
    elif extension == ".webp":
        image.save(dest, quality=jpeg_quality, method=4, **options)
    elif extension in {".tif", ".tiff"}:
        image.save(dest, **options)
    elif extension == ".bmp":
        image.convert("RGB").save(dest)
    else:
        image.save(dest.with_suffix(".png"), compress_level=3, **options)


def _fit_alpha(alpha: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    """Resize a transparency mask to the new image size. The upscaler never sees it."""
    if alpha is None:
        return None
    if alpha.shape[0] == height and alpha.shape[1] == width:
        return alpha
    plate = Image.fromarray(alpha, mode="L").resize((width, height), Image.Resampling.LANCZOS)
    return gray_array(plate)


def remove_visible(engine: Engine, rgb: np.ndarray, mode: str, mark: str = "auto") -> tuple[np.ndarray, CleanInfo]:
    """Gemini sparkle, or a visible corner label from another generator."""
    from marks import MARK_IDS, MARKS, clear_labels

    mark = str(mark or "auto")
    if mark not in MARK_IDS:
        return rgb, CleanInfo("failed", f"Unknown mark {mark}")
    if mark == "gemini":
        return engine.clean_rgb(rgb, mode)
    if mark == "auto":
        if mode in ("standard", "current"):
            hinted = engine.match_layout(rgb, mode)
            if hinted is not None:
                detail = f"{hinted.size}px sparkle, {hinted.inset}px from the corner"
                return apply_sparkle(rgb, hinted), CleanInfo(
                    "removed",
                    detail,
                    size=hinted.size,
                    inset=hinted.inset,
                    survival=hinted.survival,
                )
        cleaned, info = engine.clean_rgb(rgb, "auto")
        if info.status == "removed":
            return cleaned, info
        hit = clear_labels(rgb, None)
        if hit is not None:
            return hit.image, CleanInfo("removed", hit.detail)
        if mode in ("standard", "current"):
            # A fixed corner was asked for and nothing else matched, so clean it.
            return engine.clean_rgb(rgb, mode)
        return rgb, CleanInfo("skipped", "No visible mark found")
    hit = clear_labels(rgb, mark)
    if hit is None:
        title = next(item.title for item in MARKS if item.id == mark)
        return rgb, CleanInfo("skipped", f"No {title} mark found")
    return hit.image, CleanInfo("removed", hit.detail)


def process_file(
    engine: Engine,
    source: Path,
    dest: Path,
    mode: str = "auto",
    jpeg_quality: int = 95,
    task: str = "clean",
    upscaler=None,
    model: str = "photo",
    scale: int = 4,
    cancel: threading.Event | None = None,
    mark: str = "auto",
    keep_audio: bool = True,
    video_crf: int = 18,
    combine: bool = False,
    on_progress=None,
) -> CleanInfo:
    """Remove a visible mark, upscale a picture, or both when combine is set.

    A video is cleaned and is not enlarged. Upscale on a video is skipped.
    A file with no visible mark is copied unchanged when the job is only Remove.
    Originals are never overwritten.
    """
    jpeg_quality = int(min(100, max(70, jpeg_quality)))
    if task not in {"clean", "upscale"}:
        return CleanInfo("failed", "Choose Remove or Upscale.")
    combine = bool(combine)
    wants_remove = task == "clean" or combine
    wants_upscale = task == "upscale" or combine
    try:
        if source.resolve() == dest.resolve():
            return CleanInfo("failed", "Refusing to overwrite the original")
        if is_video(source):
            if wants_upscale and not wants_remove:
                return CleanInfo("skipped", "Videos are not upscaled.")
            from video import process_video

            return process_video(
                engine,
                source,
                dest,
                mode,
                mark,
                cancel=cancel,
                keep_audio=keep_audio,
                crf=video_crf,
                upscaler=None,
                model=model,
                scale=scale,
                on_progress=on_progress,
                remove=True,
            )
        with Image.open(source) as opened:
            oriented = ImageOps.exif_transpose(opened)
            if oriented is None:
                oriented = opened
            alpha = None
            if oriented.mode in {"RGBA", "LA"} or (oriented.mode == "P" and "transparency" in oriented.info):
                alpha = gray_array(oriented.convert("RGBA").getchannel("A"))
            rgb = rgb_array(oriented)
            exif_obj = oriented.getexif()
            exif_obj.pop(274, None)
            exif = exif_obj.tobytes() if len(exif_obj) else None
            icc = oriented.info.get("icc_profile")
            if isinstance(icc, str):
                icc = icc.encode("latin1")

        if wants_remove and wants_upscale:
            if upscaler is None:
                return CleanInfo("failed", "No upscaler is loaded")
            cleaned, info = remove_visible(engine, rgb, mode, mark)
            if info.status == "failed":
                return info
            scaled = upscaler.upscale(cleaned, model, scale, cancel=cancel)
            phrase = upscaler.describe(model, scale)
            _save_image(
                dest,
                scaled,
                _fit_alpha(alpha, scaled.shape[1], scaled.shape[0]),
                jpeg_quality,
                exif,
                icc,
            )
            if info.status == "removed":
                return CleanInfo("removed", f"{info.detail}. {phrase}")
            return CleanInfo("upscaled", f"{info.detail}. {phrase}")

        if task == "clean":
            cleaned, info = remove_visible(engine, rgb, mode, mark)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if info.status == "skipped":
                shutil.copy2(source, dest)
                return info
            if info.status != "removed":
                return info
            _save_image(dest, cleaned, alpha, jpeg_quality, exif, icc)
            return info

        if upscaler is None:
            return CleanInfo("failed", "No upscaler is loaded")

        scaled = upscaler.upscale(rgb, model, scale, cancel=cancel)
        phrase = upscaler.describe(model, scale)
        _save_image(dest, scaled, _fit_alpha(alpha, scaled.shape[1], scaled.shape[0]), jpeg_quality, exif, icc)
        return CleanInfo("upscaled", phrase)
    except Stopped:
        return CleanInfo("failed", "Stopped")
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
        return CleanInfo("failed", str(error))


def prepare_frame_removal(engine: Engine, rgb: np.ndarray, mode: str, mark: str = "auto"):
    """Detect on one frame and return a repeater for the same corner.

    The mark stays in one corner. A match from one clear frame is painted
    onto the rest, including a frame whose own check missed.
    """
    from marks import MARK_IDS, MARKS, clear_labels

    mark = str(mark or "auto")
    if mark not in MARK_IDS:
        return rgb, CleanInfo("failed", f"Unknown mark {mark}"), None
    if mark == "gemini":
        cleaned, info, found = engine.clean_with_match(rgb, mode)
        repeater = SparkleRepeater(engine, found) if info.status == "removed" and found is not None else None
        return cleaned, info, repeater
    if mark == "auto":
        if mode in ("standard", "current"):
            hinted = engine.match_layout(rgb, mode)
            if hinted is not None:
                detail = f"{hinted.size}px sparkle, {hinted.inset}px from the corner"
                info = CleanInfo("removed", detail, size=hinted.size, inset=hinted.inset, survival=hinted.survival)
                return apply_sparkle(rgb, hinted), info, SparkleRepeater(engine, hinted)
        cleaned, info, found = engine.clean_with_match(rgb, "auto")
        if info.status == "failed":
            return cleaned, info, None
        if info.status == "removed" and found is not None:
            return cleaned, info, SparkleRepeater(engine, found)
        hit = clear_labels(rgb, None)
        if hit is not None:
            return hit.image, CleanInfo("removed", hit.detail), LabelRepeater(hit)
        if mode in ("standard", "current"):
            cleaned, info, found = engine.clean_with_match(rgb, mode)
            return cleaned, info, SparkleRepeater(engine, found) if found is not None else None
        return rgb, CleanInfo("skipped", "No visible mark found"), None
    hit = clear_labels(rgb, mark)
    if hit is None:
        title = next(item.title for item in MARKS if item.id == mark)
        return rgb, CleanInfo("skipped", f"No {title} mark found"), None
    return hit.image, CleanInfo("removed", hit.detail), LabelRepeater(hit)


class SparkleRepeater:
    """Clean one known sparkle on every frame of a video.

    tune() studies frames from the clip. It replaces the stored capture with
    the star this clip actually carries, which is what leaves dark scenes
    clean, and works out whether the mark is on every frame. When it is not,
    each frame is measured so a frame without the mark stays as it was.
    """

    PAD = 8

    def __init__(self, engine: Engine, match: Match) -> None:
        self.engine = engine
        self.match = match
        self.gated = False
        self._alpha: np.ndarray | None = None
        self._strength = 0.0

    def _crop(self, frame: np.ndarray) -> np.ndarray | None:
        match = self.match
        height, width = frame.shape[:2]
        size = match.size
        x = width - match.right - size
        y = height - match.bottom - size
        pad = self.PAD
        if x - pad < 0 or y - pad < 0 or x + size + pad > width or y + size + pad > height:
            return None
        return frame[y - pad : y + size + pad, x - pad : x + size + pad]

    def tune(self, frames: list[np.ndarray]) -> None:
        crops = [crop for crop in (self._crop(frame) for frame in frames) if crop is not None]
        if len(crops) < 3:
            return
        pad = self.PAD
        shape = np.pad(self.match.prepared.alpha, pad)
        unit = shape / max(float(self.match.prepared.scale), 0.01)
        hole = _grow(shape > 0.0, 1)
        grays = [crop.astype(np.float32).mean(axis=2) for crop in crops]
        fits = [_fit_opacity(gray, unit, hole) for gray in grays]
        strengths = np.array([fit[0] for fit in fits], dtype=np.float64)
        top = float(np.percentile(strengths, 80))
        if top < 0.2:
            return
        present = strengths >= max(0.2, 0.45 * top)
        estimates = []
        for gray, fit, here in zip(grays, fits, present):
            if here:
                back = fit[2]
                estimates.append((gray - back) / np.maximum(255.0 - back, 1.0))
        typical = float(np.median(strengths[present]))
        base = unit * typical
        layers = np.stack(estimates)
        found = np.median(layers, axis=0).astype(np.float32)
        # Scene detail the median could not remove must not become part of the star.
        # Detail that never moves cannot be told from the star, so a still,
        # textured shot stays close to the capture.
        texture = float(np.median([gray[~hole].std() for gray in grays]))
        motion = float(np.median(layers.std(axis=0)[hole]))
        drift = MAX_ALPHA_DRIFT if texture <= 6.0 else float(np.clip(2.0 * motion, 0.02, MAX_ALPHA_DRIFT))
        found = np.clip(found, base - drift, base + drift)
        found = np.where(hole, np.clip(found, 0.0, MAX_ALPHA), 0.0).astype(np.float32)
        self._alpha = found[pad:-pad, pad:-pad]
        self._strength = typical
        steady = float(strengths[present].std()) <= 0.12 * max(typical, 0.05)
        self.gated = not (bool(present.all()) and steady)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        match = self.match
        alpha = match.prepared.alpha if self._alpha is None else self._alpha
        amount = 1.0
        if self.gated:
            crop = self._crop(frame)
            if crop is None:
                return frame
            pad = self.PAD
            unit = np.pad(alpha, pad)
            k, _rest, _back = _fit_opacity(crop.astype(np.float32).mean(axis=2), unit, _grow(unit > 0.0, 1))
            if k < 0.2:
                return frame
            amount = min(1.0, k)
            # Texture can fake an opacity. Reversal only erases an outline that is there.
            margin = pad - PATCH_MARGIN
            patch = crop[margin:-margin, margin:-margin]
            weight = _silhouette_weight(alpha)
            before = _edge_energy(patch, weight)
            if before < 1.0:
                return frame
            trial = patch.copy()
            end = PATCH_MARGIN + alpha.shape[0]
            trial[PATCH_MARGIN:end, PATCH_MARGIN:end] = _reverse_region(
                patch[PATCH_MARGIN:end, PATCH_MARGIN:end], alpha * amount, fill=False
            )
            if _edge_energy(trial, weight) / before > MAX_SURVIVAL:
                return frame
        size = match.size
        height, width = frame.shape[:2]
        x = width - match.right - size
        y = height - match.bottom - size
        if x < 0 or y < 0:
            return frame
        out = frame.copy()
        cleaned = _reverse_region(frame[y : y + size, x : x + size], alpha * amount)
        out[y : y + size, x : x + size] = _soften_dark(cleaned, alpha * amount)
        return out


class LabelRepeater:
    """Clean one known corner label with the same mask on every frame."""

    def __init__(self, hit) -> None:
        self.hit = hit
        self._mask = hit.mask

    def tune(self, frames: list[np.ndarray]) -> None:
        from marks import stable_mask

        self._mask = stable_mask(frames, self.hit)

    def present_on(self, frames: list[np.ndarray]) -> bool:
        """A mark stays put. A subtitle or a caption does not."""
        from marks import mask_share

        return mask_share(frames, self.hit) >= 0.5

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        from marks import apply_mask

        return apply_mask(frame, self.hit, self._mask)
