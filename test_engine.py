"""Checks for the sparkle remover. Run: .venv\\Scripts\\python.exe test_engine.py"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from engine import Engine, _grow, _reverse_region, collect_images, process_file

engine = Engine()


def stamp(rgb: np.ndarray, x: int, y: int, alpha: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    height, width = alpha.shape
    region = out[y : y + height, x : x + width].astype(np.float32)
    blended = alpha[..., None] * 255.0 + (1.0 - alpha[..., None]) * region
    out[y : y + height, x : x + width] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out


def place(rgb: np.ndarray, size: int, inset: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    prepared = engine._prepare(size, scale)
    height, width = rgb.shape[:2]
    x = width - inset - size
    y = height - inset - size
    return stamp(rgb, x, y, prepared.alpha), prepared.alpha


def photo(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.random((max(2, height // 8), max(2, width // 8)))
    plate = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    plate = plate.resize((width, height), Image.Resampling.BICUBIC)
    base = np.asarray(plate, dtype=np.float32)
    ys = np.linspace(0, 50, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 35, width, dtype=np.float32)[None, :]
    rgb = np.dstack((base * 0.55 + 30 + ys, base * 0.45 + 60, base * 0.35 + 80 + xs))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def flat(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def blob(rgb: np.ndarray, x: int, y: int, size: int, color: int = 235) -> np.ndarray:
    out = rgb.copy()
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    alpha = np.clip(1.0 - radius / (size * 0.42), 0.0, 1.0) ** 1.4
    region = out[y : y + size, x : x + size].astype(np.float32)
    mixed = alpha[..., None] * color + (1.0 - alpha[..., None]) * region
    out[y : y + size, x : x + size] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return out


def max_error(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def silk(height: int, width: int) -> np.ndarray:
    """Almost no texture, like the dark cloth behind the missed sparkles."""
    ys = np.linspace(0, 6, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 4, width, dtype=np.float32)[None, :]
    base = 18.0 + ys + xs
    rgb = np.dstack((base + 2.0, base, base + 4.0))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def expect_removed(name: str, original: np.ndarray, size: int, inset: int, scale: float = 1.0) -> None:
    marked, _alpha = place(original, size, inset, scale)
    cleaned, info = engine.clean_rgb(marked, "auto")
    error = max_error(cleaned, original)
    print(f"  {name}: {info.status} {info.detail} survival={info.survival} err={error}")
    assert info.status == "removed", info.detail
    assert info.size == size, (info.size, size)
    assert info.inset == inset, (info.inset, inset)
    assert error <= 2, error


def expect_skipped(name: str, image: np.ndarray) -> None:
    cleaned, info = engine.clean_rgb(image, "auto")
    print(f"  {name}: {info.status} {info.detail}")
    assert info.status == "skipped", info.detail
    assert cleaned is image


def test_math_roundtrip() -> None:
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    recovered = _reverse_region(stamp(original, 0, 0, engine._alpha48), engine._alpha48)
    error = max_error(recovered, original)
    print(f"  exact 48 alpha roundtrip err={error}")
    assert error <= 1

    original96 = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    recovered96 = _reverse_region(stamp(original96, 0, 0, engine._alpha96), engine._alpha96)
    error96 = max_error(recovered96, original96)
    print(f"  exact 96 alpha roundtrip err={error96}")
    assert error96 <= 1


def test_known_layouts() -> None:
    expect_removed("flat 48/32", flat(700, 900, (140, 150, 160)), 48, 32)
    expect_removed("flat 48/96", flat(768, 1408, (90, 110, 130)), 48, 96)
    expect_removed("flat 48/64", flat(800, 800, (40, 40, 40)), 48, 64)
    expect_removed("flat 96/64", flat(1400, 1600, (180, 170, 160)), 96, 64)
    expect_removed("flat 96/192", flat(1600, 2000, (100, 120, 80)), 96, 192)
    expect_removed("flat 36/96", flat(900, 1200, (70, 80, 90)), 36, 96)
    expect_removed("photo 48/96", photo(1024, 1024, 3), 48, 96)
    expect_removed("photo 96/64", photo(1400, 1400, 4), 96, 64)
    expect_removed("faint 48/96", photo(900, 1200, 5), 48, 96, scale=0.70)
    expect_removed("faint 96/192", photo(1500, 1800, 16), 96, 192, scale=0.40)
    expect_removed("faint silk 96/192", silk(1500, 1800), 96, 192, scale=0.40)
    expect_removed("faint off-grid 52/80", flat(900, 1100, (150, 140, 130)), 52, 80, scale=0.40)
    expect_removed("dark 48/32", flat(640, 640, (12, 14, 18)), 48, 32)
    expect_removed("near-white 48/32", flat(640, 640, (248, 246, 244)), 48, 32)
    expect_removed("off-grid 52/80", flat(900, 1100, (150, 140, 130)), 52, 80)


def test_dark_backgrounds() -> None:
    """Busy or smooth dark corners still match, and a leftover star loses to a flatter capture."""
    rng = np.random.default_rng(22)
    noise = rng.normal(0, 7.0, (1500, 1800, 3))
    dark = np.clip(16 + noise + np.linspace(0, 5, 1500)[:, None, None], 0, 255).astype(np.uint8)
    expect_removed("dark noise 96/192 faint", dark, 96, 192, 0.40)
    expect_removed("dark noise 48/96", dark[:900, :1200], 48, 96, 0.70)
    alpha = engine._prepare(96, 1.0, "2026").alpha
    height, width = dark.shape[:2]
    marked = stamp(dark, width - 192 - 96, height - 192 - 96, alpha)
    cleaned, info = engine.clean_rgb(marked, "auto")
    error = max_error(cleaned, dark)
    print(f"  dark 2026: {info.status} {info.detail} survival={info.survival} err={error}")
    assert info.status == "removed", info.detail
    assert info.size == 96 and info.inset == 192
    assert error <= 4, error
    expect_skipped("clean dark noise", dark)


def test_auto_layout_does_not_force_gemini() -> None:
    from engine import remove_visible

    height, width = 700, 1000
    yy, xx = np.mgrid[0:height, 0:width]
    base = 48.0 + xx * 0.08 + yy * 0.04
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 2.5, (height, width))
    plain = np.clip(np.stack([base + noise, base * 0.92 + noise, base * 0.75 + 8], axis=2), 0, 255).astype(np.uint8)
    grok = plain.copy()
    _stamp_label(grok, 880, 652, bars=4)
    cleaned, info = remove_visible(engine, grok, "standard", "auto")
    print(f"  auto+standard grok: {info.status} {info.detail}")
    assert info.status == "removed", info.detail
    assert info.detail == "Grok mark", info.detail
    hinted = engine.match_layout(plain, "standard")
    assert hinted is None


def test_leaves_clean_images_alone() -> None:
    expect_skipped("clean flat", flat(800, 1000, (120, 130, 125)))
    expect_skipped("clean photo", photo(1024, 768, 8))
    expect_skipped("clean noise", np.random.default_rng(9).integers(0, 256, size=(700, 900, 3), dtype=np.uint8))
    marked = photo(1024, 1024, 2)
    height, width = marked.shape[:2]
    impostor = blob(marked, width - 96 - 48, height - 96 - 48, 48, color=240)
    expect_skipped("bright blob in the corner", impostor)
    expect_skipped("clean silk", silk(1500, 1800))
    glow = silk(1500, 1800)
    glow_h, glow_w = glow.shape[:2]
    impostor = blob(glow, glow_w - 192 - 96, glow_h - 192 - 96, 96, color=230)
    expect_skipped("bright blob on silk", impostor)


def test_forced_modes_and_files() -> None:
    original = photo(600, 800, 6)
    marked, _alpha = place(original, 48, 32, 1.0)
    cleaned, info = engine.clean_rgb(marked, "standard")
    print(f"  forced standard: {info.status} {info.detail} err={max_error(cleaned, original)}")
    assert info.status == "removed"
    assert max_error(cleaned, original) <= 2

    current_marked, _alpha = place(original, 48, 96, 1.0)
    cleaned, info = engine.clean_rgb(current_marked, "current")
    print(f"  forced current: {info.status} {info.detail} err={max_error(cleaned, original)}")
    assert info.status == "removed" and info.inset == 96
    assert max_error(cleaned, original) <= 2

    too_small = flat(40, 40, (10, 10, 10))
    _cleaned, info = engine.clean_rgb(too_small, "standard")
    assert info.status == "failed"

    root = Path(tempfile.mkdtemp(prefix="gwr-"))
    try:
        source_dir = root / "in" / "batch"
        nested = source_dir / "set"
        nested.mkdir(parents=True)
        clean = photo(480, 640, 11)
        marked_img, _alpha = place(photo(480, 640, 12), 48, 32, 1.0)
        Image.fromarray(clean, "RGB").save(source_dir / "clean.png")
        Image.fromarray(marked_img, "RGB").save(nested / "marked.png")
        original_bytes = (source_dir / "clean.png").read_bytes()
        output = root / "out"
        files = collect_images(source_dir, recursive=True, exclude=output)
        assert [path.name for path in files] == ["clean.png", "marked.png"]
        for path in files:
            info = process_file(engine, path, output / path.relative_to(source_dir), mode="auto")
            print(f"  file {path.name}: {info.status} {info.detail}")
        assert (output / "clean.png").read_bytes() == original_bytes
        assert (source_dir / "clean.png").read_bytes() == original_bytes
        saved = np.asarray(Image.open(output / "set" / "marked.png").convert("RGB"))
        assert max_error(saved, photo(480, 640, 12)) <= 2
        same = process_file(engine, source_dir / "clean.png", source_dir / "clean.png")
        assert same.status == "failed"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _corner_lift(rgb: np.ndarray, alpha: np.ndarray, size: int, inset: int) -> float:
    height, width = rgb.shape[:2]
    x = width - inset - size
    y = height - inset - size
    gray = rgb[y : y + size, x : x + size].astype(np.float32).mean(axis=2)
    high = alpha > 0.20
    low = alpha < 0.05
    return float(gray[high].mean() - gray[low].mean())


def test_2026_capture() -> None:
    original = photo(1500, 1800, 19)
    alpha = engine._prepare(96, 1.0, "2026").alpha
    height, width = original.shape[:2]
    marked = stamp(original, width - 192 - 96, height - 192 - 96, alpha)
    cleaned, info = engine.clean_rgb(marked, "auto")
    error = max_error(cleaned, original)
    print(f"  2026 capture: {info.status} {info.detail} survival={info.survival} err={error}")
    assert info.status == "removed", info.detail
    assert info.size == 96 and info.inset == 192
    assert error <= 2, error


def test_sample_folder() -> None:
    root = Path(__file__).resolve().parent / "portable"
    folder = root / "FrameHeart" / "Sample"
    if not folder.is_dir():
        folder = root / "Bulk AI Watermark Remover" / "Sample"
    if not folder.is_dir():
        print("  sample folder absent")
        return
    template = engine._prepare(96, 1.0).alpha
    for path in sorted(folder.glob("*.png")):
        original = np.asarray(Image.open(path).convert("RGB"))
        cleaned, info = engine.clean_rgb(original, "auto")
        match = engine.find_sparkle(original)
        before = _corner_lift(original, template, 96, 192)
        after = _corner_lift(cleaned, template, 96, 192)
        variant = None if match is None else match.prepared.variant
        scale = None if match is None else match.prepared.scale
        print(
            f"  {path.name}: {info.status} {info.detail} {variant} x{scale} "
            f"survival={info.survival} lift {before:.1f}->{after:.1f}"
        )
        assert info.status == "removed", info.detail
        assert info.size == 96, info.size
        assert info.inset == 192, info.inset
        assert variant == "2026", variant
        assert scale is not None and abs(scale - 1.0) <= 0.16
        assert abs(after) < 10, (before, after)
        outside = cleaned.copy()
        height, width = original.shape[:2]
        x = width - 192 - 96
        y = height - 192 - 96
        outside[y : y + 96, x : x + 96] = original[y : y + 96, x : x + 96]
        assert max_error(outside, original) == 0


def test_jpeg_still_matches() -> None:
    original = photo(768, 1024, 15)
    marked, _alpha = place(original, 48, 96, 1.0)
    root = Path(tempfile.mkdtemp(prefix="gwr-jpg-"))
    try:
        src = root / "marked.jpg"
        dest = root / "clean.jpg"
        Image.fromarray(marked, "RGB").save(src, quality=90, subsampling=0)
        original_bytes = src.read_bytes()
        info = process_file(engine, src, dest, mode="auto", jpeg_quality=95)
        print(f"  jpeg: {info.status} {info.detail}")
        assert info.status == "removed", info.detail
        assert dest.is_file()
        assert src.read_bytes() == original_bytes
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_png_alpha_is_kept() -> None:
    original = photo(420, 560, 21)
    marked, _alpha = place(original, 48, 32, 1.0)
    mask = np.full(marked.shape[:2], 255, dtype=np.uint8)
    mask[12:28, 18:40] = 0
    image = Image.fromarray(marked, "RGB").convert("RGBA")
    image.putalpha(Image.fromarray(mask, mode="L"))
    root = Path(tempfile.mkdtemp(prefix="gwr-alpha-"))
    try:
        src = root / "marked.png"
        dest = root / "clean.png"
        image.save(src)
        info = process_file(engine, src, dest, mode="auto")
        print(f"  alpha: {info.status} {info.detail}")
        assert info.status == "removed"
        saved = Image.open(dest)
        assert saved.mode == "RGBA"
        saved_alpha = np.asarray(saved.getchannel("A"))
        assert int(saved_alpha[12, 18]) == 0
        assert int(saved_alpha[80, 80]) == 255
        rgb = np.asarray(saved.convert("RGB"))
        assert max_error(rgb, original) <= 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _fake_param(input_name: str = "data", output_name: str = "output") -> str:
    return (
        "7767517\n"
        "2 2\n"
        f"Input                    layer                    0 1 {input_name}\n"
        f"Convolution              conv                     1 1 {input_name} {output_name} 0=3\n"
    )


def test_models() -> None:
    import upscale
    from engine import asset_dir
    from models import DownloadError, add_model, download_model, help_line, inspect_param, list_models, resolve

    assets = asset_dir() / "models"
    info = inspect_param(assets / "realesr-animevideov3-x2.param")
    assert info.input_name == "data" and info.output_name == "output", info
    assert resolve("photo", 2).phrase == "Photo 2×, fitted from 4×"
    assert resolve("photo", 4).phrase == "Photo 4×"
    assert resolve("anime", 2).phrase == "Anime 2×"
    assert resolve("anime", 2).native == 2
    assert resolve("anime", 4).native == 4
    assert resolve("general", 2).phrase == "General 2×, fitted from 4×"
    assert resolve("general", 4).stem == "RealESRGAN_General_x4_v3"
    assert upscale.describe("photo", 2) == "Photo 2×, fitted from 4×"
    try:
        resolve("photo", 3)
    except ValueError as error:
        assert "Photo" in str(error)
    else:
        raise AssertionError("photo has no 3×")

    empty = Path(tempfile.mkdtemp(prefix="gwr-models-"))
    try:
        listed = list_models(empty, assets)
        ids = [entry.id for entry in listed]
        assert ids[:3] == ["photo", "anime", "general"], ids
        by_id = {entry.id: entry for entry in listed}
        assert by_id["photo"].installed and by_id["photo"].badge == "Ready"
        assert by_id["anime"].scales == (2, 4)
        assert not by_id["general"].installed and by_id["general"].badge == "Download"
        assert by_id["siax"].size_label == "64 MB"
        for model_id in ("ultrasharp", "animesharp", "remacri"):
            assert model_id in ids, ids
            assert by_id[model_id].scales == (2, 4)
            assert not by_id[model_id].installed
            assert by_id[model_id].badge == "Download"
        assert resolve("ultrasharp", 4).phrase == "UltraSharp 4×"
        assert resolve("ultrasharp", 2).phrase == "UltraSharp 2×, fitted from 4×"
        assert resolve("animesharp", 4).phrase == "AnimeSharp 4×"
        assert resolve("remacri", 4).phrase == "Remacri 4×"
        assert "Foolhardy" in help_line("remacri", 4)
        try:
            download_model("photo", empty)
        except DownloadError as error:
            assert "included" in str(error).lower()
        else:
            raise AssertionError("bundled models are not downloaded")

        junk = empty / "page-x4.param"
        junk.write_bytes(b"<!DOCTYPE html><html>nope</html>")
        weights = empty / "page-x4.bin"
        weights.write_bytes(b"\x00" * 12000)
        try:
            add_model(junk, weights, empty)
        except ValueError as error:
            assert "ncnn" in str(error).lower()
        else:
            raise AssertionError("html should be rejected")

        unnamed = empty / "plain.param"
        unnamed.write_text(_fake_param(), encoding="utf-8")
        (empty / "plain.bin").write_bytes(b"\x00" * 12000)
        try:
            add_model(unnamed, empty / "plain.bin", empty)
        except ValueError as error:
            assert "x2" in str(error)
        else:
            raise AssertionError("a scale in the file name is required")

        odd = empty / "dots-x3.param"
        odd.write_text(_fake_param("pixels", "result"), encoding="utf-8")
        (empty / "dots-x3.bin").write_bytes(b"\x00" * 12000)
        added = add_model(odd, empty / "dots-x3.bin", empty)
        assert added.scales == (3,) and added.badge == "Custom", added
        resolved = resolve(added.id, 3, [empty])
        assert resolved.input_name == "pixels" and resolved.output_name == "result", resolved
        assert resolved.native == 3
        try:
            resolve(added.id, 4, [empty])
        except ValueError:
            pass
        else:
            raise AssertionError("3× model should not offer 4×")

        wide = empty / "wide-x4.param"
        wide.write_text(_fake_param(), encoding="utf-8")
        (empty / "wide-x4.bin").write_bytes(b"\x00" * 12000)
        fitted = add_model(wide, empty / "wide-x4.bin", empty)
        assert fitted.scales == (2, 4)
        assert "fitted from 4×" in resolve(fitted.id, 2, [empty]).phrase

        pair = empty / "pair"
        pair.mkdir()
        shutil.copy(assets / "realesr-animevideov3-x2.param", pair / "sample-x2.param")
        shutil.copy(assets / "realesr-animevideov3-x2.bin", pair / "sample-x2.bin")
        custom_root = empty / "user"
        copied = add_model(pair / "sample-x2.param", pair / "sample-x2.bin", custom_root)
        assert copied.scales == (2,)
        scaler = upscale.Upscaler(custom_root)
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        tiny[..., 1] = 140
        scaled = scaler.upscale(tiny, copied.id, 2)
        assert scaled.shape == (32, 32, 3), scaled.shape
        assert scaled.dtype == np.uint8
        print(f"  custom model {copied.id} on {scaler.device}")
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def test_upscale() -> None:
    from upscale import Upscaler

    scaler = Upscaler()
    print(f"  device: {scaler.device}")

    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[..., 0] = 180
    red[..., 1] = 40
    red[..., 2] = 40
    doubled = scaler.upscale(red, "anime", 2)
    assert doubled.shape == (64, 64, 3), doubled.shape
    assert doubled.dtype == np.uint8
    center = doubled[20:44, 20:44].astype(np.float32).mean(axis=(0, 1))
    print(f"  anime 2x red center {center}")
    assert center[0] > center[1] + 20 and center[0] > center[2] + 20

    odd = photo(41, 53, 8)
    odd_up = scaler.upscale(odd, "anime", 4)
    assert odd_up.shape == (164, 212, 3), odd_up.shape
    assert odd_up.dtype == np.uint8

    small = photo(16, 20, 9)
    photo_4 = scaler.upscale(small, "photo", 4)
    photo_2 = scaler.upscale(small, "photo", 2)
    assert photo_4.shape == (64, 80, 3), photo_4.shape
    assert photo_2.shape == (32, 40, 3), photo_2.shape
    assert photo_4.dtype == np.uint8 and photo_2.dtype == np.uint8

    smooth = photo(96, 96, 3)
    whole = scaler.upscale(smooth, "anime", 2, tile=96)
    tiled = scaler.upscale(smooth, "anime", 2, tile=32)
    diff = np.abs(whole.astype(np.int16) - tiled.astype(np.int16))
    interior = diff[24:-24, 24:-24]
    print(f"  seam mean {interior.mean():.3f} max {int(interior.max())}")
    assert float(interior.mean()) < 3.0
    assert int(interior.max()) < 30

    root = Path(tempfile.mkdtemp(prefix="gwr-up-"))
    try:
        source = root / "in.png"
        dest = root / "out" / "in.png"
        Image.fromarray(small, "RGB").save(source)
        original_bytes = source.read_bytes()
        info = process_file(engine, source, dest, task="upscale", upscaler=scaler, model="anime", scale=2)
        print(f"  file upscale: {info.status} {info.detail}")
        assert info.status == "upscaled", info.detail
        assert "Anime 2×" in info.detail
        assert source.read_bytes() == original_bytes
        saved = np.asarray(Image.open(dest).convert("RGB"))
        assert saved.shape == (32, 40, 3)
        info = process_file(
            engine,
            source,
            root / "combined-plain.png",
            task="upscale",
            upscaler=scaler,
            model="anime",
            scale=2,
            combine=True,
        )
        assert info.status == "upscaled", info.detail
        assert "Anime 2×" in info.detail
        plain_both = np.asarray(Image.open(root / "combined-plain.png").convert("RGB"))
        assert plain_both.shape == (32, 40, 3)

        original = photo(240, 320, 4)
        marked, _alpha = place(original, 48, 32, 1.0)
        marked_path = root / "marked.png"
        clean_path = root / "clean.png"
        up_path = root / "up.png"
        Image.fromarray(marked, "RGB").save(marked_path)
        marked_bytes = marked_path.read_bytes()
        refused = process_file(engine, marked_path, root / "nope.png", task="both", upscaler=scaler, model="anime", scale=2)
        assert refused.status == "failed", refused.detail
        assert refused.detail == "Choose Clean or Upscale."
        info = process_file(engine, marked_path, clean_path, task="clean")
        assert info.status == "removed", info.detail
        assert marked_path.read_bytes() == marked_bytes
        info = process_file(engine, clean_path, up_path, task="upscale", upscaler=scaler, model="anime", scale=2)
        print(f"  remove then upscale: {info.status} {info.detail}")
        assert info.status == "upscaled", info.detail
        assert "Anime 2×" in info.detail
        cleaned_up = np.asarray(Image.open(up_path).convert("RGB"))
        assert cleaned_up.shape == (480, 640, 3)
        direct = scaler.upscale(original, "anime", 2)
        leftover = np.abs(cleaned_up.astype(np.int16) - direct.astype(np.int16))
        print(f"  cleaned upscale vs original upscale mean {leftover.mean():.3f} max {int(leftover.max())}")
        assert float(leftover.mean()) < 2.0
        watermarked = scaler.upscale(marked, "anime", 2)
        x = (320 - 32 - 48) * 2
        y = (240 - 32 - 48) * 2
        star = np.abs(cleaned_up[y : y + 96, x : x + 96].astype(np.int16) - watermarked[y : y + 96, x : x + 96].astype(np.int16))
        print(f"  star gone mean {star.mean():.3f}")
        assert float(star.mean()) > 5.0

        both_path = root / "both.png"
        info = process_file(
            engine,
            marked_path,
            both_path,
            task="clean",
            upscaler=scaler,
            model="anime",
            scale=2,
            combine=True,
        )
        print(f"  combine: {info.status} {info.detail}")
        assert info.status == "removed", info.detail
        assert "Anime 2×" in info.detail
        assert marked_path.read_bytes() == marked_bytes
        both = np.asarray(Image.open(both_path).convert("RGB"))
        assert both.shape == (480, 640, 3)
        close = np.abs(both.astype(np.int16) - direct.astype(np.int16))
        assert float(close.mean()) < 2.0

        hole = photo(32, 40, 5)
        mask = np.full(hole.shape[:2], 255, dtype=np.uint8)
        mask[5:20, 6:22] = 0
        image = Image.fromarray(hole, "RGB").convert("RGBA")
        image.putalpha(Image.fromarray(mask, mode="L"))
        alpha_path = root / "alpha.png"
        alpha_out = root / "alpha-out.png"
        image.save(alpha_path)
        info = process_file(engine, alpha_path, alpha_out, task="upscale", upscaler=scaler, model="anime", scale=2)
        assert info.status == "upscaled"
        saved_image = Image.open(alpha_out)
        assert saved_image.mode == "RGBA"
        saved_alpha = np.asarray(saved_image.getchannel("A"))
        assert saved_alpha.shape == (64, 80)
        assert int(saved_alpha[2, 2]) > 200
        assert int(saved_alpha[24, 26]) < 40
    finally:
        shutil.rmtree(root, ignore_errors=True)

    empty = Path(tempfile.mkdtemp(prefix="gwr-nomodel-"))
    try:
        missing = Upscaler(empty)
        try:
            missing.upscale(np.zeros((16, 16, 3), dtype=np.uint8), "anime", 2)
        except FileNotFoundError as error:
            assert "missing" in str(error).lower(), error
        else:
            raise AssertionError("a missing model should fail")
    finally:
        shutil.rmtree(empty, ignore_errors=True)

    cpu_only = Upscaler(prefer_gpu=False)
    assert cpu_only.device == "CPU"
    assert cpu_only._vulkan is False


def test_video_paths() -> None:
    import video
    from engine import remove_visible
    from video import _encoder_log, _output_frame, _partial_path, _video_target

    assert _partial_path(Path("clip.mp4")) == Path("clip.partial.mp4")
    assert _partial_path(Path("clip.mkv")).suffix == ".mkv"
    assert _partial_path(Path("clip.mov")).name == "clip.partial.mov"

    log = _encoder_log()
    with log.open("ab") as handle:
        handle.write(b"ok")
    log.unlink()

    assert _video_target(Path("clip.mp4")) == (Path("clip.mp4"), False)
    assert _video_target(Path("clip.MOV"))[1] is False
    assert _video_target(Path("clip.mkv"))[1] is False
    webm, renamed = _video_target(Path("clip.webm"))
    assert webm == Path("clip.mp4") and renamed is True
    m4v, renamed = _video_target(Path("clip.m4v"))
    assert m4v == Path("clip.mp4") and renamed is True
    from video import _encoder_args

    # Encoder choice is behavior, not source text: a frozen build ships no .py
    # files, so assert on the arguments actually produced.
    original_nvenc = video._nvenc_available
    try:
        video._nvenc_available = lambda _tool: False
        fallback = _encoder_args(Path("ffmpeg"), 18, 1920, 1080)
        assert "libx264" in fallback
        assert "veryfast" in fallback
        assert "yuv420p" in fallback
        video._nvenc_available = lambda _tool: True
        accelerated = _encoder_args(Path("ffmpeg"), 18, 1920, 1080)
        assert "h264_nvenc" in accelerated
        assert "yuv420p" in accelerated
    finally:
        video._nvenc_available = original_nvenc
    from video import display_size

    assert display_size(1920, 1080, 0) == (1920, 1080)
    assert display_size(1920, 1080, 90) == (1080, 1920)
    assert display_size(1920, 1080, 180) == (1920, 1080)
    assert display_size(1920, 1080, 270) == (1080, 1920)
    from video import video_dest

    assert video_dest(Path("clip.webm")) == Path("clip.mp4")
    import inspect

    assert "force_x264" in inspect.signature(video._transcode).parameters
    assert "192k" in video.AUDIO_ARGS

    class Grow:
        def upscale(self, rgb, model, scale, cancel=None):
            factor = int(scale)
            return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)

    grown, info = _output_frame(
        photo(20, 30, 2), False, engine, "auto", "gemini", remove_visible, Grow(), "anime", 2, None
    )
    assert info is None
    assert grown.shape == (40, 60, 3)
    grown, info = _output_frame(
        photo(20, 30, 3), True, engine, "auto", "gemini", remove_visible, Grow(), "anime", 2, None
    )
    assert info.status == "skipped", info.detail
    assert grown.shape == (40, 60, 3)
    fitted, info = _output_frame(
        photo(21, 31, 4), False, engine, "auto", "gemini", remove_visible, None, "anime", 1, None
    )
    assert info is None
    assert fitted.shape == (22, 32, 3)

    root = Path(tempfile.mkdtemp(prefix="gwr-video-"))
    saved = (video.find_ffmpeg, video._ffprobe, video._probe, video._one_frame, video.process_video)
    try:
        source = root / "clip.mp4"
        source.write_bytes(b"fake-video")
        dest = root / "out.mp4"
        video.find_ffmpeg = lambda: None
        missing = process_file(engine, source, dest, mark="auto")
        assert missing.status == "failed", missing.detail
        assert "FFmpeg is not installed" in missing.detail
        assert not dest.exists()

        frame = photo(180, 240, 11)
        video.find_ffmpeg = lambda: Path("ffmpeg")
        video._ffprobe = lambda _tool: Path("ffprobe")
        video._probe = lambda _tool, _source: (frame.shape[1], frame.shape[0], 24.0)

        def fake_frame(_tool, _source, _width, _height):
            return frame

        video._one_frame = fake_frame
        copied = process_file(engine, source, dest, mark="gemini")
        assert copied.status == "skipped", copied.detail
        assert dest.read_bytes() == source.read_bytes()

        seen: dict = {}
        progress: list = []

        def fake_process_video(*_args, **kwargs):
            seen["kwargs"] = kwargs
            callback = kwargs.get("on_progress")
            if callback is not None:
                callback(2, 8)
            from engine import CleanInfo

            return CleanInfo("removed", "48px sparkle, 96px from the corner")

        video.process_video = fake_process_video
        grow = Grow()
        info = process_file(
            engine,
            source,
            root / "up.mp4",
            task="upscale",
            upscaler=grow,
            model="anime",
            scale=2,
            mark="auto",
            on_progress=lambda done, frames: progress.append((done, frames)),
        )
        assert info.status == "skipped", info.detail
        assert info.detail == "Videos are not upscaled."
        assert "kwargs" not in seen
        assert progress == []
        assert not (root / "up.mp4").exists()
        info = process_file(
            engine,
            source,
            root / "both.mp4",
            task="clean",
            upscaler=grow,
            model="anime",
            scale=2,
            mark="auto",
            combine=True,
        )
        assert info.status == "removed", info.detail
        assert seen["kwargs"]["remove"] is True
        assert seen["kwargs"]["upscaler"] is None
        info = process_file(
            engine,
            source,
            root / "up-combine.mp4",
            task="upscale",
            upscaler=grow,
            model="anime",
            scale=2,
            mark="auto",
            combine=True,
        )
        assert seen["kwargs"]["remove"] is True
        assert seen["kwargs"]["upscaler"] is None
        missing_scaler = process_file(engine, source, root / "none.mp4", task="upscale", mark="auto")
        assert missing_scaler.status == "skipped", missing_scaler.detail
        assert missing_scaler.detail == "Videos are not upscaled."
    finally:
        video.find_ffmpeg, video._ffprobe, video._probe, video._one_frame, video.process_video = saved
        shutil.rmtree(root, ignore_errors=True)


def test_video_pipeline() -> None:
    """Decode, enlarge, and encode overlap, and the file keeps every frame."""
    import subprocess

    import video
    from video import _transcode

    ffmpeg = Path(__file__).resolve().parent / "portable" / "FrameHeart" / "tools" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        found = video.find_ffmpeg()
        assert found is not None, "FFmpeg is not installed"
        ffmpeg = found
    root = Path(tempfile.mkdtemp(prefix="gwr-pipe-"))
    try:
        source = root / "in.mp4"
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[..., 0] = 200
        proc = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "64x64",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            input=frame.tobytes() * 6,
            capture_output=True,
            timeout=40,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-400:]

        class Grow:
            def upscale(self, rgb, model, scale, cancel=None):
                factor = int(scale)
                return np.repeat(np.repeat(rgb, factor, axis=0), factor, axis=1)

        dest = root / "out.mp4"
        _transcode(
            ffmpeg,
            source,
            dest,
            64,
            64,
            10.0,
            None,
            "auto",
            "gemini",
            None,
            True,
            18,
            None,
            upscaler=Grow(),
            model="anime",
            scale=2,
            remove=False,
            frames=6,
        )
        assert dest.is_file() and dest.stat().st_size > 500
        assert not list(root.glob("*.partial.*"))
        probe = ffmpeg.with_name("ffprobe.exe")
        meta = subprocess.run(
            [
                str(probe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames",
                "-of",
                "csv=p=0",
                str(dest),
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
        text = meta.stdout.decode("utf-8", "replace").strip()
        assert text.startswith("128,128,"), text
        count = int(text.split(",")[-1])
        assert count == 6, text
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_video_mark_after_opening_frame() -> None:
    """A sparkle that the opening frame misses is still removed."""
    import subprocess

    import video
    from video import _frame_at, process_video

    ffmpeg = Path(__file__).resolve().parent / "portable" / "FrameHeart" / "tools" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        found = video.find_ffmpeg()
        assert found is not None, "FFmpeg is not installed"
        ffmpeg = found
    saved_find = video.find_ffmpeg
    video.find_ffmpeg = lambda: ffmpeg
    root = Path(tempfile.mkdtemp(prefix="gwr-vmark-"))
    try:
        plain = photo(240, 320, 6)
        marked, _alpha = place(plain, 48, 96, 1.0)
        source = root / "in.mp4"
        raw = plain.tobytes() + marked.tobytes() * 19
        proc = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "320x240",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-g",
                "250",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            input=raw,
            capture_output=True,
            timeout=40,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-300:]
        dest = root / "out.mp4"
        info = process_video(engine, source, dest, "auto", "auto")
        print(f"  later mark: {info.status} {info.detail}")
        assert info.status == "removed", info.detail
        late = _frame_at(ffmpeg, dest, 320, 240, 1.0, accurate=True)
        assert late is not None
        x = 320 - 96 - 48
        y = 240 - 96 - 48
        cleaned = late[y : y + 48, x : x + 48].astype(np.int16)
        plain_err = np.abs(cleaned - plain[y : y + 48, x : x + 48].astype(np.int16)).mean()
        marked_err = np.abs(cleaned - marked[y : y + 48, x : x + 48].astype(np.int16)).mean()
        print(f"  corner vs plain {plain_err:.2f} vs marked {marked_err:.2f}")
        assert plain_err < marked_err
        assert plain_err < 18

        clean_src = root / "clean.mp4"
        clean_dest = root / "clean-out.mp4"
        proc = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "320x240",
                "-r",
                "10",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clean_src),
            ],
            input=plain.tobytes() * 8,
            capture_output=True,
            timeout=40,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-300:]
        skipped = process_video(engine, clean_src, clean_dest, "auto", "auto")
        assert skipped.status == "skipped", skipped.detail
        assert clean_dest.read_bytes() == clean_src.read_bytes()
    finally:
        video.find_ffmpeg = saved_find
        shutil.rmtree(root, ignore_errors=True)


def _stamp_label(rgb: np.ndarray, x: int, y: int, alpha: float = 0.55, bars: int = 6) -> None:
    """Pale bars. Six of them are about 15% of a 1000px-wide image."""
    for index in range(bars):
        left = x + index * 26
        block = rgb[y : y + 24, left : left + 16].astype(np.float32)
        rgb[y : y + 24, left : left + 16] = np.clip(alpha * 255.0 + (1.0 - alpha) * block, 0, 255).astype(np.uint8)


def test_other_platform_marks() -> None:
    from engine import remove_visible

    height, width = 700, 1000
    yy, xx = np.mgrid[0:height, 0:width]
    base = 48.0 + xx * 0.08 + yy * 0.04
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 2.5, (height, width))
    plain = np.clip(
        np.stack([base + noise, base * 0.92 + noise, base * 0.75 + 8], axis=2),
        0,
        255,
    ).astype(np.uint8)
    untouched, skipped = remove_visible(engine, plain, "auto", "auto")
    assert skipped.status == "skipped", skipped.detail
    assert np.array_equal(untouched, plain)

    marked = plain.copy()
    _stamp_label(marked, 790, 640)
    cleaned, info = remove_visible(engine, marked, "auto", "auto")
    assert info.status == "removed", info.detail
    assert info.detail == "Doubao mark"
    changed = np.any(marked != plain, axis=2)
    error = np.abs(cleaned.astype(np.int16) - plain.astype(np.int16))[changed].mean()
    assert error < 8, error
    near = _grow(changed, 4)
    assert np.array_equal(cleaned[~near], plain[~near])

    solid = plain.copy()
    solid[640:680, 760:960] = 250
    left, solid_info = remove_visible(engine, solid, "auto", "doubao")
    assert solid_info.status == "skipped", solid_info.detail
    assert np.array_equal(left, solid)

    grok = plain.copy()
    _stamp_label(grok, 880, 652, bars=4)
    cleaned, info = remove_visible(engine, grok, "auto", "auto")
    assert info.status == "removed", info.detail
    assert info.detail == "Grok mark", info.detail
    changed = np.any(grok != plain, axis=2)
    error = np.abs(cleaned.astype(np.int16) - plain.astype(np.int16))[changed].mean()
    assert error < 8, error
    near = _grow(changed, 4)
    assert np.array_equal(cleaned[~near], plain[~near])
    forced, forced_info = remove_visible(engine, grok, "auto", "grok")
    assert forced_info.detail == "Grok mark", forced_info.detail
    assert np.array_equal(forced, cleaned)


def test_session_stays_usable_after_download() -> None:
    from desktop import Session, shell_text

    events: list[str] = []
    called: list[str] = []
    settings_dir = Path(tempfile.mkdtemp(prefix="gwr-settings-"))
    previous = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(settings_dir / "settings.json")
    try:
        _session_download_and_settings(events, called, Session, shell_text)
    finally:
        if previous is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = previous
        shutil.rmtree(settings_dir, ignore_errors=True)


def _session_download_and_settings(events, called, Session, shell_text) -> None:
    from models import DownloadCancelled

    session = Session(lambda name, _data: events.append(name))

    def runner(model_id, progress=None, cancel=None):
        called.append(model_id)
        if progress is not None:
            progress(10, 100)
            progress(10, 100)
            progress(100, 100)

    session.set_model("compact")
    session._download_with(runner)
    assert session.wait_download(3), "download thread did not finish"
    assert called == ["compact"]
    assert session.downloading is False
    snap = session.snapshot()
    assert snap["downloading"] is False
    assert snap["running"] is False
    assert snap["stopping"] is False
    session.set_model("photo")
    session.set_task("upscale")
    session.set_scale(2)
    assert session.model_id == "photo"
    assert session.task == "upscale"
    assert session.scale == 2

    def boom(model_id, progress=None, cancel=None):
        raise RuntimeError("disk full")

    session.set_model("general")
    session._download_with(boom)
    assert session.wait_download(3)
    assert session.downloading is False
    session.set_task("clean")
    assert session.task == "clean"
    assert "toast" in events and "state" in events

    def slow(model_id, progress=None, cancel=None):
        assert cancel is not None and cancel.wait(2)
        raise DownloadCancelled()

    session.set_model("nomos")
    session._download_with(slow)
    session.cancel_download()
    assert session.wait_download(3)
    assert session.downloading is False
    session.set_model("anime")
    assert session.model_id == "anime"

    shell = shell_text()
    assert "<title>Frame Heart</title>" in shell
    assert "<strong>Frame Heart</strong>" in shell
    assert "A Heart App by Inigo" in shell
    assert 'id="compare"' in shell
    assert "Before and after" in shell
    assert "Settings" in shell
    assert 'data-task="both"' not in shell
    assert "Upscale after cleaning" in shell
    assert ">Selected<" in shell
    assert "H.264" in shell
    assert 'id="bar"' in shell
    assert "Ready" in shell
    assert "styles.css?v=24" in shell
    assert "app.js?v=24" in shell
    assert 'id="intro"' in shell
    assert "Originals stay where they are" in shell
    assert 'id="app-version"' in shell
    assert "v1.2.1" in shell
    assert 'picturesOnly ? "Folder of images"' in shell
    assert "Drop images, videos, or a folder" in shell
    assert "is-log-collapsed" in shell
    assert "Drop images, videos or a folder" in shell
    assert 'id="log-toggle"' in shell
    assert "Hide" in shell
    assert 'data-task="queue"' in shell
    assert "Clear finished" in shell
    assert "Videos are not upscaled." in shell
    assert "Videos are cleaned, not enlarged." in shell
    for phrase in ("SynthID", "Originals stay in place", "Checking the GPU"):
        assert phrase not in shell, phrase

    folder = Path(tempfile.mkdtemp(prefix="gwr-task-"))
    try:
        session.set_input(str(folder))
        session.set_task("upscale")
        assert session.output.endswith("_upscaled"), session.output
        session.set_combine(True)
        snap = session.snapshot()
        assert snap["action"] == "Upscale"
        assert snap["combine"] is True
        assert snap["showModels"] is True
        assert session.output.endswith("_upscaled")
        session.set_task("clean")
        snap = session.snapshot()
        assert snap["action"] == "Clean and upscale"
        assert session.output.endswith("_upscaled"), session.output
        session.set_combine(False)
        assert session.output.endswith("_cleaned"), session.output
        session.set_task("upscale")
        assert session.output.endswith("_upscaled")
        kept = session.output
        session.set_task("settings")
        assert session.task == "settings"
        assert session.output == kept
        session.set_processor("cpu")
        snap = session.snapshot()
        assert snap["processor"] == "cpu"
        assert snap["device"] == "CPU"
        assert snap["showSettings"] is True
        assert snap["progressLabel"] == "Ready"
        session.set_video_crf(22)
        session.set_keep_audio(False)
        session.set_quality(88)
        saved = json.loads(Path(os.environ["GWR_SETTINGS"]).read_text(encoding="utf-8"))
        assert saved == {
            "processor": "cpu",
            "quality": 88,
            "videoCrf": 22,
            "keepAudio": False,
            "combine": False,
            "seenIntro": False,
        }
        assert session.snapshot()["seenIntro"] is False
        session.accept_intro()
        assert session.snapshot()["seenIntro"] is True
        saved = json.loads(Path(os.environ["GWR_SETTINGS"]).read_text(encoding="utf-8"))
        assert saved["seenIntro"] is True
        session.set_combine(True)
        assert session.output == kept
        session.set_combine(False)
        assert session.output == kept

        import time

        import desktop as desktop_mod

        image = folder / "photo.png"
        image.write_bytes(b"x")
        session._refresh_count()
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("selection") != "photo.png":
            time.sleep(0.05)
            snap = session.snapshot()
        assert snap["selection"] == "photo.png", snap
        assert snap["selectionPath"] == str(image)
        other = folder / "two.png"
        other.write_bytes(b"y")
        session._refresh_count()
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("imageCount") != 2:
            time.sleep(0.05)
            snap = session.snapshot()
        assert snap["selection"] == ""
        assert snap["imageCount"] == 2

        solo = folder / "solo.png"
        second = folder / "second.png"
        solo.write_bytes(b"z")
        second.write_bytes(b"w")
        original_pick = desktop_mod.pick_files

        def pick_solo(_kind, _initial=""):
            return [str(solo)]

        def pick_second(_kind, _initial=""):
            return [str(second)]

        try:
            desktop_mod.pick_files = pick_solo
            fresh = Session(lambda _name, _data: None)
            assert fresh.snapshot()["progressLabel"] == "Ready"
            fresh.add_files()
            assert fresh.input_text == str(solo)
            assert fresh.folder is None
            assert Path(fresh.output).name == "cleaned"
            fresh.set_combine(True)
            assert Path(fresh.output).name == "upscaled"
            deadline = time.perf_counter() + 3
            snap = fresh.snapshot()
            while time.perf_counter() < deadline and snap.get("selection") != "solo.png":
                time.sleep(0.05)
                snap = fresh.snapshot()
            assert snap["selection"] == "solo.png", snap
            assert snap["selectionPath"] == str(solo)
            desktop_mod.pick_files = pick_second
            fresh.add_files()
            assert fresh.input_text == ""
            assert len(fresh.extra) == 2
        finally:
            desktop_mod.pick_files = original_pick
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_folder_dialog() -> None:
    import subprocess

    import pick

    # The dialog script is data bundled with the app, so it exists in a frozen
    # build even when the .py source does not.
    source = pick._SCRIPT
    assert "FolderBrowserDialog" not in source
    assert "IFileOpenDialog" in source
    assert "0x1000" in source
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        Path(kwargs["env"]["GWR_PICK_OUT"]).write_text("C:\\plates\\in\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    original_run = pick.subprocess.run
    pick.subprocess.run = fake_run
    try:
        assert pick.pick_folder("") == "C:\\plates\\in"
        assert calls[-1][:6] == ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File"]
    finally:
        pick.subprocess.run = original_run
    script = pick._SCRIPT
    end = script.index('"@')
    compile_script = script[: end + 2] + "\n[GwrPick.FolderDialog].GetMethods() | Out-Null\n"
    folder = Path(tempfile.mkdtemp(prefix="gwr-pick-"))
    path = folder / "compile.ps1"
    try:
        path.write_text(compile_script, encoding="utf-8")
        env = os.environ.copy()
        env.pop("PSExecutionPolicyPreference", None)
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(path)],
            capture_output=True,
            timeout=60,
            env=env,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")
            raise AssertionError(detail[-800:])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_drop_paths() -> None:
    import time

    from desktop import Session

    folder = Path(tempfile.mkdtemp(prefix="gwr-drop-"))
    previous = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(folder / "settings.json")
    try:
        events = []
        session = Session(lambda name, data: events.append((name, data)))
        image = folder / "one.png"
        clip = folder / "clip.mp4"
        note = folder / "note.txt"
        image.write_bytes(b"png")
        clip.write_bytes(b"mp4")
        note.write_bytes(b"txt")
        nested = folder / "pack"
        nested.mkdir()
        (nested / "two.png").write_bytes(b"png")
        (nested / "skip.txt").write_bytes(b"txt")

        session.drop_paths([str(note)])
        assert any(name == "toast" and "Drop images" in data.get("text", "") for name, data in events)
        assert session.extra == []

        session.drop_paths([str(nested)])
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("selection") != "two.png":
            time.sleep(0.05)
            snap = session.snapshot()
        assert session.folder == nested
        assert snap["selection"] == "two.png", snap
        assert snap["count"] == 1, snap

        session.drop_paths([str(image), str(clip), str(note)])
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("count") != 3:
            time.sleep(0.05)
            snap = session.snapshot()
        assert snap["count"] == 3, snap
        assert snap["imageCount"] == 2 and snap["videoCount"] == 1, snap
        assert {path.name for path in session.extra} == {"one.png", "clip.mp4"}
    finally:
        if previous is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = previous
        shutil.rmtree(folder, ignore_errors=True)


def test_queue_drops_canceled_files() -> None:
    import time

    from desktop import Session

    folder = Path(tempfile.mkdtemp(prefix="gwr-queue-"))
    previous = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(folder / "settings.json")
    try:
        first = folder / "one.png"
        second = folder / "two.png"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        events = []
        session = Session(lambda name, data: events.append((name, data)))
        session.output = str(folder / "out")
        session.extra = [first, second]
        session._refresh_count()
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline and len(session.queue) < 2:
            time.sleep(0.05)
        assert len(session.queue) == 2, session.queue
        ident = str(first.resolve())
        session._preview_id = ident
        session.queue_cancel(ident)
        assert session._preview_id == ""
        assert any(name == "preview" and not data.get("before") for name, data in events)
        assert all(item["id"] != ident for item in session.queue)
        held = len(session.queue)
        session.queue_cancel(ident)
        assert len(session.queue) == held
        session._refresh_count()
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline and any(item["id"] == ident for item in session.queue):
            time.sleep(0.05)
        assert all(item["id"] != ident for item in session.queue)
        assert len(session.queue) == 1
        session.set_task("upscale")
        kept = session.output
        session.set_task("queue")
        assert session.task == "queue"
        assert session.work_task == "upscale"
        assert session.output == kept
        assert session.snapshot()["showQueue"] is True
        session.queue_clear_finished()
        session.queue_remove_all()
        assert session.queue == []
        assert session.extra == []
        session.running = True
        session.work_task = "clean"
        session.task = "clean"
        session.set_task("queue")
        assert session.task == "queue"
        session.set_task("upscale")
        assert session.task == "queue"
        session.running = False
        session.set_task("upscale")
        assert session.task == "upscale"
    finally:
        if previous is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = previous
        shutil.rmtree(folder, ignore_errors=True)


def test_frame_progress() -> None:
    import desktop as desktop_mod
    from desktop import Session

    settings_dir = Path(tempfile.mkdtemp(prefix="gwr-progress-"))
    previous = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(settings_dir / "settings.json")
    events: list = []
    original = desktop_mod.process_file
    try:
        session = Session(lambda name, data: events.append((name, data)))

        def fake_process(*_args, **kwargs):
            assert kwargs.get("combine") is True
            callback = kwargs.get("on_progress")
            for frame in range(1, 6):
                callback(frame, 5)
            from engine import CleanInfo

            return CleanInfo("upscaled", "Anime 2×")

        desktop_mod.process_file = fake_process
        session._send_preview = lambda *_args, **_kwargs: None
        session._engine_instance = lambda: None
        session._scaler_instance = lambda: type("Scaler", (), {"device": "CPU"})()
        session._run(
            [(Path("clip.mp4"), Path("out.mp4"))],
            "clean",
            "auto",
            "anime",
            2,
            95,
            None,
            "auto",
            True,
            18,
            True,
        )
        labels = [item[1]["label"] for item in events if item[0] == "progress"]
        assert "Frame 1 of 5 · clip.mp4" in labels, labels
        assert "Frame 5 of 5 · clip.mp4" in labels, labels
        assert "Frame 3 of 5 · clip.mp4" not in labels, labels
        assert session.progress == 1
        assert session.progress_label == "Done"
    finally:
        desktop_mod.process_file = original
        if previous is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = previous
        shutil.rmtree(settings_dir, ignore_errors=True)


def test_video_job_skips_enlarge() -> None:
    """A video-only batch is cleaned, not enlarged, and does not ask for a model."""
    import time

    import desktop as desktop_mod
    from desktop import Session, _preview_payload

    frame = photo(64, 80, 3)

    class Boom:
        def upscale(self, *_args, **_kwargs):
            raise AssertionError("a video preview must not upscale")

    before, after, caption = _preview_payload(
        frame, "upscale", "auto", "photo", 4, engine, Boom(), "gemini", False, True
    )
    assert caption == "Videos are not upscaled."
    assert before == after
    _before, _after, cleaned = _preview_payload(
        frame, "clean", "auto", "photo", 4, engine, Boom(), "gemini", True, True
    )
    assert "upscaled" not in cleaned.lower()

    folder = Path(tempfile.mkdtemp(prefix="gwr-videojob-"))
    previous = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(folder / "settings.json")
    events: list = []
    original = desktop_mod.process_file
    try:
        batch = folder / "Batch"
        batch.mkdir()
        (batch / "clip.mp4").write_bytes(b"video")
        session = Session(lambda name, data: events.append((name, data)))
        session.set_task("upscale")
        session.set_combine(True)
        session.set_input(str(batch))
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("videoCount") != 1:
            time.sleep(0.05)
            snap = session.snapshot()
        assert snap["videoCount"] == 1 and snap["imageCount"] == 0, snap
        assert snap["action"] == "Upscale"
        assert Path(session.output).name == "Batch_cleaned", session.output
        assert session._reject(False) == "Videos are not upscaled."
        session.set_task("clean")
        assert session._reject(False) is None
        session.set_task("upscale")
        session.set_combine(False)
        assert session._reject(False) == "Videos are not upscaled."
        assert Path(session.output).name == "Batch_cleaned", session.output

        picture = batch / "still.png"
        picture.write_bytes(b"png")
        session._refresh_count()
        deadline = time.perf_counter() + 3
        snap = session.snapshot()
        while time.perf_counter() < deadline and snap.get("imageCount") != 1:
            time.sleep(0.05)
            snap = session.snapshot()
        assert snap["imageCount"] == 1 and snap["videoCount"] == 1, snap
        assert Path(session.output).name == "Batch_upscaled", session.output
        assert session._reject(False) is None

        def fake_process(*_args, **kwargs):
            assert kwargs.get("combine") is True
            from engine import CleanInfo

            return CleanInfo("removed", "48px sparkle, 96px from the corner")

        desktop_mod.process_file = fake_process
        session._send_preview = lambda *_args, **_kwargs: None
        session._engine_instance = lambda: None

        def no_scaler():
            raise AssertionError("a video-only job must not load the upscaler")

        session._scaler_instance = no_scaler
        events.clear()
        session._run(
            [(batch / "clip.mp4", folder / "out.mp4")],
            "upscale",
            "auto",
            "photo",
            4,
            95,
            None,
            "auto",
            True,
            18,
            True,
        )
        logs = [data for name, data in events if name == "log"]
        assert any("Cleaned" in line and "clip.mp4" in line for line in logs), logs
        assert all("Cleaned and upscaled" not in line for line in logs), logs
        assert any(line.startswith("Done.") and "1 cleaned" in line and "upscaled" not in line for line in logs), logs
    finally:
        desktop_mod.process_file = original
        if previous is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = previous
        shutil.rmtree(folder, ignore_errors=True)


def test_done_line_counts_combine() -> None:
    from app import format_summary

    assert format_summary(combined=6) == "6 cleaned and upscaled"
    assert "upscaled" in format_summary(combined=6)
    assert "0 upscaled" not in format_summary(combined=6)
    assert format_summary(removed=2, failed=1) == "2 cleaned, 1 failed"


def test_auto_only_window() -> None:
    """The window has no platform or layout choice left."""
    from desktop import Api, Session, shell_text

    assert not hasattr(Api, "set_mark") and not hasattr(Api, "set_mode")
    page = shell_text()
    for gone in ('id="marks"', 'id="modes"', "Gemini layout", 'id="mark-row"'):
        assert gone not in page, gone
    snap = Session(lambda name, data: None).snapshot()
    assert "marks" not in snap and "mode" not in snap


def test_black_backgrounds_every_layout() -> None:
    for background in (0, 6, 14):
        for size, inset in ((48, 32), (48, 96), (96, 64), (96, 192), (72, 144), (60, 90)):
            for scale in (1.0, 0.7, 0.4, 0.25):
                original = flat(1100, 1300, (background, background, background))
                marked, _alpha = place(original, size, inset, scale)
                out, info = engine.clean_rgb(marked, "auto")
                assert info.status == "removed", (background, size, inset, scale, info.detail)
                assert max_error(out, original) <= 6, (background, size, inset, scale, max_error(out, original))


def test_any_margin_is_found_quickly() -> None:
    original = photo(1200, 1400, 31)
    marked, _alpha = place(original, 66, 133, 0.8)
    _out, info = engine.clean_rgb(marked, "auto")
    assert info.status == "removed", info.detail
    started = time.perf_counter()
    for seed in (41, 42, 43):
        assert engine.find_sparkle(photo(1400, 1400, seed)) is None
    each = (time.perf_counter() - started) / 3
    print(f"  clean picture: {each:.2f}s")
    assert each < 2.5, each


def test_no_box_around_the_star() -> None:
    """Noise in the capture used to darken the whole square."""
    original = flat(900, 900, (30, 34, 38))
    marked, _alpha = place(original, 96, 64, 1.0)
    out, _info = engine.clean_rgb(marked, "auto")
    assert max_error(out, original) <= 2


def test_text_in_a_picture_is_left_alone() -> None:
    from PIL import ImageDraw, ImageFont

    from engine import remove_visible

    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
    for text, spot in (("(c) Jane Doe 2026", (1560, 985)), ("Subtitle line along the bottom", (1300, 960))):
        picture = Image.fromarray(photo(1080, 1920, 5))
        ImageDraw.Draw(picture).text(spot, text, fill=(235, 235, 235), font=font)
        rgb = np.array(picture)
        out, info = remove_visible(engine, rgb, "auto", "auto")
        assert info.status == "skipped", (text, info.detail)
        assert np.array_equal(out, rgb)


def test_dark_label_on_a_light_picture() -> None:
    from PIL import ImageDraw, ImageFont

    from engine import remove_visible

    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
    ramp = np.linspace(190, 230, 768, dtype=np.float32)[None, :].repeat(1152, axis=0)
    original = np.clip(np.dstack((ramp, ramp * 0.9, ramp * 0.78)), 0, 255).astype(np.uint8)
    picture = Image.fromarray(original)
    layer = Image.new("L", picture.size, 0)
    ImageDraw.Draw(layer).text((668, 1118), "O Grok", fill=170, font=font)
    picture = Image.composite(Image.new("RGB", picture.size, (40, 36, 34)), picture, layer)
    marked = np.array(picture)
    out, info = remove_visible(engine, marked, "auto", "auto")
    assert info.status == "removed", info.detail
    assert max_error(out, original) <= 12, max_error(out, original)


def test_deep_grayscale_is_not_white() -> None:
    from engine import rgb_array

    ramp = np.linspace(0, 65535, 256 * 256).reshape(256, 256).astype(np.uint16)
    root = Path(tempfile.mkdtemp(prefix="gwr-deep-"))
    try:
        path = root / "deep.png"
        Image.fromarray(ramp).save(path)
        with Image.open(path) as image:
            rgb = rgb_array(image)
        assert abs(float(rgb.mean()) - 127.5) < 2, rgb.mean()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_same_name_videos_keep_both_outputs() -> None:
    from app import _jobs_for
    from video import video_dest

    root = Path(tempfile.mkdtemp(prefix="gwr-names-"))
    try:
        for name in ("one.mp4", "one.m4v", "one.webm", "two.mov"):
            (root / name).write_bytes(b"x")
        jobs = _jobs_for(root, [], root / "out", False)
        targets = [video_dest(dest).name for _source, dest in jobs]
        assert len(set(targets)) == 4, targets
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_video_never_replaces_its_source() -> None:
    from video import find_ffmpeg

    root = Path(tempfile.mkdtemp(prefix="gwr-own-"))
    try:
        source = root / "clip.mp4"
        source.write_bytes(b"original")
        if find_ffmpeg() is None:
            print("  no ffmpeg")
            return
        info = process_file(engine, source, root / "clip.webm")
        assert info.status == "failed" and "original" in info.detail.lower(), info.detail
        assert source.read_bytes() == b"original"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _wait_idle(session, seconds: float = 40.0) -> None:
    time.sleep(0.2)
    deadline = time.perf_counter() + seconds
    while session.running and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert not session.running
    time.sleep(0.4)


def test_stop_then_start_resumes() -> None:
    import desktop
    from engine import CleanInfo

    root = Path(tempfile.mkdtemp(prefix="gwr-stop-"))
    old = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(root / "settings.json")
    real = desktop.process_file
    try:
        batch = root / "Batch"
        batch.mkdir()
        for index in range(4):
            Image.fromarray(flat(120, 120, (40 + index, 50, 60))).save(batch / f"f{index}.png")
        session = desktop.Session(lambda name, data: None)
        session.set_input(str(batch))
        time.sleep(0.5)
        touched: list[str] = []

        def fake(engine_, source, dest, *args, **kwargs):
            touched.append(source.name)
            if source.name == "f1.png" and len(touched) == 2:
                session.toggle()
                return CleanInfo("failed", "Stopped")
            return real(engine_, source, dest, *args, **kwargs)

        desktop.process_file = fake
        session._start()
        _wait_idle(session)
        assert touched == ["f0.png", "f1.png"], touched
        assert session.count == 4, session.count
        touched.clear()
        session._start()
        _wait_idle(session)
        assert touched == ["f1.png", "f2.png", "f3.png"], touched
    finally:
        desktop.process_file = real
        if old is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = old
        shutil.rmtree(root, ignore_errors=True)


def test_clear_finished_then_same_folder() -> None:
    import desktop

    root = Path(tempfile.mkdtemp(prefix="gwr-clear-"))
    old = os.environ.get("GWR_SETTINGS")
    os.environ["GWR_SETTINGS"] = str(root / "settings.json")
    try:
        batch = root / "Batch"
        batch.mkdir()
        for index in range(3):
            Image.fromarray(flat(120, 120, (40 + index, 50, 60))).save(batch / f"f{index}.png")
        session = desktop.Session(lambda name, data: None)
        session.set_input(str(batch))
        time.sleep(0.5)
        session._start()
        _wait_idle(session)
        session.queue_clear_finished()
        time.sleep(0.5)
        assert session.count == 0
        session.set_input(str(batch))
        time.sleep(0.5)
        assert session.count == 3, session.count
        assert session._reject(False) is None
    finally:
        if old is None:
            os.environ.pop("GWR_SETTINGS", None)
        else:
            os.environ["GWR_SETTINGS"] = old
        shutil.rmtree(root, ignore_errors=True)


def test_video_frames_without_the_mark_stay_clean() -> None:
    from engine import SparkleRepeater

    plain = photo(720, 1280, 8)
    marked, _alpha = place(plain, 48, 96, 1.0)
    match = engine.find_sparkle(marked)
    assert match is not None
    repeater = SparkleRepeater(engine, match)
    repeater.tune([marked] * 8 + [plain] * 4)
    assert repeater.gated
    assert np.array_equal(repeater(plain), plain)
    assert max_error(repeater(marked), plain) <= 26


def test_video_star_is_learned_from_the_clip() -> None:
    """A dark clip whose star is stronger than the capture leaves no ghost."""
    from engine import SparkleRepeater

    frames = []
    clean = []
    prepared = engine._prepare(48, 1.0)
    for level in (6, 10, 14, 18, 8, 12, 16, 20):
        base = flat(720, 1280, (level, level + 2, level + 4))
        x, y = 1280 - 96 - 48, 720 - 96 - 48
        marked = stamp(base, x, y, np.clip(prepared.alpha * 1.05, 0, 1))
        frames.append(marked)
        clean.append(base)
    match = engine.find_sparkle(frames[0])
    assert match is not None
    repeater = SparkleRepeater(engine, match)
    repeater.tune(frames)
    assert not repeater.gated
    for frame, base in zip(frames, clean):
        assert max_error(repeater(frame), base) <= 8


def test_colour_tags_survive_a_video() -> None:
    from video import _colour_args

    args = _colour_args({"matrix": "bt601", "space": "smpte170m", "primaries": "bt709", "transfer": "iec61966-2-1"})
    text = " ".join(args)
    assert "out_color_matrix=bt601" in text
    assert "-colorspace smpte170m" in text and "-color_primaries bt709" in text


def test_clean_is_the_action_and_the_queue_speaks_plainly() -> None:
    from desktop import _ACTIONS, _TITLES, shell_text

    assert _TITLES["clean"] == "Clean" and _ACTIONS["clean"] == "Clean"
    page = shell_text()
    assert ">Clear list<" in page and 'id="queue-stop"' not in page
    assert "Take off list" in page and ">Remove all<" not in page
    assert 'row.append(name, status, action)' in page


def test_show_selects_the_file_in_explorer() -> None:
    import desktop

    calls = []
    real = desktop.subprocess.Popen
    desktop.subprocess.Popen = lambda command, *a, **k: calls.append(command)
    root = Path(tempfile.mkdtemp(prefix="gwr show "))
    try:
        target = root / "a file.png"
        target.write_bytes(b"x")
        session = desktop.Session(lambda name, data: None)
        session.output = str(target)
        session.open_output()
        assert calls == [f'explorer /select,"{target}"'], calls
    finally:
        desktop.subprocess.Popen = real
        shutil.rmtree(root, ignore_errors=True)


def test_old_folder_moves() -> None:
    import home

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "Plates" / "models").mkdir(parents=True)
        (root / "Plates" / "models" / "custom.json").write_text("{}", encoding="utf-8")
        exe = root / "FrameHeart.exe"
        exe.write_bytes(b"x")
        moved = home.portable_home(exe)
        assert moved.name == "FrameHeart"
        assert (moved / "models" / "custom.json").is_file()
        assert not (root / "Plates").exists()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "Plates"
        root.mkdir()
        exe = root / "FrameHeart.exe"
        exe.write_bytes(b"x")
        assert home.portable_home(exe).name == "Plates"


def main() -> None:
    print("summary")
    test_done_line_counts_combine()
    print("math")
    test_math_roundtrip()
    print("layouts")
    test_known_layouts()
    print("dark backgrounds")
    test_dark_backgrounds()
    print("clean images")
    test_leaves_clean_images_alone()
    print("2026 capture")
    test_2026_capture()
    print("samples")
    test_sample_folder()
    print("forced and files")
    test_forced_modes_and_files()
    print("jpeg")
    test_jpeg_still_matches()
    print("alpha")
    test_png_alpha_is_kept()
    print("models")
    test_models()
    print("upscale")
    test_upscale()
    print("video")
    test_video_paths()
    print("video pipeline")
    test_video_pipeline()
    print("video later mark")
    test_video_mark_after_opening_frame()
    print("video job")
    test_video_job_skips_enlarge()
    print("folder dialog")
    test_folder_dialog()
    print("frame progress")
    test_frame_progress()
    print("drop")
    test_drop_paths()
    print("queue")
    test_queue_drops_canceled_files()
    print("other marks")
    test_other_platform_marks()
    print("auto layout")
    test_auto_layout_does_not_force_gemini()
    print("session")
    test_session_stays_usable_after_download()
    print("old folder")
    test_old_folder_moves()
    print("plain wording")
    test_clean_is_the_action_and_the_queue_speaks_plainly()
    print("show in explorer")
    test_show_selects_the_file_in_explorer()
    print("auto only")
    test_auto_only_window()
    print("black backgrounds, every layout")
    test_black_backgrounds_every_layout()
    print("any margin, quickly")
    test_any_margin_is_found_quickly()
    print("no box")
    test_no_box_around_the_star()
    print("text stays")
    test_text_in_a_picture_is_left_alone()
    print("dark label")
    test_dark_label_on_a_light_picture()
    print("deep grayscale")
    test_deep_grayscale_is_not_white()
    print("same-name videos")
    test_same_name_videos_keep_both_outputs()
    print("own source")
    test_a_video_never_replaces_its_source()
    print("stop and resume")
    test_stop_then_start_resumes()
    print("clear finished")
    test_clear_finished_then_same_folder()
    print("gated frames")
    test_video_frames_without_the_mark_stay_clean()
    print("clip star")
    test_video_star_is_learned_from_the_clip()
    print("colour tags")
    test_colour_tags_survive_a_video()
    print("all passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        from upscale import shutdown

        shutdown()
