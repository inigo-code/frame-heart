"""Catalog of ncnn upscalers, plus download and add-your-own.

Photo and Anime ship with the app. The other popular models download into a
models folder next to the app, not into the bundled assets. A custom model is
an ncnn .param and .bin pair whose file name contains x2, x3, or x4.

A 4× network can also be fitted down to 2×. A 2× or 3× network is not run
twice to invent a larger scale.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from engine import asset_dir
from version import __version__

RAW = "https://raw.githubusercontent.com/upscayl/custom-models/main/models"
UPSCAYL = "https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models"
ANIME_SHARP = "https://github.com/TransparentLC/realesrgan-gui/releases/download/additional-models"
REALESRGAN_ZIP = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)
REALESRGAN_ZIP_SIZE = 45474481
_USER_AGENT = f"Plates/{__version__}"
_MIN_BIN = 10_000
_LIGHT_BIN = 8 * 1024 * 1024
_INDEX = "custom.json"
_SCALE_RE = re.compile(r"(?i)(?:^|[^a-z0-9])(?:x([234])|([234])x)(?:[^a-z0-9]|$)")


class DownloadError(Exception):
    pass


class DownloadCancelled(Exception):
    pass


@dataclass(frozen=True)
class ParamInfo:
    input_name: str
    output_name: str


@dataclass(frozen=True)
class _Remote:
    name: str
    size: int
    url: str = ""


@dataclass(frozen=True)
class _Spec:
    id: str
    title: str
    native: int
    scales: tuple[int, ...]
    stems: tuple[tuple[int, str], ...]
    tile: int
    heavy: bool
    bundled: bool
    size_label: str
    lines: dict[int, str]
    files: tuple[_Remote, ...] = ()
    zip_url: str = ""
    zip_size: int = 0


@dataclass(frozen=True)
class ModelEntry:
    id: str
    title: str
    detail: str
    scales: tuple[int, ...]
    native: int
    installed: bool
    kind: str
    heavy: bool
    size_label: str
    badge: str


@dataclass(frozen=True)
class Resolved:
    id: str
    title: str
    scale: int
    native: int
    stem: str
    tile: int
    input_name: str
    output_name: str

    @property
    def phrase(self) -> str:
        if self.id == "photo" and self.scale == 2:
            return "Photo 2×, fitted from 4×"
        if self.id == "photo":
            return "Photo 4×"
        if self.id == "anime" and self.scale == 2:
            return "Anime 2×"
        if self.id == "anime":
            return "Anime 4×"
        if self.scale != self.native:
            return f"{self.title} {self.scale}×, fitted from {self.native}×"
        return f"{self.title} {self.scale}×"


def _remote(name: str, size: int) -> _Remote:
    return _Remote(name, size, f"{RAW}/{name}")


def _hosted(base: str, name: str, size: int) -> _Remote:
    return _Remote(name, size, f"{base}/{name}")


_CATALOG: tuple[_Spec, ...] = (
    _Spec(
        id="photo",
        title="Photo",
        native=4,
        scales=(2, 4),
        stems=((2, "realesrgan-x4plus"), (4, "realesrgan-x4plus")),
        tile=128,
        heavy=True,
        bundled=True,
        size_label="Included",
        lines={
            2: "Photo 2× runs the 4× model, then fits the image to 2×.",
            4: "Photo uses Real-ESRGAN x4plus at 4×.",
        },
    ),
    _Spec(
        id="anime",
        title="Anime",
        native=4,
        scales=(2, 4),
        stems=((2, "realesr-animevideov3-x2"), (4, "realesr-animevideov3-x4")),
        tile=256,
        heavy=False,
        bundled=True,
        size_label="Included",
        lines={
            2: "Anime uses Real-ESRGAN animevideov3 at 2×.",
            4: "Anime uses Real-ESRGAN animevideov3 at 4×.",
        },
    ),
    _Spec(
        id="general",
        title="General",
        native=4,
        scales=(2, 4),
        stems=((2, "RealESRGAN_General_x4_v3"), (4, "RealESRGAN_General_x4_v3")),
        tile=256,
        heavy=False,
        bundled=False,
        size_label="2.3 MB",
        lines={
            2: "General 2× runs the small Real-ESRGAN v3 model, then fits the image to 2×.",
            4: "General is the small Real-ESRGAN v3 model at 4×.",
        },
        files=(
            _remote("RealESRGAN_General_x4_v3.param", 5019),
            _remote("RealESRGAN_General_x4_v3.bin", 2435272),
        ),
    ),
    _Spec(
        id="compact",
        title="Compact",
        native=4,
        scales=(2, 4),
        stems=((2, "4xLSDIRCompactC3"), (4, "4xLSDIRCompactC3")),
        tile=256,
        heavy=False,
        bundled=False,
        size_label="1.2 MB",
        lines={
            2: "Compact 2× runs the 4× model, then fits the image to 2×.",
            4: "Compact is a small photo model at 4×. It is faster than Photo.",
        },
        files=(
            _remote("4xLSDIRCompactC3.param", 2767),
            _remote("4xLSDIRCompactC3.bin", 1247368),
        ),
    ),
    _Spec(
        id="anime-detail",
        title="Anime detail",
        native=4,
        scales=(2, 4),
        stems=((2, "realesrgan-x4plus-anime"), (4, "realesrgan-x4plus-anime")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="43 MB",
        lines={
            2: "Anime detail 2× runs the 4× model, then fits the image to 2×.",
            4: "Anime detail is Real-ESRGAN x4plus-anime at 4×.",
        },
        files=(
            _Remote("realesrgan-x4plus-anime.param", 0),
            _Remote("realesrgan-x4plus-anime.bin", 0),
        ),
        zip_url=REALESRGAN_ZIP,
        zip_size=REALESRGAN_ZIP_SIZE,
    ),
    _Spec(
        id="nomos",
        title="Nomos8k",
        native=4,
        scales=(2, 4),
        stems=((2, "4xNomos8kSC"), (4, "4xNomos8kSC")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="32 MB",
        lines={
            2: "Nomos8k 2× runs the 4× model, then fits the image to 2×.",
            4: "Nomos8k is a sharp photo model at 4×. The download is 32 MB.",
        },
        files=(
            _remote("4xNomos8kSC.param", 108039),
            _remote("4xNomos8kSC.bin", 33424520),
        ),
    ),
    _Spec(
        id="siax",
        title="Siax",
        native=4,
        scales=(2, 4),
        stems=((2, "4x_NMKD-Siax_200k"), (4, "4x_NMKD-Siax_200k")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="64 MB",
        lines={
            2: "Siax 2× runs the 4× model, then fits the image to 2×.",
            4: "Siax is a general photo model at 4×. The download is 64 MB.",
        },
        files=(
            _remote("4x_NMKD-Siax_200k.param", 108037),
            _remote("4x_NMKD-Siax_200k.bin", 66793352),
        ),
    ),
    _Spec(
        id="ultrasharp",
        title="UltraSharp",
        native=4,
        scales=(2, 4),
        stems=((2, "ultrasharp-4x"), (4, "ultrasharp-4x")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="32 MB",
        lines={
            2: "UltraSharp 2× runs the 4× model, then fits the image to 2×.",
            4: "UltraSharp is the 4× UltraSharp photo model.",
        },
        files=(
            _hosted(UPSCAYL, "ultrasharp-4x.param", 116029),
            _hosted(UPSCAYL, "ultrasharp-4x.bin", 33424520),
        ),
    ),
    _Spec(
        id="animesharp",
        title="AnimeSharp",
        native=4,
        scales=(2, 4),
        stems=((2, "4x-AnimeSharp"), (4, "4x-AnimeSharp")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="32 MB",
        lines={
            2: "AnimeSharp 2× runs the 4× model, then fits the image to 2×.",
            4: "AnimeSharp is the 4× AnimeSharp model.",
        },
        files=(
            _hosted(ANIME_SHARP, "4x-AnimeSharp.param", 116029),
            _hosted(ANIME_SHARP, "4x-AnimeSharp.bin", 33424520),
        ),
    ),
    _Spec(
        id="remacri",
        title="Remacri",
        native=4,
        scales=(2, 4),
        stems=((2, "remacri-4x"), (4, "remacri-4x")),
        tile=128,
        heavy=True,
        bundled=False,
        size_label="32 MB",
        lines={
            2: "Remacri 2× runs the 4× model, then fits the image to 2×.",
            4: "Remacri is the 4× Foolhardy Remacri model.",
        },
        files=(
            _hosted(UPSCAYL, "remacri-4x.param", 140295),
            _hosted(UPSCAYL, "remacri-4x.bin", 33424520),
        ),
    ),
)
_BY_ID = {spec.id: spec for spec in _CATALOG}


def user_model_dir() -> Path:
    """Writable model folder beside the exe, or beside this project when run from source."""
    import sys

    if getattr(sys, "frozen", False):
        from home import home_dir

        return home_dir() / "models"
    return Path(__file__).resolve().parent / "models"


def search_dirs(user_dir: Path | None = None, assets: Path | None = None) -> list[Path]:
    if user_dir is None:
        user_dir = user_model_dir()
    if assets is None:
        assets = asset_dir() / "models"
    found: list[Path] = []
    user_path = Path(user_dir)
    asset_path = Path(assets)
    if user_path.is_dir():
        found.append(user_path)
    if asset_path.is_dir() and asset_path not in found:
        found.append(asset_path)
    return found


def scale_from_name(name: str) -> int | None:
    found = None
    for match in _SCALE_RE.finditer(name):
        found = int(match.group(1) or match.group(2))
    return found


def offered_scales(native: int) -> tuple[int, ...]:
    if native == 4:
        return (2, 4)
    if native in (2, 3):
        return (native,)
    raise ValueError("Only 2×, 3×, and 4× models can be added.")


def _junk(head: bytes) -> bool:
    sample = head.lstrip()[:80].lower()
    return (
        sample.startswith(b"version https://git-lfs")
        or sample.startswith(b"<!doctype")
        or sample.startswith(b"<html")
        or sample.startswith(b"<?xml")
    )


def inspect_param(path: Path) -> ParamInfo:
    """Read an ncnn param and return its input and output blob names."""
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read {path.name}. {error}") from error
    if _junk(data[:120]):
        raise ValueError(f"{path.name} is not an ncnn model. It looks like a web page or a Git LFS pointer.")
    text = data.decode("utf-8", "replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "7767517":
        raise ValueError(f"{path.name} is not an ncnn model. It should start with 7767517.")

    input_name = ""
    output_name = ""
    last_output = ""
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            n_in = int(parts[2])
            n_out = int(parts[3])
        except ValueError:
            continue
        names: list[str] = []
        for token in parts[4:]:
            if "=" in token:
                break
            names.append(token)
        if len(names) < n_in + n_out or n_out < 1:
            continue
        outputs = names[n_in : n_in + n_out]
        last_output = outputs[-1]
        if parts[0] == "Input" and not input_name:
            input_name = outputs[0]
        if "output" in outputs:
            output_name = "output"
    if not input_name:
        raise ValueError(f"{path.name} has no Input layer.")
    if not output_name:
        output_name = last_output
    if not output_name:
        raise ValueError(f"{path.name} has no output blob.")
    return ParamInfo(input_name, output_name)


def _check_bin(path: Path) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(200)
    except OSError as error:
        raise ValueError(f"Could not read {path.name}. {error}") from error
    if size < _MIN_BIN or _junk(head):
        raise ValueError(f"{path.name} is not ncnn model weights.")


def _safe_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")
    return (cleaned or "custom-model")[:80]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _read_customs(folder: Path) -> list[dict]:
    path = folder / _INDEX
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, dict)]


def _present(stem: str, dirs: list[Path]) -> bool:
    for folder in dirs:
        if (folder / f"{stem}.param").is_file() and (folder / f"{stem}.bin").is_file():
            return True
    return False


def _blobs(stem: str, dirs: list[Path], default_in: str = "data", default_out: str = "output") -> tuple[str, str]:
    for folder in dirs:
        meta = folder / f"{stem}.meta.json"
        if not meta.is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if isinstance(payload, dict):
            return str(payload.get("input") or default_in), str(payload.get("output") or default_out)
    return default_in, default_out


def _scale_text(scales: tuple[int, ...]) -> str:
    if len(scales) == 1:
        return f"{scales[0]}×"
    return " and ".join(f"{item}×" for item in scales)


def _badge(kind: str, installed: bool) -> str:
    if installed:
        return "Ready"
    if kind == "included":
        return "Missing"
    return "Download"


def _entry_from_spec(spec: _Spec, dirs: list[Path]) -> ModelEntry:
    stems = tuple(dict.fromkeys(stem for _, stem in spec.stems))
    installed = all(_present(stem, dirs) for stem in stems)
    kind = "included" if spec.bundled else "download"
    return ModelEntry(
        id=spec.id,
        title=spec.title,
        detail=f"{spec.size_label} · {_scale_text(spec.scales)}",
        scales=spec.scales,
        native=spec.native,
        installed=installed,
        kind=kind,
        heavy=spec.heavy,
        size_label=spec.size_label,
        badge=_badge(kind, installed),
    )


def _entry_from_custom(item: dict, dirs: list[Path]) -> ModelEntry | None:
    try:
        stem = str(item["stem"])
        native = int(item["native"])
        scales = offered_scales(native)
    except (KeyError, TypeError, ValueError):
        return None
    if not _present(stem, dirs):
        return None
    title = str(item.get("title") or stem)
    kind = "custom"
    heavy = int(item.get("tile") or 128) <= 128
    return ModelEntry(
        id=str(item.get("id") or f"user-{stem}"),
        title=title,
        detail=f"Your file · {_scale_text(scales)}",
        scales=scales,
        native=native,
        installed=True,
        kind=kind,
        heavy=heavy,
        size_label="Custom",
        badge="Custom",
    )


def list_models(user_dir: Path | None = None, assets: Path | None = None) -> list[ModelEntry]:
    if user_dir is None:
        user_dir = user_model_dir()
    if assets is None:
        assets = asset_dir() / "models"
    dirs = search_dirs(user_dir, assets)
    entries = [_entry_from_spec(spec, dirs) for spec in _CATALOG]
    seen = {entry.id for entry in entries}
    for folder in dirs:
        for item in _read_customs(folder):
            entry = _entry_from_custom(item, dirs)
            if entry is not None and entry.id not in seen:
                seen.add(entry.id)
                entries.append(entry)
    return entries


def find_entry(model_id: str, user_dir: Path | None = None, assets: Path | None = None) -> ModelEntry | None:
    key = str(model_id).strip().lower()
    for entry in list_models(user_dir, assets):
        if entry.id.lower() == key or entry.title.lower() == key:
            return entry
    return None


def help_line(model_id: str, scale: int, user_dir: Path | None = None, assets: Path | None = None) -> str:
    try:
        resolved = resolve(model_id, scale, search_dirs(user_dir, assets) if user_dir or assets else None)
    except ValueError:
        return "Choose a model from the list."
    spec = _BY_ID.get(resolved.id)
    if spec is not None and scale in spec.lines:
        return spec.lines[int(scale)]
    if resolved.scale != resolved.native:
        return f"{resolved.title} {resolved.scale}× runs the {resolved.native}× model, then fits the image to {resolved.scale}×."
    return f"{resolved.title} at {resolved.scale}×. Added from your own ncnn files."


def _find_custom(model: str, dirs: list[Path]) -> dict | None:
    key = model.strip().lower()
    for folder in dirs:
        for item in _read_customs(folder):
            ident = str(item.get("id") or "")
            stem = str(item.get("stem") or "")
            title = str(item.get("title") or "")
            if key in {ident.lower(), stem.lower(), title.lower()}:
                return item
    return None


def resolve(model: str, scale: int, dirs: list[Path] | None = None) -> Resolved:
    model_text = str(model).strip()
    scale = int(scale)
    if dirs is None:
        dirs = search_dirs()
    spec = _BY_ID.get(model_text.lower())
    if spec is not None:
        match = [pair for pair in spec.stems if pair[0] == scale]
        if not match:
            offered = _scale_text(spec.scales)
            raise ValueError(f"{spec.title} supports {offered}.")
        stem = match[0][1]
        # Anime 2× is its own network. Every other 2× choice fits a 4× network down.
        native = 2 if spec.id == "anime" and scale == 2 else spec.native
        input_name, output_name = _blobs(stem, dirs)
        return Resolved(spec.id, spec.title, scale, native, stem, spec.tile, input_name, output_name)

    item = _find_custom(model_text, dirs)
    if item is None:
        raise ValueError(
            "Unknown model. Choose photo, anime, general, anime-detail, compact, nomos, siax, ultrasharp, animesharp, remacri, or a model you added."
        )
    try:
        native = int(item["native"])
        stem = str(item["stem"])
        title = str(item.get("title") or stem)
        tile = int(item.get("tile") or 128)
        input_name = str(item.get("input") or "data")
        output_name = str(item.get("output") or "output")
        ident = str(item.get("id") or f"user-{stem}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("That custom model record is damaged.") from error
    scales = offered_scales(native)
    if scale not in scales:
        raise ValueError(f"{title} supports {_scale_text(scales)}.")
    meta_in, meta_out = _blobs(stem, dirs, input_name, output_name)
    return Resolved(ident, title, scale, native, stem, tile, meta_in, meta_out)


def is_ready(model: str, scale: int, dirs: list[Path] | None = None) -> bool:
    if dirs is None:
        dirs = search_dirs()
    resolved = resolve(model, scale, dirs)
    return _present(resolved.stem, dirs)


def _guard_dest(dest: Path) -> None:
    bundled = (asset_dir() / "models").resolve()
    if dest.resolve() == bundled:
        raise DownloadError("Downloads stay in the models folder next to the app.")


def _unique_stem(root: Path, stem: str) -> str:
    candidate = stem
    number = 2
    while (root / f"{candidate}.param").exists() or (root / f"{candidate}.bin").exists():
        candidate = f"{stem}-{number}"
        number += 1
        if number > 40:
            raise ValueError("Too many models already use that name.")
    return candidate


def add_model(param: Path, weights: Path, root: Path | None = None) -> ModelEntry:
    """Copy a local ncnn pair into the writable models folder and remember it."""
    param = Path(param)
    weights = Path(weights)
    if param.suffix.lower() != ".param" or weights.suffix.lower() != ".bin":
        raise ValueError("Choose a .param file and its matching .bin file.")
    if param.resolve() == weights.resolve():
        raise ValueError("The param and the weights have to be two different files.")
    info = inspect_param(param)
    _check_bin(weights)
    native = scale_from_name(param.stem)
    if native is None:
        native = scale_from_name(weights.stem)
    if native is None:
        raise ValueError("Rename the pair so the file name contains x2, x3, or x4.")

    if root is None:
        root = user_model_dir()
    root = Path(root)
    _guard_dest(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(param.stem)
    catalog_stems = {item for spec in _CATALOG for _scale, item in spec.stems}
    if stem in catalog_stems:
        stem = f"custom-{stem}"
    same_folder = param.resolve().parent == root.resolve() and weights.resolve().parent == root.resolve()
    if not (same_folder and param.stem == stem and weights.stem == stem):
        stem = _unique_stem(root, stem)
        shutil.copy2(param, root / f"{stem}.param")
        shutil.copy2(weights, root / f"{stem}.bin")
    tile = 256 if weights.stat().st_size <= _LIGHT_BIN else 128
    record = {
        "id": f"user-{stem}",
        "title": stem,
        "stem": stem,
        "native": native,
        "tile": tile,
        "input": info.input_name,
        "output": info.output_name,
    }
    current = [item for item in _read_customs(root) if str(item.get("id")) != record["id"]]
    current.append(record)
    _write_json(root / _INDEX, {"version": 1, "models": current})
    if info.input_name != "data" or info.output_name != "output":
        _write_json(root / f"{stem}.meta.json", {"input": info.input_name, "output": info.output_name})
    entry = _entry_from_custom(record, [root])
    if entry is None:
        raise ValueError("The model was copied, but it could not be listed.")
    return entry


def _download_stream(
    url: str,
    dest: Path,
    expected: int,
    progress,
    cancel,
    done: int,
    total: int,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    partial = dest.with_suffix(dest.suffix + ".partial")
    got = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                got += len(chunk)
                if progress is not None and total:
                    progress(min(done + got, total), total)
        if expected and got != expected:
            raise DownloadError(f"{dest.name} was {got} bytes, expected {expected}. The download was not saved.")
        partial.replace(dest)
    except DownloadCancelled:
        partial.unlink(missing_ok=True)
        raise
    except DownloadError:
        partial.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {dest.name}. The server returned {error.code}.") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {dest.name}. {error}") from error


def _usable_param(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    if expected and path.stat().st_size != expected:
        return False
    try:
        inspect_param(path)
    except ValueError:
        return False
    return True


def _usable_bin(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    if expected and path.stat().st_size != expected:
        return False
    try:
        _check_bin(path)
    except ValueError:
        return False
    return True


def _finish_pair(dest: Path, stem: str) -> None:
    param = dest / f"{stem}.param"
    weights = dest / f"{stem}.bin"
    try:
        info = inspect_param(param)
        _check_bin(weights)
    except ValueError as error:
        param.unlink(missing_ok=True)
        weights.unlink(missing_ok=True)
        raise DownloadError(str(error)) from error
    if info.input_name != "data" or info.output_name != "output":
        _write_json(dest / f"{stem}.meta.json", {"input": info.input_name, "output": info.output_name})


def _extract_zip_members(zip_path: Path, dest: Path, names: tuple[str, ...], cancel) -> None:
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as error:
        raise DownloadError("The Real-ESRGAN package was not a zip file.") from error
    written: list[Path] = []
    try:
        with archive:
            members = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
            for name in names:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                info = members.get(name)
                if info is None:
                    raise DownloadError(f"The Real-ESRGAN package did not contain {name}.")
                if info.file_size > 80 * 1024 * 1024:
                    raise DownloadError(f"{name} inside the package is unexpectedly large.")
                target = dest / name
                partial = target.with_suffix(target.suffix + ".partial")
                with archive.open(info) as source, partial.open("wb") as handle:
                    shutil.copyfileobj(source, handle, 256 * 1024)
                partial.replace(target)
                written.append(target)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def download_model(model_id: str, dest: Path | None = None, progress=None, cancel=None) -> None:
    """Download one catalog model into dest. Existing valid files are left in place."""
    spec = _BY_ID.get(str(model_id).strip().lower())
    if spec is None:
        raise DownloadError(
            "That model cannot be downloaded. Choose one from the model list."
        )
    if spec.bundled:
        raise DownloadError(f"{spec.title} is already included with the app.")
    if dest is None:
        dest = user_model_dir()
    dest = Path(dest)
    _guard_dest(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if spec.zip_url:
        names = tuple(item.name for item in spec.files)
        if all(_usable_param(dest / name, 0) if name.endswith(".param") else _usable_bin(dest / name, 0) for name in names):
            return
        archive = Path(tempfile.gettempdir()) / "gwr-model-download.zip"
        try:
            _download_stream(spec.zip_url, archive, spec.zip_size, progress, cancel, 0, spec.zip_size)
            _extract_zip_members(archive, dest, names, cancel)
            _finish_pair(dest, Path(names[0]).stem)
        finally:
            archive.unlink(missing_ok=True)
        return

    total = sum(item.size for item in spec.files)
    done = 0
    written: list[Path] = []
    try:
        for item in spec.files:
            target = dest / item.name
            usable = _usable_param(target, item.size) if item.name.endswith(".param") else _usable_bin(target, item.size)
            if usable:
                done += item.size
                if progress is not None and total:
                    progress(done, total)
                continue
            _download_stream(item.url, target, item.size, progress, cancel, done, total)
            written.append(target)
            done += item.size
        stem = Path(spec.files[0].name).stem
        _finish_pair(dest, stem)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
