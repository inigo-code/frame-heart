"""Window for the remover.

Plates shares the Spoolhouse look with Stills and Rushes. Watermark detection is
always automatic, so the window has no platform or layout choice. Model rows
are redrawn in the page. Rebuilding native widgets after a download used to
stop the window accepting input.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import winreg
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app import (
    _display_name,
    _format_line,
    _jobs_for,
    _output_suffix,
    _write_log,
    format_summary,
)
from engine import MEDIA_EXTENSIONS, Engine, collect_images, is_video, process_file, remove_visible, rgb_array
from models import DownloadCancelled, add_model, download_model, find_entry, list_models
from pick import pick_files, pick_folder
from upscale import Upscaler, describe
from version import __version__

VIEW = 180
CROP = 340
_ANTI_FLASH = (
    "--disable-features=CalculateNativeWinOcclusion "
    "--disable-backgrounding-occluded-windows "
    "--disable-renderer-backgrounding "
    "--disable-background-timer-throttling"
)
_TITLES = {"clean": "Remove", "upscale": "Upscale", "settings": "Settings", "queue": "Queue"}
_ACTIONS = {"clean": "Remove", "upscale": "Upscale", "settings": "Settings", "queue": "Queue"}
_WORK = {"clean", "upscale"}
_FINISHED = {"removed", "upscaled", "unchanged", "failed", "stopped"}


def _path_id(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def web_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "web"
    return Path(__file__).resolve().parent / "web"


def shell_text() -> str:
    root = web_dir()
    return "\n".join((root / name).read_text(encoding="utf-8") for name in ("index.html", "styles.css", "app.js"))


def _icon_path() -> Path | None:
    candidate = Path(__file__).resolve().parent / "assets" / "app.ico"
    if getattr(sys, "frozen", False):
        candidate = Path(getattr(sys, "_MEIPASS")) / "assets" / "app.ico"
    return candidate if candidate.is_file() else None


def _open_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened) or opened
        return rgb_array(oriented)


def _fit(rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if image.size != (VIEW, VIEW):
        image = image.resize((VIEW, VIEW), Image.Resampling.LANCZOS)
    return image


def _square_crop(rgb: np.ndarray, where: str, side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    side = max(1, min(side, height, width))
    if where == "center":
        y0 = (height - side) // 2
        x0 = (width - side) // 2
    else:
        y0 = height - side
        x0 = width - side
    return np.ascontiguousarray(rgb[y0 : y0 + side, x0 : x0 + side])


def _focus_pair(before: np.ndarray, after: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Square crop around the pixels that changed, or the bottom-right corner."""
    height, width = before.shape[:2]
    side = max(1, min(side, height, width))
    changed = np.any(before != after, axis=2)
    if not changed.any():
        crop = _square_crop(before, "corner", side)
        return crop, _square_crop(after, "corner", side)
    ys, xs = np.nonzero(changed)
    y0 = min(max(0, int((int(ys.min()) + int(ys.max()) - side) / 2)), height - side)
    x0 = min(max(0, int((int(xs.min()) + int(xs.max()) - side) / 2)), width - side)
    return (
        np.ascontiguousarray(before[y0 : y0 + side, x0 : x0 + side]),
        np.ascontiguousarray(after[y0 : y0 + side, x0 : x0 + side]),
    )


def _jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=84)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _ffmpeg_ready() -> bool:
    from video import ffmpeg_ready

    return ffmpeg_ready()


def _settings_path() -> Path:
    override = os.environ.get("GWR_SETTINGS", "").strip()
    if override:
        return Path(override)
    base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    current = base / "Plates" / "settings.json"
    previous = base / "BulkAIWatermarkRemover" / "settings.json"
    if not current.is_file() and previous.is_file():
        try:
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_bytes(previous.read_bytes())
        except OSError:
            return previous
    return current


def _preview_payload(
    rgb,
    task: str,
    mode: str,
    model: str,
    scale: int,
    engine: Engine,
    scaler,
    mark: str,
    combine: bool = False,
    video: bool = False,
):
    if video and task == "upscale" and not combine:
        fitted = _jpeg(_fit(rgb))
        return fitted, fitted, "Videos are not upscaled."
    if task == "upscale" and not combine:
        crop = _square_crop(rgb, "center", 128)
        up = scaler.upscale(crop, model, scale)
        return _jpeg(_fit(crop)), _jpeg(_fit(up)), describe(model, scale)

    cleaned, info = remove_visible(engine, rgb, mode, mark)
    side = min(CROP, rgb.shape[0], rgb.shape[1])
    before, cleaned_crop = _focus_pair(rgb, cleaned, side)
    if video or not combine:
        return _jpeg(_fit(before)), _jpeg(_fit(cleaned_crop)), info.detail
    after = scaler.upscale(cleaned_crop, model, scale)
    phrase = describe(model, scale)
    return _jpeg(_fit(before)), _jpeg(_fit(after)), f"{info.detail}. {phrase}"


class Bridge:
    """Push events into the page without touching the window before it is listening."""

    def __init__(self) -> None:
        self._window = None
        self._ready = False
        self._pending: list[tuple[str, object]] = []
        self._lock = threading.Lock()

    def emit(self, name: str, data: object) -> None:
        with self._lock:
            window = self._window
            ready = self._ready
            if not ready or window is None:
                self._pending.append((name, data))
                return
        self._send(window, name, data)

    def attach(self, window) -> None:
        with self._lock:
            self._window = window
            self._ready = True
            pending = self._pending
            self._pending = []
        for name, data in pending:
            self._send(window, name, data)

    def _send(self, window, name: str, data: object) -> None:
        payload = json.dumps({"name": name, "data": data}, ensure_ascii=True)
        try:
            window.evaluate_js(f"window.app&&window.app.onEvent({payload})")
        except Exception:
            _write_log(traceback.format_exc())


class Session:
    def __init__(self, emit) -> None:
        self._emit_fn = emit
        self._lock = threading.Lock()
        self.task = "clean"
        self.work_task = "clean"
        self.mode = "auto"
        self.mark = "auto"
        self.model_id = "photo"
        self.scale = 4
        self.folder: Path | None = None
        self.input_text = ""
        self.output = ""
        self.output_auto = True
        self.extra: list[Path] = []
        self.queue: list[dict] = []
        self._removed: set[str] = set()
        self._running_id = ""
        self.recursive = True
        self.quality = 95
        self.processor = "gpu"
        self.video_crf = 18
        self.keep_audio = True
        self.combine = False
        self.seen_intro = False
        self.count = 0
        self.image_count = 0
        self.video_count = 0
        self.selection = ""
        self.selection_path = ""
        self._load_settings()
        self._count_token = 0
        self.running = False
        self.stopping = False
        self.cancel = threading.Event()
        self.file_cancel = threading.Event()
        self.downloading = False
        self.download_pct: int | None = None
        self._download_model_id = ""
        self._download_kind = ""
        self._download_cancel: threading.Event | None = None
        self._download_gen = 0
        self._download_thread: threading.Thread | None = None
        self.device = ""
        self.progress = 0.0
        self.progress_label = "Ready"
        self._engine: Engine | None = None
        self._scaler: Upscaler | None = None
        self._preview_id = ""
        self._preview_token = 0

    def _emit(self, name: str, data: object) -> None:
        try:
            self._emit_fn(name, data)
        except Exception:
            _write_log(traceback.format_exc())

    def _load_settings(self) -> None:
        path = _settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if not isinstance(data, dict):
            return
        if str(data.get("processor", "")).lower() == "cpu":
            self.processor = "cpu"
        try:
            self.quality = min(100, max(80, int(data.get("quality", self.quality))))
        except (TypeError, ValueError):
            pass
        try:
            self.video_crf = min(28, max(16, int(data.get("videoCrf", self.video_crf))))
        except (TypeError, ValueError):
            pass
        if "keepAudio" in data:
            self.keep_audio = bool(data.get("keepAudio"))
        if "combine" in data:
            self.combine = bool(data.get("combine"))
        if "seenIntro" in data:
            self.seen_intro = bool(data.get("seenIntro"))

    def _save_settings(self) -> None:
        with self._lock:
            payload = {
                "processor": self.processor,
                "quality": self.quality,
                "videoCrf": self.video_crf,
                "keepAudio": self.keep_audio,
                "combine": self.combine,
                "seenIntro": self.seen_intro,
            }
        path = _settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            _write_log(traceback.format_exc())

    def snapshot(self) -> dict:
        with self._lock:
            return self._view()

    def _view(self) -> dict:
        models = list_models()
        selected = next((item for item in models if item.id == self.model_id), None)
        if selected is None and models:
            selected = models[0]
            self.model_id = selected.id
        if selected is not None and self.scale not in selected.scales:
            self.scale = int(selected.scales[-1])
        work = self.work_task if self.work_task in _WORK else "clean"
        visible = self.task if self.task in _TITLES else "clean"
        if self.stopping:
            action = "Stopping"
        elif self.running:
            action = "Stop"
        elif self.combine and work == "clean" and visible != "settings":
            action = "Remove and upscale"
        elif visible == "queue":
            action = _ACTIONS[work]
        else:
            action = _ACTIONS[visible]
        ffmpeg_ok = _ffmpeg_ready()
        listed = self.queue[:500]
        return {
            "task": visible,
            "title": _TITLES[visible],
            "action": action,
            "model": self.model_id,
            "scale": self.scale,
            "scales": [int(item) for item in selected.scales] if selected is not None else [],
            "showModels": visible not in {"settings", "queue"} and (work == "upscale" or (self.combine and work == "clean")),
            "combine": self.combine,
            "seenIntro": self.seen_intro,
            "showSettings": visible == "settings",
            "showQueue": visible == "queue",
            "queue": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "status": item["status"],
                    "detail": item.get("detail", ""),
                }
                for item in listed
            ],
            "queueMore": max(0, len(self.queue) - len(listed)),
            "input": self.input_text,
            "output": self.output,
            "recursive": self.recursive,
            "quality": self.quality,
            "processor": self.processor,
            "videoCrf": self.video_crf,
            "keepAudio": self.keep_audio,
            "count": self.count,
            "imageCount": self.image_count,
            "videoCount": self.video_count,
            "selection": self.selection,
            "selectionPath": self.selection_path,
            "downloadKind": self._download_kind if self.downloading else "",
            "running": self.running,
            "stopping": self.stopping,
            "downloading": self.downloading,
            "downloadPct": self.download_pct,
            "device": self.device,
            "progress": self.progress,
            "progressLabel": self.progress_label,
            "ffmpegReady": ffmpeg_ok,
            "ffmpegNote": "FFmpeg is installed." if ffmpeg_ok else "FFmpeg is needed for video.",
            "version": __version__,
            "models": [
                {
                    "id": item.id,
                    "title": item.title,
                    "detail": item.detail,
                    "badge": item.badge,
                    "installed": item.installed,
                    "kind": item.kind,
                }
                for item in models
            ],
        }

    def _jobs(self) -> list[tuple[Path, Path]]:
        text = self.output.strip()
        if not text:
            return []
        try:
            jobs = _jobs_for(self.folder, list(self.extra), Path(text), self.recursive)
        except OSError:
            return []
        if not self._removed:
            return jobs
        return [(source, dest) for source, dest in jobs if _path_id(source) not in self._removed]

    def _forget_locked(self, ident: str) -> bool:
        """Drop one path from the queue so a cancel cannot leave it behind."""
        self._removed.add(ident)
        self.extra = [path for path in self.extra if _path_id(path) != ident]
        self.queue = [item for item in self.queue if item["id"] != ident]
        if self.input_text and _path_id(Path(self.input_text)) == ident:
            self.input_text = ""
            self.folder = None
        clear = bool(ident) and ident == self._preview_id
        if clear:
            self._preview_id = ""
            self._preview_token += 1
        return clear

    def _emit_preview_clear(self) -> None:
        self._emit("preview", {"before": "", "after": "", "caption": ""})

    def _mark_queue(self, source: Path, status: str, detail: str) -> None:
        ident = _path_id(source)
        text = detail[:160]
        with self._lock:
            self._running_id = ident if status == "running" else ""
            found = False
            for item in self.queue:
                if item["id"] == ident:
                    item["status"] = status
                    item["detail"] = text
                    found = True
                    break
            if not found and status == "running":
                self.queue.append({"id": ident, "name": source.name, "status": status, "detail": text})
        self._emit("state", self.snapshot())

    def _refresh_count(self) -> None:
        with self._lock:
            self._count_token += 1
            token = self._count_token
            folder = self.folder
            extra = list(self.extra)
            output_text = self.output.strip()
            recursive = self.recursive
            removed = set(self._removed)
            previous = {item["id"]: dict(item) for item in self.queue}
            running_id = self._running_id
            running = self.running

        def work() -> None:
            jobs: list = []
            try:
                jobs = _jobs_for(folder, extra, Path(output_text or "__gwr_no_output__"), recursive)
                jobs = [(source, dest) for source, dest in jobs if _path_id(source) not in removed]
                videos = sum(1 for source, _dest in jobs if is_video(source))
                images = len(jobs) - videos
            except OSError:
                images = videos = 0
                jobs = []
            rows = []
            for source, _dest in jobs:
                ident = _path_id(source)
                old = previous.get(ident)
                if running and ident == running_id:
                    status = "running"
                    detail = (old or {}).get("detail", "")
                elif old is not None and old.get("status") not in {None, "", "waiting"}:
                    status = old["status"]
                    detail = old.get("detail", "")
                else:
                    status, detail = "waiting", ""
                rows.append({"id": ident, "name": source.name, "status": status, "detail": detail})
            with self._lock:
                if token != self._count_token:
                    return
                live = {item["id"]: item for item in self.queue}
                merged = []
                for row in rows:
                    current = live.get(row["id"])
                    if current is not None and current.get("status") in _FINISHED:
                        row = dict(current)
                    elif self.running and row["id"] == self._running_id:
                        row["status"] = "running"
                    merged.append(row)
                self.queue = merged
                self.image_count = images
                self.video_count = videos
                self.count = images + videos
                if not self.running and self.progress_label == "Done":
                    self.progress = 0
                    self.progress_label = "Ready"
                if self.output_auto and not self.running and self.task in _WORK:
                    self._write_auto_suffix(_output_suffix(self.task, self.combine, videos > 0 and images == 0))
                if len(jobs) == 1:
                    self.selection = jobs[0][0].name
                    self.selection_path = str(jobs[0][0])
                else:
                    self.selection = ""
                    self.selection_path = ""
            self._emit("state", self.snapshot())

        threading.Thread(target=work, daemon=True).start()

    def _write_auto_suffix(self, suffix: str) -> None:
        if self.folder is not None:
            self.output = str(self.folder.parent / f"{self.folder.name}{suffix}")
            return
        if not self.output:
            return
        path = Path(self.output)
        name = path.name
        if name in {"cleaned", "upscaled"}:
            self.output = str(path.with_name("upscaled" if suffix == "_upscaled" else "cleaned"))
            return
        for old in ("_cleaned", "_upscaled"):
            if name.endswith(old):
                self.output = str(path.with_name(name[: -len(old)] + suffix))
                return

    def _apply_auto_output(self) -> None:
        if not self.output_auto or self.task not in _WORK:
            return
        videos_only = self.video_count > 0 and self.image_count == 0
        self._write_auto_suffix(_output_suffix(self.task, self.combine, videos_only))

    def set_task(self, task: str) -> dict:
        task = str(task or "")
        with self._lock:
            if task not in _TITLES or task == self.task:
                pass
            elif self.running:
                # Queue and Settings stay reachable so a file can be cancelled.
                if task not in _WORK:
                    self.task = task
            else:
                self.task = task
                if task in _WORK:
                    self.work_task = task
                    self._apply_auto_output()
        self._refresh_count()
        return self.snapshot()

    def _busy(self) -> bool:
        with self._lock:
            return self.running

    def _wait_toast(self) -> dict:
        self._emit("toast", {"kind": "warn", "text": "Wait for this batch to finish."})
        return self.snapshot()

    def set_model(self, model_id: str) -> dict:
        if self._busy():
            return self._wait_toast()
        model_id = str(model_id or "")
        with self._lock:
            if any(item.id == model_id for item in list_models()):
                self.model_id = model_id
        return self.snapshot()

    def set_scale(self, scale: int) -> dict:
        if self._busy():
            return self._wait_toast()
        try:
            scale = int(scale)
        except (TypeError, ValueError):
            return self.snapshot()
        with self._lock:
            selected = next((item for item in list_models() if item.id == self.model_id), None)
            if selected is not None and scale in selected.scales:
                self.scale = scale
        return self.snapshot()

    def set_quality(self, quality: int) -> dict:
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            quality = 95
        with self._lock:
            self.quality = min(100, max(80, quality))
        self._save_settings()
        return self.snapshot()

    def set_processor(self, value: str) -> dict:
        choice = "cpu" if str(value or "").lower() == "cpu" else "gpu"
        with self._lock:
            changed = self.processor != choice
            self.processor = choice
            scaler = self._scaler
        if changed:
            self._save_settings()
            if scaler is not None:
                scaler.set_prefer_gpu(choice == "gpu")
                name = scaler.device
            else:
                from upscale import device_label

                name = device_label(choice == "gpu")
            with self._lock:
                self.device = name
            self._emit("state", self.snapshot())
        return self.snapshot()

    def set_video_crf(self, value: int) -> dict:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 18
        with self._lock:
            self.video_crf = min(28, max(16, value))
        self._save_settings()
        return self.snapshot()

    def set_keep_audio(self, value: bool) -> dict:
        with self._lock:
            self.keep_audio = bool(value)
        self._save_settings()
        return self.snapshot()

    def set_combine(self, value: bool) -> dict:
        if self._busy():
            return self._wait_toast()
        with self._lock:
            self.combine = bool(value)
            self._apply_auto_output()
        self._save_settings()
        self._refresh_count()
        return self.snapshot()

    def accept_intro(self) -> dict:
        with self._lock:
            self.seen_intro = True
        self._save_settings()
        return self.snapshot()

    def set_recursive(self, value: bool) -> dict:
        if self._busy():
            return self._wait_toast()
        with self._lock:
            self.recursive = bool(value)
        self._refresh_count()
        return self.snapshot()

    def set_input(self, text: str) -> dict:
        if self._busy():
            return self._wait_toast()
        text = str(text or "").strip()
        with self._lock:
            self.input_text = text
            # Choosing a folder asks for everything in it again.
            self._removed.clear()
            if not text:
                self.folder = None
            else:
                folder = Path(text)
                try:
                    is_dir = folder.is_dir()
                except OSError:
                    is_dir = False
                if is_dir:
                    self.folder = folder
                    if self.output_auto or not self.output.strip():
                        self.output_auto = True
                        work = self.task if self.task in _WORK else self.work_task
                        videos_only = self.video_count > 0 and self.image_count == 0
                        self.output = str(folder.parent / f"{folder.name}{_output_suffix(work, self.combine, videos_only)}")
                else:
                    self.folder = None
        self._refresh_count()
        return self.snapshot()

    def set_output(self, text: str) -> dict:
        if self._busy():
            return self._wait_toast()
        with self._lock:
            self.output = str(text or "").strip()
            self.output_auto = False
        self._refresh_count()
        return self.snapshot()

    def browse_input(self) -> dict:
        if self._busy():
            return self._wait_toast()
        try:
            chosen = pick_folder(self.input_text)
        except Exception as error:
            self._emit("toast", {"kind": "critical", "text": str(error)})
            return self.snapshot()
        if not chosen:
            return self.snapshot()
        return self.set_input(chosen)

    def browse_output(self) -> dict:
        if self._busy():
            return self._wait_toast()
        try:
            chosen = pick_folder(self.output)
        except Exception as error:
            self._emit("toast", {"kind": "critical", "text": str(error)})
            return self.snapshot()
        if not chosen:
            return self.snapshot()
        with self._lock:
            self.output = chosen
            self.output_auto = False
        self._refresh_count()
        return self.snapshot()

    def drop_paths(self, paths) -> dict:
        """A folder sets Files. Loose images and videos are added to the batch."""
        if self._busy():
            return self._wait_toast()
        folders: list[Path] = []
        files: list[Path] = []
        for item in paths or []:
            path = Path(str(item))
            try:
                if path.is_dir():
                    folders.append(path)
                elif path.is_file():
                    files.append(path)
            except OSError:
                continue
        media = [path for path in files if path.suffix.lower() in MEDIA_EXTENSIONS]
        if len(folders) == 1 and not media:
            return self.set_input(str(folders[0]))
        with self._lock:
            recursive = self.recursive
            output_text = self.output.strip()
        exclude = Path(output_text) if output_text else None
        if exclude is not None:
            try:
                exists = exclude.exists()
            except OSError:
                exists = False
            if not exists:
                exclude = None
        found = list(media)
        for folder in folders:
            found.extend(collect_images(folder, recursive, exclude))
        if not found:
            self._emit("toast", {"kind": "warn", "text": "Drop images, videos, or a folder of them."})
            return self.snapshot()
        return self._take_files([str(path) for path in found])

    def add_files(self) -> dict:
        if self._busy():
            return self._wait_toast()
        try:
            chosen = pick_files("images", self.input_text or self.output)
        except Exception as error:
            self._emit("toast", {"kind": "critical", "text": str(error)})
            return self.snapshot()
        if not chosen:
            return self.snapshot()
        return self._take_files(chosen)

    def _take_files(self, chosen: list[str]) -> dict:
        with self._lock:
            seen = {path.resolve() for path in self.extra}
            for item in chosen:
                path = Path(item)
                try:
                    key = path.resolve()
                except OSError:
                    key = path
                self._removed.discard(_path_id(path))
                if key not in seen:
                    self.extra.append(path)
                    seen.add(key)
            if not self.output.strip():
                parent = Path(chosen[0]).parent
                work = self.task if self.task in _WORK else self.work_task
                pictures = any(not is_video(Path(item)) for item in chosen)
                enlarge = pictures and (work == "upscale" or self.combine)
                self.output = str(parent / ("upscaled" if enlarge else "cleaned"))
                self.output_auto = True
            if self.folder is None and len(self.extra) == 1 and not self.input_text.strip():
                self.input_text = str(self.extra[0])
            elif len(self.extra) > 1 and self.input_text:
                candidate = Path(self.input_text)
                try:
                    is_file = candidate.is_file()
                except OSError:
                    is_file = False
                if is_file:
                    self.input_text = ""
        self._refresh_count()
        return self.snapshot()

    def _reject(self, preview: bool) -> str | None:
        with self._lock:
            if preview and self.running:
                return "Wait for this batch to finish."
            if self.task == "settings":
                return "Open Remove or Upscale to process files."
            task = self.task if self.task in _WORK else self.work_task
            model_id = self.model_id
            downloading = self.downloading
            download_id = self._download_model_id
            output_text = self.output.strip()
            folder = self.folder
            combine = self.combine and task == "clean"
            jobs = self._jobs()
        videos_only = bool(jobs) and all(is_video(source) for source, _dest in jobs)
        if task == "upscale" and not combine and videos_only:
            return "Videos are not upscaled."
        if (task == "upscale" or combine) and not videos_only:
            entry = next((item for item in list_models() if item.id == model_id), None)
            if entry is None:
                return "Choose a model."
            if downloading and download_id == entry.id and not entry.installed:
                return f"{entry.title} is still downloading."
            if not entry.installed:
                return f"Download {entry.title} first."
        if not output_text:
            return "Choose a save folder."
        if folder is not None:
            try:
                if Path(output_text).resolve() == folder.resolve():
                    return "Choose a different save folder."
            except OSError:
                pass
        if not jobs:
            return "No files found."
        return None

    def download(self) -> dict:
        return self._download_with(None)

    def download_ffmpeg(self) -> dict:
        from video import download_ffmpeg as fetch

        if _ffmpeg_ready():
            return self.snapshot()

        def runner(_model_id, progress=None, cancel=None):
            fetch(progress=progress, cancel=cancel)

        return self._download_with(runner, kind="ffmpeg")

    def _download_with(self, runner, kind: str = "model") -> dict:
        with self._lock:
            busy = self.downloading
            model_id = self.model_id
        if busy:
            return self.snapshot()
        if kind == "ffmpeg":
            title = "FFmpeg"
            model_key = ""
        else:
            models = list_models()
            entry = next((item for item in models if item.id == model_id), None)
            if entry is None or (runner is None and (entry.installed or entry.kind != "download")):
                return self.snapshot()
            title = entry.title
            model_key = entry.id
        cancel = threading.Event()
        with self._lock:
            if self.downloading:
                return self.snapshot()
            self.downloading = True
            self.download_pct = 0
            self._download_kind = kind
            self._download_model_id = model_key
            self._download_cancel = cancel
            self._download_gen += 1
            generation = self._download_gen

        def work() -> None:
            def progress(got: int, total: int) -> None:
                pct = 0 if total <= 0 else min(100, int(got * 100 / total))
                with self._lock:
                    if generation != self._download_gen or not self.downloading or pct == self.download_pct:
                        return
                    self.download_pct = pct
                self._emit("download", {"pct": pct, "kind": kind})

            try:
                target = runner or download_model
                target(model_key, progress=progress, cancel=cancel)
                self._end_download("", title)
            except DownloadCancelled:
                self._end_download("cancelled", title)
            except Exception as error:
                self._end_download(str(error), title)

        thread = threading.Thread(target=work, daemon=True)
        with self._lock:
            self._download_thread = thread
        thread.start()
        self._emit("state", self.snapshot())
        return self.snapshot()

    def _end_download(self, error: str, title: str) -> None:
        with self._lock:
            self.downloading = False
            self.download_pct = None
            self._download_kind = ""
            self._download_cancel = None
            self._download_gen += 1
        if error == "cancelled":
            self._emit("toast", {"kind": "warn", "text": "Download cancelled."})
        elif error:
            self._emit("toast", {"kind": "critical", "text": error[:300]})
        else:
            self._emit("toast", {"kind": "ok", "text": f"{title} is ready."})
        self._emit("state", self.snapshot())

    def wait_download(self, timeout: float) -> bool:
        with self._lock:
            thread = self._download_thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def cancel_download(self) -> dict:
        with self._lock:
            cancel = self._download_cancel
        if cancel is not None:
            cancel.set()
        return self.snapshot()

    def add_model_files(self) -> dict:
        try:
            chosen = pick_files("param", self.input_text)
        except Exception as error:
            self._emit("toast", {"kind": "critical", "text": str(error)})
            return self.snapshot()
        if not chosen:
            return self.snapshot()
        param = Path(chosen[0])
        weights = param.with_suffix(".bin")
        if not weights.is_file():
            try:
                bins = pick_files("bin", str(param.parent))
            except Exception as error:
                self._emit("toast", {"kind": "critical", "text": str(error)})
                return self.snapshot()
            if not bins:
                self._emit("toast", {"kind": "warn", "text": "Choose the .bin that belongs with that param."})
                return self.snapshot()
            weights = Path(bins[0])

        def work() -> None:
            try:
                entry = add_model(param, weights)
            except Exception as error:
                self._emit("toast", {"kind": "critical", "text": str(error)[:300]})
                return
            with self._lock:
                self.model_id = entry.id
            self._emit("toast", {"kind": "ok", "text": f"Added {entry.title}."})
            self._emit("state", self.snapshot())

        threading.Thread(target=work, daemon=True).start()
        return self.snapshot()

    def request_stop(self) -> None:
        with self._lock:
            self.cancel.set()
            self.file_cancel.set()
            download_cancel = self._download_cancel
        if download_cancel is not None:
            download_cancel.set()

    def probe_device(self) -> None:
        from upscale import device_label

        with self._lock:
            prefer = self.processor != "cpu"
        name = "CPU" if not prefer else device_label(True)
        with self._lock:
            if self.processor == "cpu":
                self.device = "CPU"
            elif not self.device:
                self.device = name
        self._emit("state", self.snapshot())

    def _engine_instance(self) -> Engine:
        with self._lock:
            if self._engine is None:
                self._engine = Engine()
            return self._engine

    def _scaler_instance(self) -> Upscaler:
        with self._lock:
            prefer = self.processor != "cpu"
            if self._scaler is not None:
                return self._scaler
        created = Upscaler(prefer_gpu=prefer)
        with self._lock:
            if self._scaler is None:
                self._scaler = created
                self.device = created.device
            return self._scaler

    def preview(self) -> dict:
        reason = self._reject(True)
        if reason:
            self._emit("toast", {"kind": "warn", "text": reason})
            return self.snapshot()
        with self._lock:
            task = self.task if self.task in _WORK else self.work_task
            mode = self.mode
            mark = self.mark
            model = self.model_id
            scale = self.scale
            folder = self.folder
            combine = self.combine and task == "clean"
            source = self._jobs()[0][0]

        def work() -> None:
            try:
                self._send_preview(source, task, mode, model, scale, folder, mark, combine)
            except Exception as error:
                self._emit("toast", {"kind": "critical", "text": str(error)[:300]})

        threading.Thread(target=work, daemon=True).start()
        return self.snapshot()

    def _send_preview(
        self,
        source: Path,
        task: str,
        mode: str,
        model: str,
        scale: int,
        folder: Path | None,
        mark: str,
        combine: bool,
    ) -> None:
        ident = _path_id(source)
        with self._lock:
            if ident in self._removed:
                return
            self._preview_token += 1
            token = self._preview_token
        engine = self._engine_instance()
        video = is_video(source)
        if video:
            from video import preview_frame

            rgb = preview_frame(source, engine, mode, mark)
        else:
            rgb = _open_rgb(source)
        scaler = None if video or not (task == "upscale" or combine) else self._scaler_instance()
        before, after, caption = _preview_payload(
            rgb, task, mode, model, scale, engine, scaler, mark, combine, video
        )
        name = _display_name(source, folder)
        with self._lock:
            if token != self._preview_token or ident in self._removed:
                return
            self._preview_id = ident
        self._emit("preview", {"before": before, "after": after, "caption": f"{name} — {caption}"})
        if scaler is not None:
            self._emit("state", self.snapshot())

    def toggle(self) -> dict:
        with self._lock:
            running = self.running
            if running:
                self.stopping = True
                self.cancel.set()
                self.file_cancel.set()
        if running:
            return self.snapshot()
        self._start()
        return self.snapshot()

    def _start(self) -> None:
        reason = self._reject(False)
        if reason:
            self._emit("toast", {"kind": "warn", "text": reason})
            return
        with self._lock:
            if self.running:
                return
            self.running = True
            self.stopping = False
            self.cancel = threading.Event()
            self.file_cancel = threading.Event()
            self._running_id = ""
            self.progress = 0
            task = self.task if self.task in _WORK else self.work_task
            mode = self.mode
            mark = self.mark
            model = self.model_id
            scale = self.scale
            quality = self.quality
            folder = self.folder
            keep_audio = self.keep_audio
            video_crf = self.video_crf
            combine = self.combine and task == "clean"
            jobs = list(self._jobs())
            # After a Stop, pick up where it left off. A finished batch runs again in full.
            finished = {item["id"] for item in self.queue if item.get("status") in _FINISHED}
            owed = [job for job in jobs if _path_id(job[0]) not in finished]
            if owed:
                jobs = owed
            self.progress_label = jobs[0][0].name if len(jobs) == 1 else f"0 of {len(jobs)}"
        if len(jobs) == 1:
            line = f"Starting {jobs[0][0].name}."
        else:
            line = f"Starting {len(jobs)} files."
        has_image = any(not is_video(source) for source, _dest in jobs)
        has_video = any(is_video(source) for source, _dest in jobs)
        if (task == "upscale" or combine) and has_image:
            try:
                phrase = describe(model, scale)
            except ValueError:
                phrase = ""
            if phrase and combine:
                line = f"{line[:-1]}, remove and {phrase}."
            elif phrase:
                line = f"{line[:-1]}, {phrase}."
        if combine and has_video and has_image:
            line = f"{line[:-1]}. Videos are cleaned, not enlarged."
        self._emit("log", line)
        self._emit("state", self.snapshot())
        threading.Thread(
            target=self._run,
            args=(jobs, task, mode, model, scale, quality, folder, mark, keep_audio, video_crf, combine),
            daemon=True,
        ).start()

    def _run(
        self,
        jobs,
        task: str,
        mode: str,
        model: str,
        scale: int,
        quality: int,
        folder: Path | None,
        mark: str,
        keep_audio: bool,
        video_crf: int,
        combine: bool,
    ) -> None:
        started = time.perf_counter()
        removed = upscaled = unchanged = failed = stopped = combined = 0
        total = len(jobs)
        try:
            engine = self._engine_instance()
            has_image = any(not is_video(source) for source, _dest in jobs)
            scaler = self._scaler_instance() if (task == "upscale" or combine) and has_image else None
            if scaler is not None:
                self._emit("log", f"Upscaling on {scaler.device}.")
                entry = find_entry(model)
                if scaler.device == "CPU" and entry is not None and entry.heavy:
                    self._emit("log", f"{entry.title} on CPU is slow for large images.")
                self._emit("state", self.snapshot())
            try:
                self._send_preview(jobs[0][0], task, mode, model, scale, folder, mark, combine)
            except Exception as error:
                self._emit("log", f"Preview failed. {error}")
            last_emit = 0.0
            for index, (source, dest) in enumerate(jobs, start=1):
                ident = _path_id(source)
                with self._lock:
                    if self.cancel.is_set():
                        stopped += total - index + 1
                        break
                    if ident in self._removed:
                        continue
                    self._running_id = ident
                    self.file_cancel.clear()
                name = _display_name(source, folder)
                self._mark_queue(source, "running", "")
                file_ratio = 0.0 if total == 0 else (index - 1) / total
                file_label = f"{index} of {total} · {name}"
                with self._lock:
                    self.progress = file_ratio
                    self.progress_label = file_label
                self._emit("progress", {"ratio": file_ratio, "label": file_label, "done": index - 1, "total": total})
                last_emit = 0.0

                def on_frame(done: int, frames: int, index: int = index, name: str = name) -> None:
                    nonlocal last_emit
                    if frames > 0:
                        inner = min(1.0, done / float(frames))
                        label = f"Frame {done} of {frames} · {name}"
                    else:
                        inner = 0.0
                        label = f"Frame {done} · {name}"
                    ratio = 0.0 if total == 0 else ((index - 1) + inner) / total
                    now = time.perf_counter()
                    complete = frames > 0 and done >= frames
                    with self._lock:
                        self.progress = ratio
                        self.progress_label = label
                    if done != 1 and not complete and now - last_emit < 0.12:
                        return
                    last_emit = now
                    self._emit("progress", {"ratio": ratio, "label": label, "done": index - 1, "total": total})

                info = process_file(
                    engine,
                    source,
                    dest,
                    mode,
                    quality,
                    task=task,
                    upscaler=None if is_video(source) else scaler,
                    model=model,
                    scale=scale,
                    cancel=self.file_cancel,
                    mark=mark,
                    keep_audio=keep_audio,
                    video_crf=video_crf,
                    combine=combine,
                    on_progress=on_frame,
                )
                if info.detail == "Stopped":
                    self._emit("log", _format_line(name, info))
                    with self._lock:
                        halted = self.cancel.is_set()
                    if halted:
                        # Stop halts the batch. This file is still owed a run.
                        self._mark_queue(source, "waiting", "")
                        stopped += total - index + 1
                        break
                    clear_preview = False
                    with self._lock:
                        clear_preview = self._forget_locked(ident)
                        self._running_id = ""
                    if clear_preview:
                        self._emit_preview_clear()
                    self._refresh_count()
                    stopped += 1
                    continue
                file_combine = combine and not is_video(source)
                if file_combine and info.status == "removed":
                    combined += 1
                    row_status = "removed"
                elif info.status == "removed":
                    removed += 1
                    row_status = "removed"
                elif info.status == "upscaled":
                    upscaled += 1
                    row_status = "upscaled"
                elif info.status == "skipped":
                    unchanged += 1
                    row_status = "unchanged"
                else:
                    failed += 1
                    row_status = "failed"
                self._mark_queue(source, row_status, info.detail)
                self._emit("log", _format_line(name, info, file_combine))
                done_ratio = 0.0 if total == 0 else index / total
                done_label = f"{index} of {total} · {name}"
                with self._lock:
                    self.progress = done_ratio
                    self.progress_label = done_label
                self._emit("progress", {"ratio": done_ratio, "label": done_label, "done": index, "total": total})
        except Exception:
            _write_log(traceback.format_exc())
            self._emit("log", "Stopped on an error.")
            failed += 1
        elapsed = time.perf_counter() - started
        summary = format_summary(removed, upscaled, unchanged, failed, stopped, combined)
        with self._lock:
            self.running = False
            self.stopping = False
            self.progress = 1
            self.progress_label = "Done"
        self._emit("log", f"Done. {summary}. {elapsed:.1f}s")
        with self._lock:
            self._running_id = ""
        self._emit("state", self.snapshot())

    def queue_cancel(self, ident: str) -> dict:
        ident = str(ident or "")
        clear_preview = False
        with self._lock:
            if self.running and ident and ident == self._running_id:
                self.file_cancel.set()
                return self._view()
            if ident:
                clear_preview = self._forget_locked(ident)
        if clear_preview:
            self._emit_preview_clear()
        self._refresh_count()
        return self.snapshot()

    def queue_remove(self, ident: str) -> dict:
        ident = str(ident or "")
        clear_preview = False
        with self._lock:
            if self.running and ident and ident == self._running_id:
                self.file_cancel.set()
            if ident:
                clear_preview = self._forget_locked(ident)
        if clear_preview:
            self._emit_preview_clear()
        self._refresh_count()
        return self.snapshot()

    def queue_clear_finished(self) -> dict:
        clear_preview = False
        with self._lock:
            finished = [item["id"] for item in self.queue if item.get("status") in _FINISHED]
            for ident in finished:
                clear_preview = self._forget_locked(ident) or clear_preview
        if clear_preview:
            self._emit_preview_clear()
        self._refresh_count()
        return self.snapshot()

    def queue_remove_all(self) -> dict:
        with self._lock:
            if self.running:
                busy = True
            else:
                busy = False
                self.extra.clear()
                self.folder = None
                self.input_text = ""
                self._removed.clear()
                self.queue.clear()
                self.selection = ""
                self.selection_path = ""
                self.count = self.image_count = self.video_count = 0
                self._preview_id = ""
                self._preview_token += 1
        if busy:
            self._emit("toast", {"kind": "warn", "text": "Stop the batch first."})
            return self.snapshot()
        self._emit_preview_clear()
        self._refresh_count()
        return self.snapshot()

    def open_output(self) -> dict:
        with self._lock:
            text = self.output.strip()
        if not text:
            self._emit("toast", {"kind": "warn", "text": "Choose a save folder."})
            return self.snapshot()
        path = Path(text)
        try:
            if path.is_file():
                subprocess.Popen(["explorer", f"/select,{path}"])
            else:
                path.mkdir(parents=True, exist_ok=True)
                subprocess.Popen(["explorer", str(path)])
        except OSError as error:
            self._emit("toast", {"kind": "critical", "text": str(error)})
        return self.snapshot()


class Api:
    def __init__(self) -> None:
        self._bridge = Bridge()
        self._session = Session(self._bridge.emit)
        self._probed = False

    def bootstrap(self) -> dict:
        return self._session.snapshot()

    def ui_ready(self) -> dict:
        import webview

        if webview.windows:
            self._bridge.attach(webview.windows[0])
        if not self._probed:
            self._probed = True
            threading.Thread(target=self._session.probe_device, daemon=True).start()
        return self._session.snapshot()

    def set_task(self, task: str) -> dict:
        return self._session.set_task(task)

    def set_model(self, model_id: str) -> dict:
        return self._session.set_model(model_id)

    def set_scale(self, scale: int) -> dict:
        return self._session.set_scale(scale)

    def set_quality(self, quality: int) -> dict:
        return self._session.set_quality(quality)

    def set_processor(self, value: str) -> dict:
        return self._session.set_processor(value)

    def set_video_crf(self, value: int) -> dict:
        return self._session.set_video_crf(value)

    def set_keep_audio(self, value: bool) -> dict:
        return self._session.set_keep_audio(value)

    def set_combine(self, value: bool) -> dict:
        return self._session.set_combine(value)

    def accept_intro(self) -> dict:
        return self._session.accept_intro()

    def download_ffmpeg(self) -> dict:
        return self._session.download_ffmpeg()

    def set_recursive(self, value: bool) -> dict:
        return self._session.set_recursive(value)

    def set_input(self, text: str) -> dict:
        return self._session.set_input(text)

    def set_output(self, text: str) -> dict:
        return self._session.set_output(text)

    def browse_input(self) -> dict:
        return self._session.browse_input()

    def browse_output(self) -> dict:
        return self._session.browse_output()

    def add_files(self) -> dict:
        return self._session.add_files()

    def drop_paths(self, paths) -> dict:
        return self._session.drop_paths(paths)

    def download(self) -> dict:
        return self._session.download()

    def cancel_download(self) -> dict:
        return self._session.cancel_download()

    def add_model(self) -> dict:
        return self._session.add_model_files()

    def preview(self) -> dict:
        return self._session.preview()

    def toggle(self) -> dict:
        return self._session.toggle()

    def open_output(self) -> dict:
        return self._session.open_output()

    def queue_cancel(self, ident: str) -> dict:
        return self._session.queue_cancel(ident)

    def queue_remove(self, ident: str) -> dict:
        return self._session.queue_remove(ident)

    def queue_clear_finished(self) -> dict:
        return self._session.queue_clear_finished()

    def queue_remove_all(self) -> dict:
        return self._session.queue_remove_all()

    def request_stop(self) -> None:
        self._session.request_stop()


def _webview2_installed() -> bool:
    if sys.platform != "win32":
        return True
    guid = r"{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    paths = [
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_CURRENT_USER, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{guid}"),
    ]
    for hive, key_path in paths:
        try:
            key = winreg.OpenKey(hive, key_path)
            winreg.CloseKey(key)
            return True
        except OSError:
            continue
    for folder in (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft" / "EdgeWebView" / "Application",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Microsoft" / "EdgeWebView" / "Application",
    ):
        if folder.is_dir():
            return True
    return False


def _message(text: str, yes_no: bool = False) -> bool:
    if sys.platform != "win32":
        print(text)
        return False
    import ctypes

    flags = 0x10 | (0x4 if yes_no else 0)
    return ctypes.windll.user32.MessageBoxW(None, text, "Plates", flags) == 6


def _bind_drops(window, api) -> None:
    """WebView2 only reports a dropped path when a page listener accepts the drop."""

    def bind() -> None:
        from webview.dom import DOMEventHandler

        def on_drop(event) -> None:
            transfer = event.get("dataTransfer") if isinstance(event, dict) else None
            files = (transfer or {}).get("files") or []
            paths = []
            for item in files:
                if isinstance(item, dict) and item.get("pywebviewFullPath"):
                    paths.append(str(item["pywebviewFullPath"]))
            if paths:
                api.drop_paths(paths)

        try:
            window.dom.body.on("drop", DOMEventHandler(on_drop, prevent_default=True))
        except Exception:
            _write_log(traceback.format_exc())

    threading.Thread(target=bind, daemon=True).start()


def launch() -> None:
    if getattr(sys, "frozen", False):
        import multiprocessing

        multiprocessing.freeze_support()
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Plates")
        except Exception:
            pass
    os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", _ANTI_FLASH)
    if not _webview2_installed():
        if _message("This app needs the Microsoft Edge WebView2 Runtime.\n\nOpen the download page?", yes_no=True):
            try:
                os.startfile("https://go.microsoft.com/fwlink/p/?LinkId=2124703")  # type: ignore[attr-defined]
            except OSError:
                pass
        return
    try:
        import webview
    except Exception as error:
        _write_log(traceback.format_exc())
        _message(f"The window could not open.\n\n{error}")
        return

    api = Api()
    page = (web_dir() / "index.html").resolve().as_uri()
    window = webview.create_window(
        f"Plates {__version__}",
        page,
        js_api=api,
        width=1120,
        height=700,
        min_size=(960, 620),
        background_color="#100f0e",
        text_select=True,
    )
    window.events.closed += api.request_stop
    window.events.loaded += lambda: _bind_drops(window, api)
    storage = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "Plates" / "webview"
    try:
        storage.mkdir(parents=True, exist_ok=True)
        storage_path = str(storage)
    except OSError:
        storage_path = None
    icon = _icon_path()
    try:
        webview.start(gui="edgechromium", icon=str(icon) if icon else None, storage_path=storage_path)
    except Exception as error:
        _write_log(traceback.format_exc())
        _message(f"The window could not open.\n\n{error}")
