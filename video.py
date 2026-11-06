"""Visible corner marks on AI video.

Detection uses the same remover as still images. Once a mark is found, frames
from across the clip are studied so the clip's own star is the one removed.
A frame that does not carry the mark is left as it was. The soundtrack is kept.
A video with no visible watermark is copied unchanged.

Output is H.264. MP4, MOV, and MKV keep that container. WebM and other files
are saved as MP4. Kept audio is AAC. When the frame fits, the NVIDIA encoder
does the work. Otherwise libx264 veryfast does. FFmpeg is downloaded into a
tools folder when it is not already on the computer.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from models import DownloadCancelled, DownloadError, _download_stream

FFMPEG_URL = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
FFMPEG_SIZE = 114768076
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# This NVIDIA H.264 encoder accepts a side from 160 through 4096.
_NVENC_MIN = 160
_NVENC_LIMIT = 4096
_NVENC: bool | None = None
AUDIO_ARGS = ("-c:a", "aac", "-b:a", "192k")
# Frames studied to learn a clip's own star.
STUDY_FRAMES = 40
STUDY_WORKERS = min(8, os.cpu_count() or 4)
# How FFmpeg names a matrix for its scaler, by the name ffprobe reports.
_MATRICES = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "bt601": "bt601",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
    "fcc": "fcc",
    "smpte240m": "smpte240m",
}


def tools_dir() -> Path:
    if getattr(sys, "frozen", False):
        from home import home_dir

        return home_dir() / "tools"
    portable = Path(__file__).resolve().parent / "portable" / "FrameHeart" / "tools"
    if portable.is_dir():
        return portable
    return Path(__file__).resolve().parent / "tools"


def find_ffmpeg() -> Path | None:
    local = tools_dir() / "ffmpeg.exe"
    if local.is_file():
        return local
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def ffmpeg_ready() -> bool:
    tool = find_ffmpeg()
    return tool is not None and _ffprobe(tool) is not None


def download_ffmpeg(progress=None, cancel=None) -> None:
    """Download the pinned FFmpeg essentials build and keep ffmpeg and ffprobe."""
    dest = tools_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if ffmpeg_ready() and (dest / "ffmpeg.exe").is_file():
        return
    archive = Path(tempfile.gettempdir()) / "gwr-ffmpeg.zip"
    try:
        _download_stream(FFMPEG_URL, archive, FFMPEG_SIZE, progress, cancel, 0, FFMPEG_SIZE)
        _extract_tools(archive, dest)
    finally:
        archive.unlink(missing_ok=True)
    if not ffmpeg_ready():
        raise DownloadError("The FFmpeg package did not contain ffmpeg and ffprobe.")


def process_video(
    engine,
    source: Path,
    dest: Path,
    mode: str,
    mark: str,
    cancel=None,
    keep_audio: bool = True,
    crf: int = 18,
    upscaler=None,
    model: str = "photo",
    scale: int = 4,
    on_progress=None,
    remove: bool = True,
):
    """Remove a visible mark, upscale each frame, or both.

    Remove checks the first frame and later frames. One hit is enough: that
    corner is cleaned on every frame. A clip with no hit is copied.
    Upscale re-encodes every frame at the chosen scale.
    """
    from engine import CleanInfo, Stopped, remove_visible

    tool = find_ffmpeg()
    if tool is None or _ffprobe(tool) is None:
        return CleanInfo("failed", "FFmpeg is not installed. Download it in Settings.")
    crf = int(min(28, max(16, crf)))
    try:
        # WebM and M4V are written as MP4, so the real target is what must differ.
        if source.resolve() == dest.resolve() or source.resolve() == video_dest(dest).resolve():
            return CleanInfo("failed", "Refusing to overwrite the original")
        width, height, fps = _probe(tool, source)
        frames = _frame_count(tool, source, fps)
        found_info = None
        found_fn = None
        if remove:
            found_info, found_fn = _find_video_mark(
                tool, source, width, height, fps, frames, engine, mode, mark, cancel
            )
            if found_info is not None and found_info.status == "failed":
                return found_info
            if cancel is not None and cancel.is_set():
                return CleanInfo("failed", "Stopped")
        if upscaler is None:
            if found_fn is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                return found_info or CleanInfo("skipped", "No visible mark found")
            target, renamed = _video_target(dest)
            _transcode(
                tool, source, target, width, height, fps, engine, mode, mark, cancel,
                keep_audio, crf, remove_visible, upscaler=None, model=model, scale=scale,
                on_progress=on_progress, remove=True, frames=frames,
                preset_fn=found_fn, preset_info=found_info,
            )
            detail = found_info.detail if not renamed else f"{found_info.detail}. Saved as MP4"
            return CleanInfo("removed", detail)

        if cancel is not None and cancel.is_set():
            return CleanInfo("failed", "Stopped")
        target, renamed = _video_target(dest)
        first_info = _transcode(
            tool, source, target, width, height, fps, engine, mode, mark, cancel,
            keep_audio, crf, remove_visible, upscaler=upscaler, model=model, scale=scale,
            on_progress=on_progress, remove=found_fn is not None, frames=frames,
            preset_fn=found_fn, preset_info=found_info,
        )
        phrase = upscaler.describe(model, scale)
        if first_info is not None and first_info.status == "removed":
            status, detail = "removed", f"{first_info.detail}. {phrase}"
        elif found_info is not None and found_fn is None:
            status, detail = "upscaled", f"{found_info.detail}. {phrase}"
        elif first_info is not None and remove:
            status, detail = "upscaled", f"{first_info.detail}. {phrase}"
        else:
            status, detail = "upscaled", phrase
        if renamed:
            detail = f"{detail}. Saved as MP4"
        return CleanInfo(status, detail)
    except Stopped:
        return CleanInfo("failed", "Stopped")
    except (OSError, RuntimeError, ValueError) as error:
        return CleanInfo("failed", str(error)[:300])


def first_frame(source: Path) -> np.ndarray:
    tool = find_ffmpeg()
    if tool is None or _ffprobe(tool) is None:
        raise RuntimeError("FFmpeg is not installed. Download it in Settings.")
    width, height, _fps = _probe(tool, source)
    frame = _one_frame(tool, source, width, height)
    if frame is None:
        raise RuntimeError("Could not read a frame from that video.")
    return frame


def preview_frame(source: Path, engine=None, mode: str = "auto", mark: str = "auto") -> np.ndarray:
    """A frame that shows the mark when one is on the clip, else the opening frame."""
    tool = find_ffmpeg()
    if tool is None or _ffprobe(tool) is None:
        raise RuntimeError("FFmpeg is not installed. Download it in Settings.")
    width, height, fps = _probe(tool, source)
    frames = _frame_count(tool, source, fps)
    picked = None
    opening = None
    for moment in _sample_times(frames, fps):
        frame = _frame_at(tool, source, width, height, moment)
        if frame is None:
            continue
        frame = _fit_even(frame)
        if opening is not None and moment > 0 and np.array_equal(frame, opening):
            frame = _frame_at(tool, source, width, height, moment, accurate=True)
            if frame is None:
                continue
            frame = _fit_even(frame)
        elif opening is None:
            opening = frame
        if picked is None:
            picked = frame
        if engine is None:
            continue
        from engine import prepare_frame_removal

        _cleaned, info, fn = prepare_frame_removal(engine, frame, mode, mark)
        if info.status == "removed" and fn is not None:
            return frame
    if picked is None:
        raise RuntimeError("Could not read a frame from that video.")
    return picked


def _video_target(dest: Path) -> tuple[Path, bool]:
    """Keep MP4, MOV, and MKV. Every other container is written as MP4."""
    target = dest if dest.suffix.lower() in {".mp4", ".mov", ".mkv"} else dest.with_suffix(".mp4")
    return target, target != dest


def video_dest(dest: Path) -> Path:
    """Path the cleaned video is actually written to."""
    return _video_target(dest)[0]


def _ffprobe(ffmpeg: Path) -> Path | None:
    name = "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
    sibling = ffmpeg.with_name(name)
    if sibling.is_file():
        return sibling
    found = shutil.which("ffprobe")
    return Path(found) if found else None


def _extract_tools(archive: Path, dest: Path) -> None:
    try:
        bundle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as error:
        raise DownloadError("The FFmpeg package was not a zip file.") from error
    wanted = {"ffmpeg.exe": dest / "ffmpeg.exe", "ffprobe.exe": dest / "ffprobe.exe"}
    found: dict[str, object] = {}
    with bundle:
        for info in bundle.infolist():
            leaf = Path(info.filename).name.lower()
            if leaf in wanted and info.file_size > 1_000_000:
                found[leaf] = info
        if set(found) != set(wanted):
            raise DownloadError("The FFmpeg package did not contain ffmpeg and ffprobe.")
        for leaf, info in found.items():
            target = wanted[leaf]
            partial = target.with_suffix(target.suffix + ".partial")
            with bundle.open(info) as source, partial.open("wb") as handle:
                shutil.copyfileobj(source, handle, 256 * 1024)
            partial.replace(target)


def display_size(width: int, height: int, rotation: float) -> tuple[int, int]:
    """Size after FFmpeg's default autorotate. 90 and 270 swap the axes."""
    turn = abs(int(round(float(rotation)))) % 360
    if turn in {90, 270}:
        return height, width
    return width, height


def _rotation(fields: dict[str, str]) -> float:
    for key in ("rotation", "rotate", "TAG:rotate"):
        text = fields.get(key, "").strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return 0.0


def _probe(ffmpeg: Path, source: Path) -> tuple[int, int, float]:
    probe = _ffprobe(ffmpeg)
    if probe is None:
        raise RuntimeError("FFmpeg is not installed. Download it in Settings.")
    proc = subprocess.run(
        [
            str(probe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate",
            "-of",
            "default=nw=1",
            str(source),
        ],
        capture_output=True,
        timeout=40,
        creationflags=_NO_WINDOW,
        check=False,
    )
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0 or not text.strip():
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(detail[-1][:240] if detail else "Could not read that video.")
    fields = _probe_fields(text)
    try:
        width, height = int(fields["width"]), int(fields["height"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Could not read that video.") from error
    width, height = display_size(width, height, _rotation(fields))
    rate = fields.get("r_frame_rate", "30/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den) if float(den) else 30.0
    else:
        fps = float(rate or 30)
    if width < 16 or height < 16:
        raise RuntimeError("That video is too small.")
    if fps <= 0 or fps > 240:
        fps = 30.0
    return width, height, fps


def _colour(ffmpeg: Path, source: Path, height: int) -> dict[str, str]:
    """The source's colour tags, so the output plays back the same."""
    probe = _ffprobe(ffmpeg)
    fields: dict[str, str] = {}
    if probe is not None:
        try:
            proc = subprocess.run(
                [
                    str(probe), "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=color_space,color_primaries,color_transfer,color_range",
                    "-of", "default=nw=1", str(source),
                ],
                capture_output=True,
                timeout=40,
                creationflags=_NO_WINDOW,
                check=False,
            )
            fields = _probe_fields(proc.stdout.decode("utf-8", "replace"))
        except (OSError, subprocess.TimeoutExpired):
            fields = {}

    def known(key: str) -> str:
        value = fields.get(key, "").lower()
        return "" if value in {"", "unknown", "unspecified", "reserved"} else value

    # An untagged clip is decoded the way a player would guess: by its height.
    tagged = known("color_space")
    guess = "bt709" if height >= 720 else "smpte170m"
    space = tagged if tagged in _MATRICES else guess
    return {
        "matrix": _MATRICES[space],
        "tagged": "yes" if tagged in _MATRICES else "",
        "space": space,
        "primaries": known("color_primaries") or ("bt709" if height >= 720 else "smpte170m"),
        "transfer": known("color_transfer") or ("bt709" if height >= 720 else "smpte170m"),
    }


def _frame_count(ffmpeg: Path, source: Path, fps: float) -> int:
    probe = _ffprobe(ffmpeg)
    if probe is None:
        return 0
    try:
        proc = subprocess.run(
            [
                str(probe),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,duration",
                "-of",
                "default=nw=1",
                str(source),
            ],
            capture_output=True,
            timeout=40,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except OSError:
        return 0
    if proc.returncode != 0:
        return 0
    fields = _probe_fields(proc.stdout.decode("utf-8", "replace"))
    frames = int(fields["nb_frames"]) if fields.get("nb_frames", "").isdigit() else 0
    duration = 0.0
    try:
        duration = float(fields.get("duration") or 0)
    except ValueError:
        duration = 0.0
    if frames <= 0 and duration > 0 and fps > 0:
        frames = int(round(duration * fps))
    return frames


def _probe_fields(text: str) -> dict[str, str]:
    """ffprobe default lines are key=value. CSV order is not the order we ask for."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _fit_even(rgb: np.ndarray) -> np.ndarray:
    """Pad top/left so the bottom-right watermark corner keeps its inset."""
    height, width = rgb.shape[:2]
    pad_y = height % 2
    pad_x = width % 2
    even_h = height + pad_y
    even_w = width + pad_x
    if even_h < 2 or even_w < 2:
        raise RuntimeError("That frame is too small to encode.")
    if pad_y or pad_x:
        rgb = np.pad(rgb, ((pad_y, 0), (pad_x, 0), (0, 0)), mode="edge")
    return np.ascontiguousarray(rgb)


def _output_frame(frame, remove, engine, mode, mark, remove_visible, upscaler, model, scale, cancel):
    """Remove a watermark, upscale, or both. The result has even width and height."""
    from engine import Stopped

    rgb = np.ascontiguousarray(frame)
    info = None
    if remove:
        cleaned, info = remove_visible(engine, rgb, mode, mark)
        if info.status == "failed" and info.detail == "Stopped":
            raise Stopped()
        if info.status == "removed":
            rgb = np.ascontiguousarray(cleaned)
    if upscaler is not None:
        rgb = upscaler.upscale(rgb, model, scale, cancel=cancel)
    return _fit_even(rgb), info


def _one_frame(ffmpeg: Path, source: Path, width: int, height: int) -> np.ndarray | None:
    proc = subprocess.Popen(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    assert proc.stdout is not None
    raw = _read_exact(proc.stdout, width * height * 3)
    err = proc.stderr.read() if proc.stderr is not None else b""
    code = proc.wait(timeout=60)
    if raw is None or code != 0:
        return None
    del err
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _frame_at(
    ffmpeg: Path,
    source: Path,
    width: int,
    height: int,
    seconds: float,
    accurate: bool = False,
) -> np.ndarray | None:
    """One RGB frame. A fast seek can land on an earlier keyframe."""
    if seconds <= 0:
        return _one_frame(ffmpeg, source, width, height)
    command = [str(ffmpeg), "-v", "error"]
    if not accurate:
        command += ["-ss", f"{seconds:.3f}"]
    command += ["-i", str(source)]
    if accurate:
        command += ["-ss", f"{seconds:.3f}"]
    command += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    assert proc.stdout is not None
    raw = _read_exact(proc.stdout, width * height * 3)
    err = proc.stderr.read() if proc.stderr is not None else b""
    code = proc.wait(timeout=120)
    if raw is None or code != 0:
        return None
    del err
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _sample_times(frames: int, fps: float) -> list[float]:
    """Times to search. The opening frame is not always the clear one."""
    if frames <= 1 or fps <= 0:
        return [0.0]
    duration = frames / fps
    step = 0.5 / fps
    times = [0.0]
    for frac in (0.08, 0.2, 0.35, 0.5, 0.7, 0.85):
        moment = min(duration - step, duration * frac)
        if moment > step and all(abs(moment - earlier) > step for earlier in times):
            times.append(moment)
    return times


def _study_times(frames: int, fps: float) -> list[float]:
    """Evenly spread moments, so dark and bright scenes both count."""
    if frames <= 1 or fps <= 0:
        return [0.0]
    duration = frames / fps
    count = min(STUDY_FRAMES, max(3, frames))
    return [duration * (index + 0.5) / count for index in range(count)]


def _find_video_mark(ffmpeg, source, width, height, fps, frames, engine, mode, mark, cancel):
    """Return the visible mark among sample frames, and how to repeat it."""
    from engine import CleanInfo, LabelRepeater, prepare_frame_removal

    last = None
    opening = None
    candidates: list[tuple[CleanInfo, object, float]] = []
    for moment in _sample_times(frames, fps):
        if cancel is not None and cancel.is_set():
            return CleanInfo("failed", "Stopped"), None
        frame = _frame_at(ffmpeg, source, width, height, moment)
        if frame is None:
            continue
        frame = _fit_even(frame)
        if opening is not None and moment > 0 and np.array_equal(frame, opening):
            frame = _frame_at(ffmpeg, source, width, height, moment, accurate=True)
            if frame is None:
                continue
            frame = _fit_even(frame)
        elif opening is None:
            opening = frame
        _cleaned, info, fn = prepare_frame_removal(engine, frame, mode, mark)
        last = info
        if info.status == "failed":
            return info, None
        if info.status == "removed" and fn is not None:
            score = 1.0 if info.survival is None else float(info.survival)
            candidates.append((info, fn, score))
            if len(candidates) >= 3:
                break
    if last is None:
        return CleanInfo("failed", "Could not read a frame from that video."), None
    if not candidates:
        return last, None
    candidates.sort(key=lambda item: item[2])
    info, fn, _score = candidates[0]
    def read(moment: float):
        if cancel is not None and cancel.is_set():
            return None
        return _frame_at(ffmpeg, source, width, height, moment, accurate=moment < 2.0)

    # Each read is its own short FFmpeg run, so they can overlap.
    with ThreadPoolExecutor(max_workers=STUDY_WORKERS) as pool:
        study = [_fit_even(frame) for frame in pool.map(read, _study_times(frames, fps)) if frame is not None]
    if cancel is not None and cancel.is_set():
        return CleanInfo("failed", "Stopped"), None
    if isinstance(fn, LabelRepeater) and mark == "auto" and len(study) >= 3 and not fn.present_on(study):
        # Text that moves or changes is part of the picture, not a mark.
        return CleanInfo("skipped", "No visible mark found"), None
    fn.tune(study)
    return info, fn


def _nvenc_available(ffmpeg: Path) -> bool:
    """True when this FFmpeg can encode H.264 on the NVIDIA chip."""
    global _NVENC
    if _NVENC is not None:
        return _NVENC
    # NVENC rejects a frame smaller than about 145 pixels on a side.
    frame = bytes(256 * 256 * 3)
    try:
        proc = subprocess.run(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "256x256",
                "-r",
                "1",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "18",
                "-b:v",
                "0",
                "-f",
                "null",
                "-",
            ],
            input=frame,
            capture_output=True,
            timeout=25,
            creationflags=_NO_WINDOW,
        )
        _NVENC = proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        _NVENC = False
    return _NVENC


def _colour_args(colour: dict[str, str] | None) -> list[str]:
    """Convert from RGB with the source's matrix and tag the file to match."""
    if not colour:
        return []
    return [
        "-vf",
        f"scale=out_color_matrix={colour['matrix']}:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
        "-colorspace", colour["space"],
        "-color_primaries", colour["primaries"],
        "-color_trc", colour["transfer"],
        "-color_range", "tv",
    ]


def _encoder_args(ffmpeg: Path, crf: int, width: int, height: int) -> list[str]:
    """H.264 arguments. The NVIDIA encoder is used when the frame fits."""
    quality = str(int(crf))
    width = int(width)
    height = int(height)
    fits = _NVENC_MIN <= min(width, height) and max(width, height) <= _NVENC_LIMIT
    if fits and _nvenc_available(ffmpeg):
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            quality,
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", quality, "-pix_fmt", "yuv420p"]


def _transcode(
    ffmpeg: Path,
    source: Path,
    dest: Path,
    width: int,
    height: int,
    fps: float,
    engine,
    mode: str,
    mark: str,
    cancel,
    keep_audio: bool,
    crf: int,
    remove_visible,
    upscaler=None,
    model: str = "photo",
    scale: int = 4,
    on_progress=None,
    remove: bool = True,
    frames: int = 0,
    preset_fn=None,
    preset_info=None,
    force_x264: bool = False,
):
    from engine import Stopped

    pad_x = width % 2
    pad_y = height % 2
    even_w = width + pad_x
    even_h = height + pad_y
    out_w = even_w if upscaler is None else even_w * int(scale)
    out_h = even_h if upscaler is None else even_h * int(scale)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(dest)
    err_path = _encoder_log()
    decode = None
    encode = None
    used_nvenc = False
    colour = _colour(ffmpeg, source, height)
    # A tagged clip is already decoded with its own matrix. Guess only when it has none.
    decode_filter = (
        []
        if colour["tagged"]
        else ["-vf", f"scale=in_color_matrix={colour['matrix']}:flags=accurate_rnd+full_chroma_int"]
    )
    try:
        decode = subprocess.Popen(
            [
                str(ffmpeg),
                "-v",
                "error",
                "-i",
                str(source),
                *decode_filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        command = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{out_w}x{out_h}",
            "-r",
            f"{fps:.5f}",
            "-i",
            "pipe:0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
        ]
        if keep_audio:
            command += ["-map", "1:a?", *AUDIO_ARGS]
        else:
            command += ["-an"]
        if force_x264:
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)), "-pix_fmt", "yuv420p"]
        else:
            enc = _encoder_args(ffmpeg, crf, out_w, out_h)
        used_nvenc = "h264_nvenc" in enc
        command += enc
        command += _colour_args(colour)
        if dest.suffix.lower() in {".mp4", ".mov"}:
            command += ["-movflags", "+faststart"]
        command.append(str(partial))
        err_handle = err_path.open("wb")
        encode = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=err_handle,
            creationflags=_NO_WINDOW,
        )
        err_handle.close()
        assert decode.stdout is not None and encode.stdin is not None
        need = width * height * 3
        done = 0
        first_info = None
        # realesrgan-ncnn overlaps load, process, and save. The reader fills
        # the next frame while this one is enlarged, and the writer feeds
        # the encoder while the GPU does the next frame.
        frames_in: queue.Queue = queue.Queue(maxsize=2)
        frames_out: queue.Queue = queue.Queue(maxsize=2)
        errors: list[BaseException] = []
        stop_read = threading.Event()

        def read_frames() -> None:
            try:
                while not stop_read.is_set():
                    if cancel is not None and cancel.is_set():
                        break
                    raw = _read_exact(decode.stdout, need)
                    if raw is None:
                        break
                    while not stop_read.is_set():
                        try:
                            frames_in.put(raw, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                errors.append(exc)
            finally:
                while not stop_read.is_set():
                    try:
                        frames_in.put(None, timeout=0.2)
                        return
                    except queue.Full:
                        continue

        def write_frames() -> None:
            try:
                while True:
                    piece = frames_out.get()
                    if piece is None:
                        return
                    encode.stdin.write(piece)
            except Exception as exc:
                errors.append(exc)
                stop_read.set()

        reader = threading.Thread(target=read_frames, name="gwr-video-read", daemon=True)
        writer = threading.Thread(target=write_frames, name="gwr-video-write", daemon=True)
        reader.start()
        writer.start()
        remembered = {"ready": False, "fn": None, "info": None}

        def remove_fast(engine, rgb, mode, mark):
            if preset_fn is not None and preset_info is not None:
                return preset_fn(rgb), preset_info
            if not remembered["ready"]:
                from engine import prepare_frame_removal

                cleaned, info, fn = prepare_frame_removal(engine, rgb, mode, mark)
                remembered["ready"] = True
                remembered["fn"] = fn
                remembered["info"] = info
                return cleaned, info
            info = remembered["info"]
            fn = remembered["fn"]
            if fn is None or info is None or info.status != "removed":
                return rgb, info
            return fn(rgb), info
        failed: BaseException | None = None
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise Stopped()
                if errors:
                    raise errors[0]
                try:
                    raw = frames_in.get(timeout=0.2)
                except queue.Empty:
                    if not reader.is_alive() and frames_in.empty():
                        if errors:
                            raise errors[0]
                        break
                    continue
                if raw is None:
                    if errors:
                        raise errors[0]
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                if pad_x or pad_y:
                    frame = np.pad(frame, ((pad_y, 0), (pad_x, 0), (0, 0)), mode="edge")
                out, info = _output_frame(
                    frame, remove, engine, mode, mark, remove_fast, upscaler, model, scale, cancel
                )
                if first_info is None:
                    first_info = info
                if out.shape[1] != out_w or out.shape[0] != out_h:
                    raise RuntimeError("The frame size changed during encoding.")
                payload = out.tobytes()
                while True:
                    if errors:
                        raise errors[0]
                    try:
                        frames_out.put(payload, timeout=0.2)
                        break
                    except queue.Full:
                        if cancel is not None and cancel.is_set():
                            raise Stopped()
                done += 1
                if on_progress is not None:
                    on_progress(done, frames)
        except BaseException as exc:
            failed = exc
        finally:
            stop_read.set()
            _kill(decode)
            try:
                frames_in.get_nowait()
            except queue.Empty:
                pass
            reader.join(timeout=2)
            # On success the writer must finish every queued frame. Dropping
            # one to unblock the queue cuts the end off the file.
            if failed is None and not errors:
                aborted = False
                while writer.is_alive():
                    try:
                        frames_out.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        if cancel is not None and cancel.is_set():
                            aborted = True
                            break
                if not aborted:
                    while writer.is_alive():
                        writer.join(timeout=0.2)
                        if cancel is not None and cancel.is_set():
                            aborted = True
                            break
                if aborted:
                    errors.append(Stopped())
                    _kill(encode)
            else:
                try:
                    frames_out.put(None, timeout=0.2)
                except queue.Full:
                    try:
                        frames_out.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frames_out.put(None, timeout=0.2)
                    except queue.Full:
                        pass
                writer.join(timeout=2)
        if failed is not None:
            raise failed
        if errors:
            raise errors[0]
        try:
            encode.stdin.close()
        except OSError as error:
            encode.stdin = None
            try:
                encode.wait(timeout=10)
            except Exception:
                pass
            detail = _log_tail(err_path)
            raise RuntimeError(detail or str(error)) from error
        encode.stdin = None
        code = encode.wait()
        decode.wait(timeout=30)
        if code != 0:
            detail = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            raise RuntimeError(detail[-1][:240] if detail else "Could not write the video.")
        partial.replace(dest)
        return first_info
    except Stopped:
        if partial.exists():
            partial.unlink(missing_ok=True)
        raise
    except Exception:
        if partial.exists():
            partial.unlink(missing_ok=True)
        if used_nvenc and not force_x264:
            return _transcode(
                ffmpeg,
                source,
                dest,
                width,
                height,
                fps,
                engine,
                mode,
                mark,
                cancel,
                keep_audio,
                crf,
                remove_visible,
                upscaler=upscaler,
                model=model,
                scale=scale,
                on_progress=on_progress,
                remove=remove,
                frames=frames,
                preset_fn=preset_fn,
                preset_info=preset_info,
                force_x264=True,
            )
        raise
    finally:
        _kill(decode)
        _kill(encode)
        try:
            err_path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_exact(stream, size: int) -> bytes | None:
    chunks = []
    got = 0
    while got < size:
        part = stream.read(size - got)
        if not part:
            return None
        chunks.append(part)
        got += len(part)
    return b"".join(chunks)


def _partial_path(dest: Path) -> Path:
    """A temporary file that still ends in the container extension.

    FFmpeg refuses to mux a name such as clip.mp4.partial.
    """
    return dest.with_name(dest.stem + ".partial" + dest.suffix)


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return lines[-1][:240] if lines else ""


def _encoder_log() -> Path:
    """A log file the encoder can write. The descriptor from mkstemp is closed first."""
    fd, name = tempfile.mkstemp(prefix="gwr-ffmpeg-", suffix=".log")
    os.close(fd)
    return Path(name)


def _kill(proc) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
