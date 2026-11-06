"""Frame Heart.

Desktop app and command-line tool. Images are processed on this computer.
Original files are never overwritten.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import home  # The portable build follows this import so the shared-folder home is included.
from engine import MODES, CleanInfo, Engine, collect_images, is_video, process_file
from marks import MARK_IDS
from models import find_entry, is_ready, resolve
from upscale import Upscaler
from version import __version__
from video import video_dest

TASKS = ("clean", "upscale")


def _log_path() -> Path:
    if getattr(sys, "frozen", False):
        return home.home_dir() / "frameheart.log"
    return Path(__file__).resolve().parent / "frameheart.log"


def _write_log(text: str) -> None:
    try:
        with _log_path().open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
    except OSError:
        pass


def _jobs_for(folder: Path | None, extra: list[Path], output: Path, recursive: bool) -> list[tuple[Path, Path]]:
    sources: list[Path] = []
    if folder is not None and folder.is_dir():
        exclude = output if output.exists() else None
        sources.extend(collect_images(folder, recursive, exclude))
    seen = {path.resolve() for path in sources}
    for path in extra:
        if path.is_file() and path.resolve() not in seen:
            sources.append(path)
            seen.add(path.resolve())

    jobs: list[tuple[Path, Path]] = []
    used: set[Path] = set()
    for source in sources:
        if folder is not None:
            try:
                relative = source.resolve().relative_to(folder.resolve())
            except ValueError:
                relative = Path(source.name)
        else:
            relative = Path(source.name)
        dest = output / relative
        # WebM and M4V are saved as MP4, so two sources can share one real file.
        target = video_dest(dest) if is_video(source) else dest
        number = 2
        while target.resolve() in used:
            dest = output / relative.with_name(f"{source.stem}_{number}{source.suffix}")
            target = video_dest(dest) if is_video(source) else dest
            number += 1
        used.add(target.resolve())
        jobs.append((source, dest))
    return jobs


def _output_suffix(task: str, combine: bool = False, videos_only: bool = False) -> str:
    """Video-only jobs stay `_cleaned`. Pictures use `_upscaled` when they are enlarged."""
    if videos_only:
        return "_cleaned"
    return "_upscaled" if task == "upscale" or combine else "_cleaned"


def _display_name(source: Path, folder: Path | None) -> str:
    if folder is not None:
        try:
            return str(source.resolve().relative_to(folder.resolve()))
        except ValueError:
            pass
    return source.name


def _format_line(name: str, info: CleanInfo, combine: bool = False) -> str:
    if info.detail == "Stopped":
        return f"Stopped    {name}"
    if combine and info.status == "removed":
        label = "Cleaned and upscaled"
    else:
        label = {"removed": "Cleaned  ", "skipped": "No mark  ", "upscaled": "Upscaled ", "failed": "Failed   "}.get(
            info.status, info.status.ljust(9)
        )
    return f"{label}  {name}  {info.detail}"


def format_summary(
    removed: int = 0,
    upscaled: int = 0,
    unchanged: int = 0,
    failed: int = 0,
    stopped: int = 0,
    combined: int = 0,
) -> str:
    """Counts for the Done line. Zero counts are left out."""
    parts = []
    if combined:
        parts.append(f"{combined} cleaned and upscaled")
    if removed:
        parts.append(f"{removed} cleaned")
    if upscaled:
        parts.append(f"{upscaled} upscaled")
    if unchanged:
        parts.append(f"{unchanged} had no mark")
    if failed:
        parts.append(f"{failed} failed")
    if stopped:
        parts.append(f"{stopped} not started")
    if not parts:
        parts.append("0 finished")
    return ", ".join(parts)


def cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="Frame Heart",
        description="Remove a visible AI corner watermark from images and videos, or upscale images. Originals are not modified.",
    )
    parser.add_argument("--version", action="version", version=f"Frame Heart {__version__}")
    parser.add_argument("input", type=Path, help="Image, video, or folder")
    parser.add_argument("output", type=Path, nargs="?", help="Destination file or folder")
    parser.add_argument("--mode", choices=MODES, default="auto", help="Gemini layout: auto, standard, or current")
    parser.add_argument(
        "--mark",
        choices=MARK_IDS,
        default="auto",
        help="auto, gemini, grok, doubao, doubao-chat, jimeng, tongyi, qwen, kling, or wenxin",
    )
    parser.add_argument("--recursive", action="store_true", help="Include subfolders when the input is a folder")
    parser.add_argument("--quality", type=int, default=95, help="JPEG and WebP quality, 80-100")
    parser.add_argument("--task", choices=TASKS, default="clean", help="clean or upscale")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Remove the watermark and upscale pictures. A video is only cleaned.",
    )
    parser.add_argument(
        "--model",
        default="photo",
        help="photo, anime, general, anime-detail, compact, nomos, siax, ultrasharp, animesharp, remacri, or a model you added",
    )
    parser.add_argument("--scale", type=int, default=4, help="2, 3, or 4, when the model supports it")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="Processor for upscaling")
    parser.add_argument("--video-crf", type=int, default=18, help="Video compression, 16-28. 18 keeps a sharp picture.")
    parser.add_argument("--no-audio", action="store_true", help="Leave the soundtrack out of the video")
    args = parser.parse_args(argv)
    quality = min(100, max(80, int(args.quality)))
    video_crf = min(28, max(16, int(args.video_crf)))
    keep_audio = not args.no_audio

    source = args.input
    if not source.exists():
        print(f"Input not found: {source}")
        return 1
    if source.is_file():
        peeked = [source]
    elif source.is_dir():
        exclude = args.output if args.output is not None and args.output.exists() else None
        peeked = collect_images(source, args.recursive, exclude)
    else:
        peeked = []
    videos_only = bool(peeked) and all(is_video(path) for path in peeked)
    needs_model = (args.task == "upscale" or args.combine) and not videos_only

    engine = Engine()
    scaler = None
    if needs_model:
        try:
            resolved = resolve(args.model, args.scale)
        except ValueError as error:
            print(error)
            return 1
        if not is_ready(args.model, args.scale):
            print(f"{resolved.title} is not on this computer. Open the app and download it, or use photo or anime.")
            return 1
        scaler = Upscaler(prefer_gpu=args.device == "gpu")
        print(f"Upscaling on {scaler.device}.")
        print(resolved.phrase)
        entry = find_entry(args.model)
        if scaler.device == "CPU" and entry is not None and entry.heavy:
            print(f"{entry.title} on CPU is slow for large images.")

    suffix = _output_suffix(args.task, args.combine, videos_only)
    if source.is_file():
        dest = args.output or source.with_name(f"{source.stem}{suffix}{source.suffix}")
        file_combine = args.combine and not is_video(source)
        info = process_file(
            engine,
            source,
            dest,
            args.mode,
            quality,
            task=args.task,
            upscaler=None if is_video(source) else scaler,
            model=args.model,
            scale=args.scale,
            mark=args.mark,
            keep_audio=keep_audio,
            video_crf=video_crf,
            combine=args.combine,
        )
        print(_format_line(source.name, info, file_combine))
        if info.status in {"removed", "upscaled"}:
            print(video_dest(dest) if is_video(source) else dest)
        return 0 if info.status in {"removed", "skipped", "upscaled"} else 1

    dest_root = args.output or source.with_name(f"{source.name}{suffix}")
    if dest_root.resolve() == source.resolve():
        print("Choose a different output folder. Originals are not overwritten.")
        return 1
    jobs = _jobs_for(source, [], dest_root, args.recursive)
    if not jobs:
        print("No files found.")
        return 1
    removed = upscaled = unchanged = failed = combined = 0
    for index, (src, dest) in enumerate(jobs, start=1):
        info = process_file(
            engine,
            src,
            dest,
            args.mode,
            quality,
            task=args.task,
            upscaler=None if is_video(src) else scaler,
            model=args.model,
            scale=args.scale,
            mark=args.mark,
            keep_audio=keep_audio,
            video_crf=video_crf,
            combine=args.combine,
        )
        file_combine = args.combine and not is_video(src)
        print(f"[{index}/{len(jobs)}] {_format_line(_display_name(src, source), info, file_combine)}")
        if file_combine and info.status == "removed":
            combined += 1
        elif info.status == "removed":
            removed += 1
        elif info.status == "upscaled":
            upscaled += 1
        elif info.status == "skipped":
            unchanged += 1
        else:
            failed += 1
    print(f"Done. {format_summary(removed, upscaled, unchanged, failed, combined=combined)}.")
    print(dest_root)
    return 0 if failed == 0 else 1


def selftest() -> int:
    import test_engine

    test_engine.main()
    print("gui ok")
    return 0


def main() -> int:
    from upscale import shutdown

    code = 0
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
            try:
                code = selftest()
            except Exception:
                detail = traceback.format_exc()
                _write_log(detail)
                print(detail, file=sys.stderr)
                code = 1
        elif len(sys.argv) > 1:
            code = cli(sys.argv[1:])
        else:
            from desktop import launch

            try:
                launch()
            except Exception:
                detail = traceback.format_exc()
                _write_log(detail)
                print(detail, file=sys.stderr)
                code = 1
    finally:
        shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
