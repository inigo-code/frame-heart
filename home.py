"""Where the portable app keeps its files."""

from __future__ import annotations

import sys
from pathlib import Path

# Portable apps that write next to their exe. Sharing one folder mixes data\.
_FAMILY = frozenset({"stills.exe", "rushes.exe", "plates.exe"})
_OWN = {
    "rushes.exe": ("rushes.log", "rushes.log.old", "captures"),
    "stills.exe": ("stills.log", "stills.log.old", "downloads", "gallery-dl.exe"),
    "plates.exe": ("remover.log", "models", "tools"),
}
_DATA = {
    "rushes.exe": ("schedule.json", "diag.txt"),
    "stills.exe": ("archive.sqlite3", "extractors.json", "scans.json"),
    "plates.exe": (),
}
_BIN = {
    "rushes.exe": ("yt-dlp.exe", "yt-dlp.exe.old", "yt-dlp-download.exe"),
    "stills.exe": ("gallery-dl.exe", "gallery-dl.exe.old"),
    "plates.exe": (),
}


def _move_into(src: Path, dest: Path) -> None:
    if not src.exists() or dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(dest)
    except OSError:
        pass


def _claim(folder: Path, home: Path, exe_name: str) -> None:
    """Move this app's files into its own folder the first time that folder is made."""
    if home.exists():
        return
    foreign = False
    data = folder / "data"
    for other, markers in _OWN.items():
        if other == exe_name:
            continue
        if any((folder / marker).exists() for marker in markers):
            foreign = True
        if any((data / marker).exists() for marker in _DATA[other]):
            foreign = True
    own = any((folder / marker).exists() for marker in _OWN[exe_name]) or any(
        (data / marker).exists() for marker in _DATA[exe_name]
    )
    for marker in _OWN[exe_name]:
        _move_into(folder / marker, home / marker)
    if own and not foreign:
        _move_into(data, home / "data")
    for name in _BIN[exe_name]:
        _move_into(folder / "bin" / name, home / "bin" / name)


def portable_home(exe: Path) -> Path:
    """Folder this portable exe should write into.

    Plates.exe always uses a Plates folder. The same rule applies to Stills and
    Rushes, whether or not the other apps are nearby. An exe that already sits
    inside a folder of its own name uses that folder.
    """
    exe = exe.resolve()
    folder = exe.parent
    if exe.name.lower() not in _FAMILY or folder.name.lower() == exe.stem.lower():
        return folder
    home = folder / exe.stem
    if home.exists() and not home.is_dir():
        return folder
    _claim(folder, home, exe.name.lower())
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return folder
    return home


def home_dir() -> Path:
    if getattr(sys, "frozen", False):
        return portable_home(Path(sys.executable))
    return Path(__file__).resolve().parent
