"""Where the portable app keeps its files."""

from __future__ import annotations

import sys
from pathlib import Path

# Portable apps that write next to their exe. Sharing one folder mixes data\.
_FAMILY = frozenset({"stillheart.exe", "rushheart.exe", "frameheart.exe"})
# The folders these apps used under their earlier names.
_OLD_FOLDER = {"stillheart.exe": "Stills", "rushheart.exe": "Rushes", "frameheart.exe": "Plates"}
_OWN = {
    "rushheart.exe": ("rushheart.log", "rushheart.log.old", "rushes.log", "rushes.log.old", "captures"),
    "stillheart.exe": ("stillheart.log", "stillheart.log.old", "stills.log", "stills.log.old", "downloads", "gallery-dl.exe"),
    "frameheart.exe": ("frameheart.log", "remover.log", "models", "tools"),
}
_DATA = {
    "rushheart.exe": ("schedule.json", "diag.txt"),
    "stillheart.exe": ("archive.sqlite3", "extractors.json", "scans.json"),
    "frameheart.exe": (),
}
_BIN = {
    "rushheart.exe": ("yt-dlp.exe", "yt-dlp.exe.old", "yt-dlp-download.exe"),
    "stillheart.exe": ("gallery-dl.exe", "gallery-dl.exe.old"),
    "frameheart.exe": (),
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

    FrameHeart.exe always uses a FrameHeart folder. The same rule applies to
    Still Heart and Rush Heart, whether or not the others are nearby. An exe that already sits
    inside a folder of its own name uses that folder.
    """
    exe = exe.resolve()
    folder = exe.parent
    name = exe.name.lower()
    if name not in _FAMILY or folder.name.lower() == exe.stem.lower():
        return folder
    old = _OLD_FOLDER[name]
    # An exe already inside its old folder stays there, with its data.
    if folder.name.lower() == old.lower():
        return folder
    home = folder / exe.stem
    if home.exists() and not home.is_dir():
        return folder
    legacy = folder / old
    if legacy.is_dir() and not home.exists():
        # Settings, models and history follow the app to its new name.
        try:
            legacy.rename(home)
        except OSError:
            return legacy
    _claim(folder, home, name)
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        return folder
    return home


def home_dir() -> Path:
    if getattr(sys, "frozen", False):
        return portable_home(Path(sys.executable))
    return Path(__file__).resolve().parent
