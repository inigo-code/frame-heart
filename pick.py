"""Folder and file dialogs that stay above the WebView2 window.

The dialog runs in PowerShell. Chosen paths are written to a temp file.
They are never interpolated into the script.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace GwrPick {
  [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
  class FileOpenDialogRCW {}
  [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IShellItem {
    void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
    void GetParent(out IShellItem ppsi);
    void GetDisplayName(uint sigdnName, out IntPtr ppszName);
    void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
    void Compare(IShellItem psi, uint hint, out int piOrder);
  }
  [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IFileOpenDialog {
    [PreserveSig] int Show(IntPtr parent);
    void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
    void SetFileTypeIndex(uint iFileType);
    void GetFileTypeIndex(out uint piFileType);
    void Advise(IntPtr pfde, out uint pdwCookie);
    void Unadvise(uint dwCookie);
    void SetOptions(uint fos);
    void GetOptions(out uint fos);
    void SetDefaultFolder(IShellItem psi);
    void SetFolder(IShellItem psi);
    void GetFolder(out IShellItem ppsi);
    void GetCurrentSelection(out IShellItem ppsi);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
    void GetResult(out IShellItem ppsi);
    void AddPlace(IShellItem psi, int fdap);
    void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
    void Close(int hr);
    void SetClientGuid(ref Guid guid);
    void ClearClientData();
    void SetFilter(IntPtr pFilter);
    void GetResults(out IntPtr ppenum);
    void GetSelectedItems(out IntPtr ppsai);
  }
  public static class FolderDialog {
    public static string Pick(string title, string initial, IntPtr owner) {
      var dialog = (IFileOpenDialog)new FileOpenDialogRCW();
      uint options;
      dialog.GetOptions(out options);
      dialog.SetOptions(options | 0x8 | 0x20 | 0x40 | 0x800 | 0x1000);
      dialog.SetTitle(string.IsNullOrEmpty(title) ? "Choose a folder" : title);
      dialog.SetOkButtonLabel("Select folder");
      if (!string.IsNullOrEmpty(initial) && System.IO.Directory.Exists(initial)) {
        Guid iid = new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE");
        IntPtr item;
        int hr = SHCreateItemFromParsingName(initial, IntPtr.Zero, ref iid, out item);
        if (hr == 0 && item != IntPtr.Zero) {
          try { dialog.SetFolder((IShellItem)Marshal.GetObjectForIUnknown(item)); }
          finally { Marshal.Release(item); }
        }
      }
      int show = dialog.Show(owner);
      if (show != 0) return "";
      IShellItem result;
      dialog.GetResult(out result);
      IntPtr name;
      result.GetDisplayName(0x80058000, out name);
      string path = Marshal.PtrToStringUni(name);
      Marshal.FreeCoTaskMem(name);
      return path ?? "";
    }
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = true)]
    static extern int SHCreateItemFromParsingName([MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc, ref Guid riid, out IntPtr ppv);
  }
}
"@
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Width = 1
$owner.Height = 1
$initial = $env:GWR_PICK_INITIAL
$out = $env:GWR_PICK_OUT
$kind = $env:GWR_PICK_KIND
$owner.Show()
try {
  $chosen = ''
  if ($kind -eq 'folder') {
    $chosen = [GwrPick.FolderDialog]::Pick('Choose a folder', $initial, $owner.Handle)
  } else {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Filter = 'All files (*.*)|*.*'
    if ($kind -eq 'images') {
      $dialog.Title = 'Files'
      $dialog.Filter = 'Images and videos (*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff;*.mp4;*.m4v;*.mov;*.mkv;*.webm)|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff;*.mp4;*.m4v;*.mov;*.mkv;*.webm|All files (*.*)|*.*'
      $dialog.Multiselect = $true
    } elseif ($kind -eq 'param') {
      $dialog.Title = 'NCNN param file'
      $dialog.Filter = 'NCNN param (*.param)|*.param|All files (*.*)|*.*'
    } else {
      $dialog.Title = 'NCNN weights'
      $dialog.Filter = 'NCNN weights (*.bin)|*.bin|All files (*.*)|*.*'
    }
    if ($initial -and (Test-Path -LiteralPath $initial)) {
      $dialog.InitialDirectory = $initial
    }
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
      if ($dialog.Multiselect) {
        $chosen = [string]::Join("`n", $dialog.FileNames)
      } else {
        $chosen = $dialog.FileName
      }
    }
  }
  if ($chosen) {
    [System.IO.File]::WriteAllText($out, $chosen, (New-Object System.Text.UTF8Encoding $false))
  }
} finally {
  $owner.Close()
  $owner.Dispose()
}
"""


def _initial_dir(initial: str) -> str:
    text = (initial or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    try:
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return str(path)
    except OSError:
        return ""
    return ""


def _pick(kind: str, initial: str = "") -> list[str]:
    if sys.platform != "win32":
        raise RuntimeError("The file dialog is only available on Windows")
    handle = tempfile.NamedTemporaryFile(prefix="gwr-pick-", suffix=".txt", delete=False)
    out = Path(handle.name)
    handle.close()
    env = os.environ.copy()
    env["GWR_PICK_KIND"] = kind
    env["GWR_PICK_INITIAL"] = _initial_dir(initial)
    env["GWR_PICK_OUT"] = str(out)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    script = Path(tempfile.gettempdir()) / f"gwr-pick-{os.getpid()}.ps1"
    try:
        script.write_text(_SCRIPT, encoding="utf-8")
        # Windows defaults to Restricted, which refuses -File. Bypass is only
        # for this PowerShell process. It does not change the user's policy.
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            capture_output=True,
            env=env,
            timeout=600,
            creationflags=flags,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(detail.splitlines()[-1][:240] if detail else "Could not open the file dialog")
        if not out.is_file() or out.stat().st_size == 0:
            return []
        text = out.read_text(encoding="utf-8-sig")
        return [line for line in text.splitlines() if line.strip()]
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("The file dialog was left open too long") from exc
    finally:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            script.unlink(missing_ok=True)
        except OSError:
            pass


def pick_folder(initial: str = "") -> str:
    chosen = _pick("folder", initial)
    return chosen[0] if chosen else ""


def pick_files(kind: str, initial: str = "") -> list[str]:
    if kind not in {"images", "param", "bin"}:
        raise ValueError("Unknown picker")
    return _pick(kind, initial)
