# Build the single portable FrameHeart.exe into dist\.
Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  py -3.11 -m venv .venv
  & $py -m pip install -r requirements.txt pyinstaller
}

& $py test_engine.py
if ($LASTEXITCODE -ne 0) { throw "The tests failed, so nothing was built." }

& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build\onefile-work plates.spec
Write-Host "Built dist\FrameHeart.exe"
