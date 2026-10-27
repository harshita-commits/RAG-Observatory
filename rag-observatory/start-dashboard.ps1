param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Project environment missing. Run python -m venv .venv and install the requirements first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $ProjectPython -X utf8 -m observatory serve --port $Port
} finally {
    Pop-Location
}
