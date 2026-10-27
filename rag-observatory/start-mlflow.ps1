param([int]$Port = 5000)
$ErrorActionPreference = 'Stop'
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Project environment missing. Install the requirements first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path artifacts -Force | Out-Null
    & $ProjectPython -X utf8 -m mlflow server --host 127.0.0.1 --port $Port --workers 1 --backend-store-uri sqlite:///artifacts/mlflow.db --artifacts-destination ./artifacts/mlartifacts
} finally {
    Pop-Location
}
