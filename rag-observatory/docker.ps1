# Use Docker Desktop's bundled Compose when `docker compose` is not on PATH.
# No advanced parameter binding: otherwise PowerShell consumes Compose's -d
# as its own -Debug switch instead of forwarding detached mode.
$ComposeArgs = $args
$ErrorActionPreference = 'Stop'
$compose = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker-compose.exe'
Push-Location $PSScriptRoot
try {
    if (Test-Path $compose) { & $compose @ComposeArgs }
    else { & docker compose @ComposeArgs }
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
