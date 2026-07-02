param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot "ragbaarnet-env\Scripts\python.exe"

if (!(Test-Path $pythonExe)) {
    throw "Python venv not found at: $pythonExe"
}

Write-Host "Starting RagbaarNet platform server..."
Write-Host "UI: http://127.0.0.1:$Port/ui/"
Write-Host "Phone (same network): http://<YOUR_LAPTOP_IP>:$Port/ui/"
Write-Host ""

& $pythonExe "modules\Platform\processor.py" --host $Host --port $Port
