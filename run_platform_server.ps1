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

& $pythonExe "modules\Platform\processor.py" --host $Host --port $Port
