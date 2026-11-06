$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script = Join-Path $PSScriptRoot 'run_opensees_model.py'
$candidates = @()
$command = Get-Command python3.12.exe -ErrorAction SilentlyContinue
if ($null -ne $command) {
    $candidates += $command.Source
}
$candidates += 'C:\Program Files\PostgreSQL\16\pgAdmin 4\python\python.exe'

$python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($null -eq $python) {
    throw 'Python 3.12 was not found. Install it and the pinned requirements-opensees-py312-HW3.txt target first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $repo '.opensees-py312\openseespy'))) {
    throw 'Missing .opensees-py312 target. Install HW3/requirements-opensees-py312-HW3.txt into that target.'
}

Push-Location $repo
try {
    & $python $script
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
