# Run after restarting Windows. Existing project/model files are mounted read-only.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$docker = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
$desktop = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'
if (-not (Test-Path -LiteralPath $docker)) { throw 'Docker Desktop CLI is not installed.' }
Start-Process -FilePath $desktop -WindowStyle Hidden
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    $engineOutput = & $docker info --format '{{.OSType}}' 2>&1
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw 'Docker Linux engine unavailable. Check restart and virtualization status.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$recordRoot = Join-Path $PSScriptRoot "records\linux_$stamp"
New-Item -ItemType Directory -Path $recordRoot | Out-Null
& $docker version 2>&1 | Tee-Object -FilePath (Join-Path $recordRoot 'docker_version.txt')
if ($LASTEXITCODE -ne 0) { throw 'Docker version failed' }
& $docker build --progress plain --tag 'mogumogu-korie-eval:20261117' $PSScriptRoot 2>&1 | Tee-Object -FilePath (Join-Path $recordRoot 'build.log')
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed; inspect build.log' }
& $docker image inspect 'mogumogu-korie-eval:20261117' | Set-Content -LiteralPath (Join-Path $recordRoot 'image_inspect.json') -Encoding UTF8
& $docker run --rm --gpus all --shm-size 2g --mount "type=bind,source=$projectRoot,target=/workspace,readonly" 'mogumogu-korie-eval:20261117' 2>&1 | Tee-Object -FilePath (Join-Path $recordRoot 'gpu_check.log')
if ($LASTEXITCODE -ne 0) { throw 'GPU backward/update check failed; OCR must not proceed' }
& $docker run --rm 'mogumogu-korie-eval:20261117' python -m pip freeze | Set-Content -LiteralPath (Join-Path $recordRoot 'packages.txt') -Encoding UTF8
Write-Output "Linux container and GPU checks passed. Records: $recordRoot"
Write-Output 'OCR evaluation is a separate next stage; this script does not claim it is complete.'

