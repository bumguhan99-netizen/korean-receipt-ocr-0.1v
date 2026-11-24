$ErrorActionPreference = 'Stop'
$log = Join-Path $PSScriptRoot 'performance_settings.log'
& powercfg /setactive e9a42b02-d5df-448d-aa00-03f14749eb61 2>&1 | Out-File -LiteralPath $log -Encoding utf8
& powercfg /getactivescheme 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
# Values queried from this GPU's own supported clocks. Power limit already at 180 W maximum.
& nvidia-smi -lgc 3090,3090 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
"Graphics clock command exit: $LASTEXITCODE" | Out-File -LiteralPath $log -Encoding utf8 -Append
& nvidia-smi -lmc 14001,14001 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
"Memory clock command exit: $LASTEXITCODE" | Out-File -LiteralPath $log -Encoding utf8 -Append
& nvidia-smi --query-gpu=power.limit,clocks.gr,clocks.mem,temperature.gpu --format=csv 2>&1 | Out-File -LiteralPath $log -Encoding utf8 -Append
