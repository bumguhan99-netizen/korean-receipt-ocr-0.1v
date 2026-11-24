$ErrorActionPreference = 'Stop'
$logPath = Join-Path $PSScriptRoot 'enable_wsl.log'
Start-Transcript -LiteralPath $logPath -Append
try {
    $results = @()
    foreach ($feature in @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')) {
        $results += Enable-WindowsOptionalFeature -Online -FeatureName $feature -All -NoRestart
    }
    $results | Format-List
    $results | Select-Object FeatureName, RestartNeeded | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'wsl_features.json') -Encoding UTF8
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'wsl_features_error.txt') -Encoding UTF8
    throw
} finally {
    Stop-Transcript
}
