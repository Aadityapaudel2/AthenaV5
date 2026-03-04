[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$PathPrefix = "/AthenaV5",
    [string]$Hostname = "",
    [string]$TunnelName = "",
    [switch]$QuickTunnel,
    [string]$CloudflaredExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-CloudflaredExe {
    param([string]$ExplicitPath)

    if ($ExplicitPath -and (Test-Path -LiteralPath $ExplicitPath)) {
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $candidates = @(
        (Join-Path $ProjectRoot "cloudflared.exe"),
        (Join-Path (Split-Path -Parent $ProjectRoot) "cloudflared.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }

    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "cloudflared executable not found. Set -CloudflaredExe or install cloudflared."
}

$CloudflaredPath = Resolve-CloudflaredExe -ExplicitPath $CloudflaredExe
$OriginUrl = "http://127.0.0.1:$Port"

Write-Host "Starting Cloudflared tunnel..."
Write-Host "cloudflared=$CloudflaredPath"
Write-Host "origin=$OriginUrl"

if ($TunnelName -and -not $QuickTunnel) {
    Write-Host "mode=named-tunnel tunnel=$TunnelName"
    if ($Hostname) {
        Write-Host "browse=https://$Hostname$PathPrefix"
    }
    & $CloudflaredPath tunnel run $TunnelName
    exit $LASTEXITCODE
}

if ($Hostname -and -not $QuickTunnel) {
    Write-Host "mode=hostname hostname=$Hostname"
    Write-Host "browse=https://$Hostname$PathPrefix"
    & $CloudflaredPath tunnel --url $OriginUrl --hostname $Hostname
    exit $LASTEXITCODE
}

Write-Host "mode=quick-tunnel"
Write-Host "After start, open the printed trycloudflare URL + $PathPrefix"
& $CloudflaredPath tunnel --url $OriginUrl
exit $LASTEXITCODE
