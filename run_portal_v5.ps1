[CmdletBinding()]
param(
    [bool]$LoadModel = $true,
    [bool]$AuthRequired = $true,
    [int]$Port = 8000,
    [string]$PathPrefix = "/AthenaV5",
    [string]$Hostname = "portal.neohmlabs.com",
    [string]$TunnelName = "athena-portal",
    [string]$ModelDir = "",
    [string]$LogRoot = "",
    [bool]$CookieSecure = $true,
    [bool]$LogDeltas = $false,
    [switch]$QuickTunnel,
    [string]$PythonExe = "",
    [string]$CloudflaredExe = "",
    [int]$HealthTimeoutSec = 90,
    [string]$AuthEnvFile = "",
    [switch]$AllowAuthFallback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PortalScript = Join-Path $ProjectRoot "portal_server.py"
$CloudflaredScript = Join-Path $ProjectRoot "cloudflared_athenav5.ps1"

if (-not (Test-Path -LiteralPath $PortalScript)) {
    throw "portal_server.py not found: $PortalScript"
}
if (-not (Test-Path -LiteralPath $CloudflaredScript)) {
    throw "cloudflared_athenav5.ps1 not found: $CloudflaredScript"
}

function Resolve-PythonExe {
    param([string]$ExplicitPath)

    if ($ExplicitPath -and (Test-Path -LiteralPath $ExplicitPath)) {
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $ProjectRoot) ".venv\Scripts\python.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    throw "python executable not found. Set -PythonExe or activate a venv."
}

function Import-EnvFile {
    param(
        [string]$FilePath,
        [switch]$OverrideExisting
    )

    if (-not (Test-Path -LiteralPath $FilePath)) {
        return $false
    }

    foreach ($rawLine in Get-Content -LiteralPath $FilePath -ErrorAction Stop) {
        $line = $rawLine.Trim()
        if (-not $line) { continue }
        if ($line.StartsWith("#")) { continue }
        if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') { continue }

        $name = $matches[1]
        $value = $matches[2].Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            if ($value.Length -ge 2) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }

        if (-not $OverrideExisting) {
            $existing = ""
            if (Test-Path "env:$name") {
                $existing = (Get-Item "env:$name").Value
            }
            if ($existing -and $existing.Trim().Length -gt 0) {
                continue
            }
        }
        Set-Item -Path "env:$name" -Value $value
    }

    Write-Host "Loaded env values from $FilePath"
    return $true
}

function Test-PortAvailable {
    param([int]$CandidatePort)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $CandidatePort)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) {
            try {
                $listener.Stop()
            } catch {
                Write-Verbose "Failed to stop temporary port listener: $($_.Exception.Message)"
            }
        }
    }
}

function Resolve-RunPort {
    param([int]$RequestedPort)
    if (Test-PortAvailable -CandidatePort $RequestedPort) {
        return $RequestedPort
    }
    for ($p = $RequestedPort + 1; $p -le ($RequestedPort + 100); $p++) {
        if (Test-PortAvailable -CandidatePort $p) {
            Write-Warning "Requested port $RequestedPort is busy. Using port $p instead."
            return $p
        }
    }
    throw "No free port found in range $RequestedPort..$($RequestedPort + 100)."
}

$ResolvedPython = Resolve-PythonExe -ExplicitPath $PythonExe
$EffectivePort = Resolve-RunPort -RequestedPort $Port

$resolvedAuthEnvFile = ""
if ($AuthEnvFile -and $AuthEnvFile.Trim().Length -gt 0) {
    $candidate = $AuthEnvFile.Trim()
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $ProjectRoot $candidate
    }
    $resolvedAuthEnvFile = (Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue).Path
    if (-not $resolvedAuthEnvFile) {
        throw "Auth env file not found: $candidate"
    }
    [void](Import-EnvFile -FilePath $resolvedAuthEnvFile -OverrideExisting)
} else {
    $defaultEnvFiles = @(
        (Join-Path $ProjectRoot "portal_auth.env"),
        (Join-Path $ProjectRoot ".env.portal"),
        (Join-Path $ProjectRoot ".env")
    )
    foreach ($f in $defaultEnvFiles) {
        if (Import-EnvFile -FilePath $f -OverrideExisting) {
            $resolvedAuthEnvFile = $f
            break
        }
    }
}

$env:ATHENA_PORTAL_PATH_PREFIX = $PathPrefix
$env:ATHENA_PORTAL_PORT = [string]$EffectivePort
$env:ATHENA_WEB_LOAD_MODEL = if ($LoadModel) { "1" } else { "0" }
$env:ATHENA_AUTH_REQUIRED = if ($AuthRequired) { "1" } else { "0" }
$env:ATHENA_PORTAL_COOKIE_SECURE = if ($CookieSecure) { "1" } else { "0" }
$env:ATHENA_LOG_DELTAS = if ($LogDeltas) { "1" } else { "0" }
if ($ModelDir -and $ModelDir.Trim().Length -gt 0) {
    $env:ATHENA_MODEL_DIR = $ModelDir.Trim()
}
if (-not $env:ATHENA_MODEL_DIR) {
    $ResolvedModelDir = & $ResolvedPython -c "from athena_paths import get_default_chat_model_dir; print(str(get_default_chat_model_dir()))"
    if ($LASTEXITCODE -eq 0 -and $ResolvedModelDir) {
        $env:ATHENA_MODEL_DIR = $ResolvedModelDir.Trim()
    }
}
if ($LogRoot -and $LogRoot.Trim().Length -gt 0) {
    $env:ATHENA_LOG_ROOT = $LogRoot.Trim()
} elseif (-not $env:ATHENA_LOG_ROOT) {
    $env:ATHENA_LOG_ROOT = (Join-Path $ProjectRoot "data\users")
}

if (-not (Test-Path -LiteralPath $env:ATHENA_MODEL_DIR)) {
    throw "Model directory does not exist: $($env:ATHENA_MODEL_DIR)"
}

$resolvedLogRoot = $env:ATHENA_LOG_ROOT
New-Item -ItemType Directory -Path $resolvedLogRoot -Force | Out-Null
$writeProbe = Join-Path $resolvedLogRoot ".write_test"
Set-Content -Path $writeProbe -Value "ok" -Encoding UTF8
Remove-Item -Path $writeProbe -Force

if ($AuthRequired) {
    if (-not $env:ATHENA_AUTH_REDIRECT_URI) {
        $env:ATHENA_AUTH_REDIRECT_URI = "https://portal.neohmlabs.com/AthenaV5/auth/callback"
    }
    $requiredAuthEnv = @(
        "ATHENA_GOOGLE_CLIENT_ID",
        "ATHENA_GOOGLE_CLIENT_SECRET",
        "ATHENA_AUTH_REDIRECT_URI",
        "ATHENA_PORTAL_SESSION_SECRET"
    )
    $missing = @()
    foreach ($name in $requiredAuthEnv) {
        $value = ""
        if (Test-Path "env:$name") {
            $value = (Get-Item "env:$name").Value
        }
        if (-not $value -or $value.Trim().Length -eq 0) {
            $missing += $name
        }
    }
    $placeholderHits = @()
    $placeholderPatterns = @(
        "your-google-client-id",
        "your-google-client-secret",
        "replace-with-long-random-string",
        "PASTE_",
        "paste-"
    )
    foreach ($name in $requiredAuthEnv) {
        $value = ""
        if (Test-Path "env:$name") {
            $value = (Get-Item "env:$name").Value
        }
        foreach ($pat in $placeholderPatterns) {
            if ($value -and $value.ToLower().Contains($pat.ToLower())) {
                $placeholderHits += $name
                break
            }
        }
    }
    if ($missing.Count -gt 0) {
        $msg = "Auth is enabled but required env vars are missing: $($missing -join ', ')."
        if ($AllowAuthFallback) {
            Write-Warning $msg
            Write-Warning "AllowAuthFallback is set. Falling back to AuthRequired=false for this run."
            $AuthRequired = $false
            $env:ATHENA_AUTH_REQUIRED = "0"
        } else {
            throw ($msg + " Set them before launch, or run with -AllowAuthFallback to bypass auth temporarily.")
        }
    }
    if ($placeholderHits.Count -gt 0) {
        throw "Auth env values still look like template placeholders for: $($placeholderHits -join ', '). Update portal_auth.env with real values."
    }
}

Write-Host "Starting Athena V5 portal stack..."
Write-Host "python=$ResolvedPython"
Write-Host "path_prefix=$PathPrefix port=$EffectivePort load_model=$($env:ATHENA_WEB_LOAD_MODEL)"
if ($env:ATHENA_MODEL_DIR) {
    Write-Host "model_override=$($env:ATHENA_MODEL_DIR)"
}
Write-Host "auth_required=$($env:ATHENA_AUTH_REQUIRED) cookie_secure=$($env:ATHENA_PORTAL_COOKIE_SECURE)"
Write-Host "log_root=$($env:ATHENA_LOG_ROOT)"
Write-Host "log_deltas=$($env:ATHENA_LOG_DELTAS)"
if ($resolvedAuthEnvFile) {
    Write-Host "auth_env_file=$resolvedAuthEnvFile"
}

$portalProc = Start-Process -FilePath $ResolvedPython -ArgumentList @($PortalScript) -WorkingDirectory $ProjectRoot -PassThru
Write-Host "Started portal_server.py (pid=$($portalProc.Id)). Waiting for health check..."

$healthUrl = "http://127.0.0.1:$EffectivePort/healthz"
$ok = $false
$deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 800
    if ($portalProc.HasExited) {
        throw "portal_server.py exited early with code $($portalProc.ExitCode)."
    }
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3
        if ($resp.ok -eq $true) {
            Write-Host "healthz ok: $($resp | ConvertTo-Json -Compress)"
            $ok = $true
            break
        }
    } catch {
        # Keep waiting until timeout.
    }
}
if (-not $ok) {
    Stop-Process -Id $portalProc.Id -Force -ErrorAction SilentlyContinue
    throw "Health check timed out: $healthUrl"
}

$tunnelArgs = @{
    Port = $EffectivePort
    PathPrefix = $PathPrefix
}
if ($Hostname -and $Hostname.Trim().Length -gt 0) {
    $tunnelArgs.Hostname = $Hostname.Trim()
}
if ($TunnelName -and $TunnelName.Trim().Length -gt 0) {
    $tunnelArgs.TunnelName = $TunnelName.Trim()
}
if ($CloudflaredExe -and $CloudflaredExe.Trim().Length -gt 0) {
    $tunnelArgs.CloudflaredExe = $CloudflaredExe.Trim()
}
if ($QuickTunnel) {
    $tunnelArgs.QuickTunnel = $true
}

Write-Host "Launching tunnel in foreground. Press Ctrl+C to stop stack."
try {
    & $CloudflaredScript @tunnelArgs
    $TunnelExit = $LASTEXITCODE

    if ($TunnelExit -ne 0 -and -not $QuickTunnel) {
        Write-Warning "Named/hostname tunnel failed (exit=$TunnelExit). Falling back to quick tunnel."
        $fallbackArgs = @{ Port = $EffectivePort; PathPrefix = $PathPrefix; QuickTunnel = $true }
        if ($CloudflaredExe -and $CloudflaredExe.Trim().Length -gt 0) {
            $fallbackArgs.CloudflaredExe = $CloudflaredExe.Trim()
        }
        & $CloudflaredScript @fallbackArgs
        $TunnelExit = $LASTEXITCODE
    }

    if ($TunnelExit -ne 0) {
        throw "cloudflared exited with code $TunnelExit"
    }
} finally {
    if (-not $portalProc.HasExited) {
        Stop-Process -Id $portalProc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped portal_server.py (pid=$($portalProc.Id))."
    }
}
