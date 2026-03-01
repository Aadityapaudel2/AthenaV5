[CmdletBinding()]
param(
    [string]$ModelDir = "",
    [switch]$LegacyTk,
    [switch]$NoAutoInstallDeps,
    [switch]$NoMathJaxBootstrap,
    [switch]$BootstrapVerbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$QtUi = Join-Path $ProjectRoot "qt_ui.py"
$TkUi = Join-Path $ProjectRoot "ui.py"
$BootstrapScript = Join-Path $ProjectRoot "scripts\bootstrap_mathjax.ps1"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

function Resolve-PythonExe {
    $Candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $ProjectRoot) ".venv\Scripts\python.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCmd) {
        return $PythonCmd.Source
    }
    throw "No Python runtime found. Activate/create a venv first."
}

function Test-QtDeps {
    param([string]$PythonExe)
    & $PythonExe -c "import PySide6; from PySide6.QtWebEngineWidgets import QWebEngineView" 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Install-UiDeps {
    param([string]$PythonExe)
    if (Test-Path -LiteralPath $Requirements) {
        & $PythonExe -m pip install --disable-pip-version-check -r $Requirements
    } else {
        & $PythonExe -m pip install --disable-pip-version-check PySide6 PySide6-Addons markdown-it-py
    }
    return ($LASTEXITCODE -eq 0)
}

function Invoke-MathJaxBootstrap {
    if ($NoMathJaxBootstrap) { return }
    if (-not (Test-Path -LiteralPath $BootstrapScript)) { return }
    if ($BootstrapVerbose) { Write-Host "[bootstrap] running MathJax bootstrap..." }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BootstrapScript -ProjectRoot $ProjectRoot
}

$PythonExe = Resolve-PythonExe
$LaunchTk = [bool]$LegacyTk
$ExitCode = 0
$ResolvedModelDir = ""
if ($ModelDir -and $ModelDir.Trim().Length -gt 0) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($ModelDir)) { $ModelDir } else { Join-Path $ProjectRoot $ModelDir }
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Model path not found: $Candidate"
    }
    $ResolvedModelDir = (Resolve-Path -LiteralPath $Candidate).Path
}

if (-not $LaunchTk) {
    if (-not (Test-Path -LiteralPath $QtUi)) {
        Write-Warning "qt_ui.py not found. Falling back to Tk UI."
        $LaunchTk = $true
    } else {
        $QtReady = Test-QtDeps -PythonExe $PythonExe
        if (-not $QtReady -and -not $NoAutoInstallDeps) {
            Write-Host "Installing UI dependencies..."
            $null = Install-UiDeps -PythonExe $PythonExe
            $QtReady = Test-QtDeps -PythonExe $PythonExe
        }
        if (-not $QtReady) {
            Write-Warning "Qt deps missing. Use -LegacyTk or install requirements."
            $LaunchTk = $true
        } else {
            Invoke-MathJaxBootstrap
        }
    }
}

Set-Location -LiteralPath $ProjectRoot

if (-not $LaunchTk) {
    if (-not $env:QTWEBENGINE_DISABLE_SANDBOX) { $env:QTWEBENGINE_DISABLE_SANDBOX = "1" }
    if (-not $env:QT_OPENGL) { $env:QT_OPENGL = "software" }
    if (-not $env:QT_QUICK_BACKEND) { $env:QT_QUICK_BACKEND = "software" }

    $QtArgs = @($QtUi)
    if ($ResolvedModelDir) { $QtArgs += @("--model-dir", $ResolvedModelDir) }
    Write-Host "Launching mode=qt-web with: $PythonExe"
    & $PythonExe @QtArgs
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        Write-Warning "Qt UI failed (exit=$ExitCode). Falling back to Tk UI."
        $LaunchTk = $true
    }
}

if ($LaunchTk) {
    if (-not (Test-Path -LiteralPath $TkUi)) {
        throw "Tk UI entrypoint not found: $TkUi"
    }
    $TkArgs = @($TkUi)
    if ($ResolvedModelDir) { $TkArgs += @("--model-dir", $ResolvedModelDir) }
    Write-Host "Launching mode=legacy-tk with: $PythonExe"
    & $PythonExe @TkArgs
    $ExitCode = $LASTEXITCODE
}

if ($MyInvocation.InvocationName -eq ".") {
    $global:LASTEXITCODE = $ExitCode
    return
}
exit $ExitCode

