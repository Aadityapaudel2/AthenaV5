param(
    [string]$RunDir = "models/tuned/AthenaV7.03_recover",
    [int]$PollSeconds = 5,
    [int]$TrainPid = 0,
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
if (-not [System.IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $ProjectRoot $RunDir
}

function Get-DirBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return [int64]0 }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if (-not $sum) { return [int64]0 }
    return [int64]$sum
}

function Get-LatestCheckpointInfo {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) { return $null }

    $ckpt = Get-ChildItem -LiteralPath $Dir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "checkpoint-*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $ckpt) { return $null }

    $step = $null
    $statePath = Join-Path $ckpt.FullName "trainer_state.json"
    if (Test-Path -LiteralPath $statePath) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $step = $state.global_step
        } catch { }
    }

    return [pscustomobject]@{
        Name = $ckpt.Name
        Step = $step
        LastWriteTime = $ckpt.LastWriteTime
    }
}

function Get-TrainProcess {
    param([int]$ProcessIdHint)

    if ($ProcessIdHint -gt 0) {
        return Get-Process -Id $ProcessIdHint -ErrorAction SilentlyContinue
    }

    $allPy = @(Get-Process -Name python -ErrorAction SilentlyContinue)
    if ($allPy.Count -gt 0) {
        $best = $allPy[0]
        foreach ($proc in $allPy) {
            $cProc = [double]($proc.CPU)
            $cBest = [double]($best.CPU)
            if ($cProc -gt $cBest) { $best = $proc }
        }
        if ($best) { return $best }
    }

    # WDDM fallback: infer Python PID from nvidia-smi process table.
    $raw = & nvidia-smi 2>$null
    if ($raw) {
        foreach ($line in $raw) {
            if ($line -match "^\|\s*0\s+N/A\s+N/A\s+(\d+)\s+.*python\.exe") {
                $gpuPid = [int]$Matches[1]
                $p = Get-Process -Id $gpuPid -ErrorAction SilentlyContinue
                if ($p) { return $p }
            }
        }
    }

    return $null
}

function Get-GpuSnapshot {
    $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null |
        Select-Object -First 1
    if (-not $line) { return $null }

    $parts = $line -split "," | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 4) { return $null }

    return [pscustomobject]@{
        Util = [double]$parts[0]
        MemUsed = [double]$parts[1]
        MemTotal = [double]$parts[2]
        Power = [double]$parts[3]
    }
}

if ($PollSeconds -lt 1) { $PollSeconds = 1 }

Write-Host "Watching training signals"
Write-Host "RunDir: $RunDir"
Write-Host "PollSeconds: $PollSeconds"
Write-Host "Tip: For most reliable CPU tracking, pass -TrainPid <pid>."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$prevPid = $null
$prevCpu = $null
$prevDirBytes = Get-DirBytes -Path $RunDir

while ($true) {
    $now = Get-Date
    $proc = Get-TrainProcess -ProcessIdHint $TrainPid
    $gpu = Get-GpuSnapshot
    $ckpt = Get-LatestCheckpointInfo -Dir $RunDir
    $dirBytes = Get-DirBytes -Path $RunDir

    if (-not $proc) {
        $cpuDelta = 0.0
        $pidText = "none"
        $cpuText = "0.00"
        $pyList = (Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id) -join ","
        if (-not $pyList) { $pyList = "none" }
        $pidText = "none (python_pids=$pyList)"
    }
    else {
        if ($prevPid -ne $proc.Id -or $null -eq $prevCpu) {
            $prevCpu = $proc.CPU
            $prevPid = $proc.Id
        }
        $cpuDelta = [math]::Round(($proc.CPU - $prevCpu), 2)
        $pidText = [string]$proc.Id
        $cpuText = [math]::Round($proc.CPU, 2).ToString("F2")
        $prevCpu = $proc.CPU
    }

    $dirDeltaMB = [math]::Round(($dirBytes - $prevDirBytes) / 1MB, 2)
    $dirMB = [math]::Round($dirBytes / 1MB, 2)
    $prevDirBytes = $dirBytes

    $gpuText = "n/a"
    if ($gpu) {
        $gpuText = ("{0}% {1}/{2}MiB {3}W" -f $gpu.Util, $gpu.MemUsed, $gpu.MemTotal, $gpu.Power)
    }

    $ckptText = "none"
    if ($ckpt) {
        $ageSec = [math]::Round((New-TimeSpan -Start $ckpt.LastWriteTime -End $now).TotalSeconds, 0)
        $ckptText = ("{0} step={1} age={2}s" -f $ckpt.Name, $ckpt.Step, $ageSec)
    }

    $active = $false
    if ($cpuDelta -gt 0.05) { $active = $true }
    if ($gpu -and $gpu.Util -ge 5) { $active = $true }
    $activeText = if ($active) { "YES" } else { "NO" }

    Write-Host ("[{0}] pid={1} cpu_total={2}s cpu_delta={3}s active={4} | gpu={5} | run_dir={6}MB delta={7}MB | {8}" -f `
        $now.ToString("yyyy-MM-dd HH:mm:ss"), $pidText, $cpuText, $cpuDelta, $activeText, $gpuText, $dirMB, $dirDeltaMB, $ckptText)

    if ($Once) { break }
    Start-Sleep -Seconds $PollSeconds
}
