param(
    [string]$ArgsFile = "Finetune/finetune_args.json",
    [string]$ModelPath,
    [string]$TrainFile,
    [string]$OutputDir,
    [string]$ResumeFromCheckpoint,
    [switch]$AllowCpu,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Resolve-OutputPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    if (-not (Test-Path -LiteralPath $Candidate)) {
        New-Item -ItemType Directory -Path $Candidate -Force | Out-Null
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Get-BoolValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Value
    )
    return [System.Convert]::ToBoolean($Value)
}

function Add-BoolFlag {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[string]]$Args,
        [Parameter(Mandatory = $true)]
        [string]$FlagName,
        [Parameter(Mandatory = $true)]
        [object]$Enabled
    )
    if (Get-BoolValue -Value $Enabled) {
        $Args.Add($FlagName)
    }
}

function Resolve-PythonExe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath
    )

    $Candidates = @()
    if ($env:VIRTUAL_ENV) {
        $Candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
    }
    $Candidates += (Join-Path $ProjectRootPath ".venv\Scripts\python.exe")
    $ParentDir = Split-Path -Parent $ProjectRootPath
    if ($ParentDir) {
        $Candidates += (Join-Path $ParentDir ".venv\Scripts\python.exe")
    }

    foreach ($Candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($Candidate)) { continue }
        if (Test-Path -LiteralPath $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }

    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCmd) {
        return $PythonCmd.Source
    }

    throw "Python executable not found. Activate a venv or create .venv in the project root."
}

$ResolvedArgsFile = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $ArgsFile
$Config = Get-Content -LiteralPath $ResolvedArgsFile -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not $Config.paths) { throw "Missing 'paths' section in args file: $ResolvedArgsFile" }
if (-not $Config.accelerate) { throw "Missing 'accelerate' section in args file: $ResolvedArgsFile" }
if (-not $Config.train) { throw "Missing 'train' section in args file: $ResolvedArgsFile" }

$SelectedModelPath = if ($PSBoundParameters.ContainsKey("ModelPath")) { $ModelPath } else { [string]$Config.paths.model_path }
$SelectedTrainFile = if ($PSBoundParameters.ContainsKey("TrainFile")) { $TrainFile } else { [string]$Config.paths.train_file }
$SelectedOutputDir = if ($PSBoundParameters.ContainsKey("OutputDir")) { $OutputDir } else { [string]$Config.paths.output_dir }
$ConfigResumeCheckpoint = ""
if ($Config.paths.PSObject.Properties.Name -contains "resume_from_checkpoint") {
    $ConfigResumeCheckpoint = [string]$Config.paths.resume_from_checkpoint
}
$SelectedResumeCheckpoint = if ($PSBoundParameters.ContainsKey("ResumeFromCheckpoint")) { $ResumeFromCheckpoint } else { $ConfigResumeCheckpoint }

if ([string]::IsNullOrWhiteSpace($SelectedModelPath)) { throw "model_path is required." }
if ([string]::IsNullOrWhiteSpace($SelectedTrainFile)) { throw "train_file is required." }
if ([string]::IsNullOrWhiteSpace($SelectedOutputDir)) { throw "output_dir is required." }

$ResolvedModelPath = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $SelectedModelPath
$ResolvedTrainFile = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $SelectedTrainFile
$ResolvedOutputDir = Resolve-OutputPath -BaseDir $ProjectRoot -PathValue $SelectedOutputDir
$ResolvedResumeCheckpoint = $null
if (-not [string]::IsNullOrWhiteSpace($SelectedResumeCheckpoint)) {
    $ResolvedResumeCheckpoint = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $SelectedResumeCheckpoint
}

if ((Get-BoolValue -Value $Config.train.bf16) -and (Get-BoolValue -Value $Config.train.fp16)) {
    throw "Invalid config: both train.bf16 and train.fp16 are true. Enable only one."
}

$Summary = [ordered]@{
    args_file = $ResolvedArgsFile
    model = $ResolvedModelPath
    train_file = $ResolvedTrainFile
    output_dir = $ResolvedOutputDir
    accelerate = [ordered]@{
        num_processes = $Config.accelerate.num_processes
        num_machines = $Config.accelerate.num_machines
        mixed_precision = $Config.accelerate.mixed_precision
        dynamo_backend = $Config.accelerate.dynamo_backend
    }
    train = [ordered]@{
        max_seq_length = $Config.train.max_seq_length
        expected_samples = $Config.train.expected_samples
        strict_no_truncation = $Config.train.strict_no_truncation
        per_device_train_batch_size = $Config.train.per_device_train_batch_size
        gradient_accumulation_steps = $Config.train.gradient_accumulation_steps
        learning_rate = $Config.train.learning_rate
        num_train_epochs = $Config.train.num_train_epochs
        warmup_ratio = $Config.train.warmup_ratio
        lr_scheduler_type = $Config.train.lr_scheduler_type
        weight_decay = $Config.train.weight_decay
        max_grad_norm = $Config.train.max_grad_norm
        logging_steps = $Config.train.logging_steps
        save_steps = $Config.train.save_steps
        save_total_limit = $Config.train.save_total_limit
        save_only_model = $Config.train.save_only_model
        bf16 = $Config.train.bf16
        fp16 = $Config.train.fp16
        gradient_checkpointing = $Config.train.gradient_checkpointing
        seed = $Config.train.seed
    }
}
if ($ResolvedResumeCheckpoint) {
    $Summary.resume_from_checkpoint = $ResolvedResumeCheckpoint
}

Set-Location -Path $ScriptDir
Write-Host "Finetune config:"
Write-Host ($Summary | ConvertTo-Json -Depth 5)

$PythonExe = Resolve-PythonExe -ProjectRootPath $ProjectRoot
Write-Host "python_exe=$PythonExe"

& $PythonExe -c "import accelerate" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Missing Python package: accelerate. Install with: `"$PythonExe`" -m pip install accelerate"
}

$TorchProbe = & $PythonExe -c "import json,sys,torch; print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'cuda_version':(torch.version.cuda or ''), 'device_count':int(torch.cuda.device_count())}))" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to probe torch runtime using $PythonExe"
}

try {
    $TorchInfo = $TorchProbe | ConvertFrom-Json
} catch {
    throw "Unable to parse torch probe output: $TorchProbe"
}

Write-Host ("torch_runtime=" + ($TorchInfo | ConvertTo-Json -Compress))
if (-not $AllowCpu -and -not [bool]$TorchInfo.cuda_available) {
    throw (
        "CUDA is not available for selected python runtime. " +
        "This will make training extremely slow on CPU. " +
        "Use the venv CUDA runtime (D:\AthenaPlayground\.venv\Scripts\python.exe), " +
        "or re-run with -AllowCpu if CPU training is intentional."
    )
}

$LaunchArgs = [System.Collections.Generic.List[string]]::new()
$LaunchArgs.Add("-m")
$LaunchArgs.Add("accelerate.commands.launch")
$LaunchArgs.Add("--num_processes"); $LaunchArgs.Add([string]$Config.accelerate.num_processes)
$LaunchArgs.Add("--num_machines"); $LaunchArgs.Add([string]$Config.accelerate.num_machines)
$LaunchArgs.Add("--mixed_precision"); $LaunchArgs.Add([string]$Config.accelerate.mixed_precision)
$LaunchArgs.Add("--dynamo_backend"); $LaunchArgs.Add([string]$Config.accelerate.dynamo_backend)
$LaunchArgs.Add((Join-Path $ScriptDir "train.py"))

$LaunchArgs.Add("--model_name_or_path"); $LaunchArgs.Add($ResolvedModelPath)
$LaunchArgs.Add("--train_file"); $LaunchArgs.Add($ResolvedTrainFile)
$LaunchArgs.Add("--output_dir"); $LaunchArgs.Add($ResolvedOutputDir)

$LaunchArgs.Add("--max_seq_length"); $LaunchArgs.Add([string]$Config.train.max_seq_length)
$LaunchArgs.Add("--expected_samples"); $LaunchArgs.Add([string]$Config.train.expected_samples)
$LaunchArgs.Add("--per_device_train_batch_size"); $LaunchArgs.Add([string]$Config.train.per_device_train_batch_size)
$LaunchArgs.Add("--gradient_accumulation_steps"); $LaunchArgs.Add([string]$Config.train.gradient_accumulation_steps)
$LaunchArgs.Add("--learning_rate"); $LaunchArgs.Add([string]$Config.train.learning_rate)
$LaunchArgs.Add("--num_train_epochs"); $LaunchArgs.Add([string]$Config.train.num_train_epochs)
$LaunchArgs.Add("--warmup_ratio"); $LaunchArgs.Add([string]$Config.train.warmup_ratio)
$LaunchArgs.Add("--lr_scheduler_type"); $LaunchArgs.Add([string]$Config.train.lr_scheduler_type)
$LaunchArgs.Add("--weight_decay"); $LaunchArgs.Add([string]$Config.train.weight_decay)
$LaunchArgs.Add("--max_grad_norm"); $LaunchArgs.Add([string]$Config.train.max_grad_norm)
$LaunchArgs.Add("--logging_steps"); $LaunchArgs.Add([string]$Config.train.logging_steps)
$LaunchArgs.Add("--save_steps"); $LaunchArgs.Add([string]$Config.train.save_steps)
$LaunchArgs.Add("--save_total_limit"); $LaunchArgs.Add([string]$Config.train.save_total_limit)
$LaunchArgs.Add("--seed"); $LaunchArgs.Add([string]$Config.train.seed)
if ($ResolvedResumeCheckpoint) {
    $LaunchArgs.Add("--resume_from_checkpoint"); $LaunchArgs.Add($ResolvedResumeCheckpoint)
}

Add-BoolFlag -Args $LaunchArgs -FlagName "--save_only_model" -Enabled $Config.train.save_only_model
Add-BoolFlag -Args $LaunchArgs -FlagName "--strict_no_truncation" -Enabled $Config.train.strict_no_truncation
Add-BoolFlag -Args $LaunchArgs -FlagName "--bf16" -Enabled $Config.train.bf16
Add-BoolFlag -Args $LaunchArgs -FlagName "--fp16" -Enabled $Config.train.fp16
Add-BoolFlag -Args $LaunchArgs -FlagName "--gradient_checkpointing" -Enabled $Config.train.gradient_checkpointing

if ($DryRun) {
    $Quoted = $LaunchArgs | ForEach-Object {
        if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
    }
    Write-Host "Dry run only. Command:"
    Write-Host ($PythonExe + " " + ($Quoted -join " "))
    return
}

& $PythonExe @LaunchArgs
