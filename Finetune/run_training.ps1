param(
    [string]$ModelPath = "models/Qwen3-1.7B",
    [string]$TrainFile = "Finetune/trainingdata/samples/math_policy_strict_train.jsonl",
    [string]$OutputDir = "models/tuned/1.7mathpolicy"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Resolve-ExistingPath([string]$BaseDir, [string]$PathValue) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Resolve-OutputPath([string]$BaseDir, [string]$PathValue) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    if (-not (Test-Path -LiteralPath $Candidate)) {
        New-Item -ItemType Directory -Path $Candidate -Force | Out-Null
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

$ResolvedModelPath = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $ModelPath
$ResolvedTrainFile = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $TrainFile
$ResolvedOutputDir = Resolve-OutputPath -BaseDir $ProjectRoot -PathValue $OutputDir

Set-Location -Path $ScriptDir
Write-Host "Accelerate training (model=$ResolvedModelPath, data=$ResolvedTrainFile, out=$ResolvedOutputDir)"

accelerate launch "$ScriptDir/train.py" `
  --model_name_or_path $ResolvedModelPath `
  --train_file $ResolvedTrainFile `
  --output_dir $ResolvedOutputDir `
  --max_seq_length 512 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 1e-5 `
  --num_train_epochs 4 `
  --warmup_ratio 0.05 `
  --logging_steps 10 `
  --save_steps 999999 `
  --save_only_model `
  --bf16 `
  --gradient_checkpointing
