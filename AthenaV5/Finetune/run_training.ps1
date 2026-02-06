param(
    [string]$ModelPath = "checkpoints/Qwen3-0.6B-Base",
    [string]$TrainFile = "trainingdata/samples/athena_commandments_train.jsonl",
    [string]$OutputDir = "trainingdata/output/qwen_commandment"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Resolve-PathRelative($path, $base) {
    if ([System.IO.Path]::IsPathRooted($path)) {
        return (Resolve-Path -LiteralPath $path).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $base $path)).Path
}

$ResolvedModelPath = Resolve-PathRelative $ModelPath $ProjectRoot
$ResolvedTrainFile = Resolve-PathRelative $TrainFile $ScriptDir
$ResolvedOutputDir = (Join-Path $ScriptDir $OutputDir)
if (-not (Test-Path $ResolvedOutputDir)) {
    New-Item -ItemType Directory -Path $ResolvedOutputDir -Force | Out-Null
}

Set-Location -Path $ScriptDir
Write-Host "Accelerate training (model=$ResolvedModelPath, data=$ResolvedTrainFile, out=$ResolvedOutputDir)"

accelerate launch "$ScriptDir/train.py" `
  --model_name_or_path $ResolvedModelPath `
  --train_file $ResolvedTrainFile `
  --output_dir $ResolvedOutputDir `
  --max_seq_length 2048 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 2e-5 `
  --num_train_epochs 1 `
  --warmup_ratio 0.03 `
  --logging_steps 10 `
  --save_steps 50 `
  --fp16 `
  --gradient_checkpointing
