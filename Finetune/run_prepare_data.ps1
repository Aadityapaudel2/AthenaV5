param(
    [string]$InputFile = "Finetune/trainingdata/bhagavadgita/bhagavaggitatrainingdata.jsonl",
    [string]$OutputFile = "Finetune/trainingdata/bhagavadgita/bhagavaggitatrainingdata_train.jsonl",
    [ValidateSet("teacher", "student")]
    [string]$AssistantRole = "teacher",
    [ValidateSet("assistant_turn", "dialogue")]
    [string]$ArtifactStyle = "assistant_turn",
    [int]$MaxContextMessages = 12,
    [int]$MinMessages = 2,
    [switch]$NoDropEmpty,
    [switch]$NoRequireUserBeforeAssistant,
    [switch]$NoMergeConsecutiveSameRole,
    [switch]$NoStripRolePrefixes
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Resolve-ExistingPath([string]$BaseDir, [string]$PathValue) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Resolve-OutputFilePath([string]$BaseDir, [string]$PathValue) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue } else { Join-Path $BaseDir $PathValue }
    $Parent = Split-Path -Parent $Candidate
    if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }
    return $Candidate
}

$ResolvedInput = Resolve-ExistingPath -BaseDir $ProjectRoot -PathValue $InputFile
$ResolvedOutput = Resolve-OutputFilePath -BaseDir $ProjectRoot -PathValue $OutputFile

Set-Location -Path $ProjectRoot
Write-Host "Preparing data (input=$ResolvedInput, output=$ResolvedOutput, assistant_role=$AssistantRole, style=$ArtifactStyle)"

$PyArgs = @(
    "$ScriptDir/prepare_data.py",
    "--input", $ResolvedInput,
    "--output", $ResolvedOutput,
    "--assistant_role", $AssistantRole,
    "--artifact_style", $ArtifactStyle,
    "--max_context_messages", "$MaxContextMessages",
    "--min_messages", "$MinMessages"
)

if (-not $NoDropEmpty) { $PyArgs += "--drop_empty" }
if (-not $NoRequireUserBeforeAssistant) { $PyArgs += "--require_user_before_assistant" }
if (-not $NoMergeConsecutiveSameRole) { $PyArgs += "--merge_consecutive_same_role" }
if (-not $NoStripRolePrefixes) { $PyArgs += "--strip_role_prefixes" }

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $PythonExe) {
    & $PythonExe @PyArgs
} else {
    python @PyArgs
}
