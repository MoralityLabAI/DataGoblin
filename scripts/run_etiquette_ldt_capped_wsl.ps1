param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$RunId = "ldt-etiquette-sorter-v1-smoke",
  [string]$Distro = "Ubuntu",
  [int]$Epochs = 40,
  [int]$BatchSize = 14,
  [int]$CheckpointEvery = 100,
  [int]$Seed = 7
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifest = (Resolve-Path $DatasetManifest).Path
$datasetDir = Split-Path -Parent $manifest
$out = [IO.Path]::GetFullPath($OutDir)
$repoPrefix = $repo.TrimEnd('\') + '\'
if (-not $out.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "OutDir must stay inside the DataGoblin workspace"
}
if ($BatchSize -lt 1 -or $BatchSize -gt 32) { throw "BatchSize must be between 1 and 32" }
if ($CheckpointEvery -lt 1) { throw "CheckpointEvery must be positive" }
if ($RunId -notmatch '^[a-zA-Z0-9_.-]+$') { throw "RunId contains unsupported characters" }

function Convert-ToWslPath([string]$WindowsPath) {
  $full = [IO.Path]::GetFullPath($WindowsPath)
  if ($full -notmatch '^([a-zA-Z]):\\(.*)$') { throw "unsupported Windows path: $full" }
  $drive = $Matches[1].ToLowerInvariant()
  $tail = $Matches[2].Replace('\', '/')
  return "/mnt/$drive/$tail"
}

$trainer = Join-Path $repo "scripts\train_etiquette_ldt.py"
$wrapper = Join-Path $repo "scripts\run_training_gym_capped_wsl.sh"
$cleanup = Join-Path $repo "scripts\post_run_hrm_cleanup.ps1"
foreach ($required in @($trainer, $wrapper, $cleanup)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "missing required file: $required" }
}

New-Item -ItemType Directory -Path $out -Force | Out-Null
$cappedOut = Join-Path $out "capped_run"
New-Item -ItemType Directory -Path $cappedOut -Force | Out-Null
$plan = [ordered]@{
  schema_version = "1.0"
  training_task_id = $RunId
  operator_approval = "user instruction: do"
  architecture = "learned public-envelope proposal head plus explicit LDT policy membrane"
  authority = "MODEL_RECOMMENDATION_ONLY"
  caps = [ordered]@{ ram_mb = 2048; cpu_percent = 50; io_mb_per_second = 50; wall_clock_seconds = 900; swap_bytes = 0 }
  chunk_strategy = "at most 32 public envelopes per batch"
  batch_size = $BatchSize
  checkpoint_interval = [ordered]@{ steps = $CheckpointEvery; seconds = 300 }
  epochs = $Epochs
  seed = $Seed
  wrapper = "scripts/run_training_gym_capped_wsl.sh"
  network = $false
  external_checkpoint = $null
  raw_content_permitted = $false
  sealed_labels_permitted = $false
  larger_model_training_permitted = $false
}
$planPath = Join-Path $out "trainer_plan.json"
$plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $planPath -Encoding UTF8

$datasetWsl = Convert-ToWslPath $datasetDir
$trainerWsl = Convert-ToWslPath $trainer
$wrapperWsl = Convert-ToWslPath $wrapper
$outWsl = Convert-ToWslPath $cappedOut
$stage = "/var/tmp/datagoblin-$RunId"
if ($stage -notmatch '^/var/tmp/datagoblin-[a-zA-Z0-9_.-]+$') { throw "unsafe WSL stage path" }

$status = 1
try {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -ne 0 -or $resolvedStage -ne $stage) { throw "WSL stage path verification failed" }
  & wsl.exe -d $Distro -u root -- rm -rf -- $stage
  if ($LASTEXITCODE -ne 0) { throw "could not clear the verified WSL stage" }
  & wsl.exe -d $Distro -u root -- mkdir -p "$stage/dataset" "$stage/out"
  if ($LASTEXITCODE -ne 0) { throw "could not create WSL stage" }
  & wsl.exe -d $Distro -u root -- cp -a "$datasetWsl/." "$stage/dataset/"
  if ($LASTEXITCODE -ne 0) { throw "could not stage public dataset" }
  & wsl.exe -d $Distro -u root -- cp $trainerWsl "$stage/train_etiquette_ldt.py"
  if ($LASTEXITCODE -ne 0) { throw "could not stage trainer" }
  & wsl.exe -d $Distro -u root -- bash $wrapperWsl $RunId /usr/bin/python3 $stage $stage "$stage/out" -- `
    /usr/bin/python3 "$stage/train_etiquette_ldt.py" `
    --dataset-manifest "$stage/dataset/dataset_manifest.json" `
    --out "$stage/out" `
    --run-id $RunId `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --checkpoint-every $CheckpointEvery `
    --seed $Seed
  $status = $LASTEXITCODE
  & wsl.exe -d $Distro -u root -- cp -a "$stage/out/." "$outWsl/"
  if ($LASTEXITCODE -ne 0) { throw "could not copy capped run artifacts" }
} finally {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -eq 0 -and $resolvedStage -eq $stage -and $resolvedStage.StartsWith('/var/tmp/datagoblin-')) {
    & wsl.exe -d $Distro -u root -- rm -rf -- $resolvedStage
    $stageCleanupStatus = $LASTEXITCODE
  } else {
    $stageCleanupStatus = 91
  }
  & $cleanup -RunId $RunId -SummaryPath (Join-Path $out "cleanup_summary.json") | Out-Null
}

$trainerSummaryPath = Join-Path $cappedOut "trainer_summary.json"
$wrapperSummaryPath = Join-Path $cappedOut "summary.json"
$cleanupSummaryPath = Join-Path $out "cleanup_summary.json"
$receipt = [ordered]@{
  schema_version = "1.0"
  run_id = $RunId
  capped_process_exit_code = $status
  wsl_stage_cleanup_exit_code = $stageCleanupStatus
  shared_wsl_runtime_stopped = $false
  caps_receipt = if (Test-Path $wrapperSummaryPath) { Get-Content $wrapperSummaryPath -Raw | ConvertFrom-Json } else { $null }
  trainer_receipt = if (Test-Path $trainerSummaryPath) { Get-Content $trainerSummaryPath -Raw | ConvertFrom-Json } else { $null }
  cleanup_receipt = if (Test-Path $cleanupSummaryPath) { Get-Content $cleanupSummaryPath -Raw | ConvertFrom-Json } else { $null }
  artifacts = [ordered]@{}
  promotion_eligible = $false
  next_gate = "independent sealed evaluation and steward signoff"
}
foreach ($name in @("events.jsonl", "summary.json", "trainer_summary.json", "model.json", "predictions.jsonl", "unit_properties.txt")) {
  $path = Join-Path $cappedOut $name
  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $receipt.artifacts[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $out "run_receipt.json") -Encoding UTF8
if ($status -ne 0) { throw "capped LDT trainer aborted with exit code $status; abort artifacts were preserved" }
Write-Output (Join-Path $out "run_receipt.json")
