param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$Preregistration = "configs/preregistrations/ldt_array_confirmation_v3.yaml",
  [string]$RunId = "ldt-array-confirmation-v3-seed-sweep",
  [string]$Distro = "Ubuntu",
  [string]$CandidateId = "",
  [int]$Seed = -1
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifest = (Resolve-Path $DatasetManifest).Path
$prereg = (Resolve-Path (Join-Path $repo $Preregistration)).Path
$out = [IO.Path]::GetFullPath($OutDir)
$repoPrefix = $repo.TrimEnd('\') + '\'
if ($out.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Generated LDT confirmation runs must stay outside the DataGoblin repository"
}
if ($RunId -notmatch '^[a-zA-Z0-9_.-]+$') { throw "RunId contains unsupported characters" }
if ($CandidateId -and $CandidateId -notmatch '^[a-zA-Z0-9_.-]+$') {
  throw "CandidateId contains unsupported characters"
}
if (-not $CandidateId -and $Seed -ge 0) {
  throw "Seed requires CandidateId"
}

function Convert-ToWslPath([string]$WindowsPath) {
  $full = [IO.Path]::GetFullPath($WindowsPath)
  if ($full -notmatch '^([a-zA-Z]):\\(.*)$') { throw "unsupported Windows path: $full" }
  $drive = $Matches[1].ToLowerInvariant()
  $tail = $Matches[2].Replace('\', '/')
  return "/mnt/$drive/$tail"
}

$trainer = Join-Path $repo "scripts\train_ldt_array_confirmation.py"
$prepare = Join-Path $repo "scripts\prepare_ldt_array_confirmation_run.py"
$wrapper = Join-Path $repo "scripts\run_training_gym_capped_wsl.sh"
$cleanup = Join-Path $repo "scripts\post_run_hrm_cleanup.ps1"
$source = Join-Path $repo "src\datagoblin"
foreach ($required in @($trainer, $prepare, $wrapper, $cleanup, $prereg, $manifest)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "missing required file: $required" }
}

New-Item -ItemType Directory -Path $out -Force | Out-Null
$planPath = Join-Path $out "trainer_plan.json"
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = Join-Path $repo "src"
  & python $prepare --preregistration $prereg --dataset-manifest $manifest --out $planPath
  if ($LASTEXITCODE -ne 0) { throw "could not prepare the frozen v3 trainer plan" }
} finally {
  $env:PYTHONPATH = $previousPythonPath
}

$cappedOut = Join-Path $out "capped_run"
New-Item -ItemType Directory -Path $cappedOut -Force | Out-Null
$datasetDirWsl = Convert-ToWslPath (Split-Path -Parent $manifest)
$trainerWsl = Convert-ToWslPath $trainer
$wrapperWsl = Convert-ToWslPath $wrapper
$sourceWsl = Convert-ToWslPath $source
$planWsl = Convert-ToWslPath $planPath
$outWsl = Convert-ToWslPath $cappedOut
$stage = "/var/tmp/datagoblin-$RunId"
if ($stage -notmatch '^/var/tmp/datagoblin-[a-zA-Z0-9_.-]+$') { throw "unsafe WSL stage path" }

$status = 1
$stageCleanupStatus = 91
try {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -ne 0 -or $resolvedStage -ne $stage) { throw "WSL stage path verification failed" }
  & wsl.exe -d $Distro -u root -- rm -rf -- $stage
  if ($LASTEXITCODE -ne 0) { throw "could not clear the verified WSL stage" }
  & wsl.exe -d $Distro -u root -- mkdir -p "$stage/dataset" "$stage/src" "$stage/out"
  if ($LASTEXITCODE -ne 0) { throw "could not create the WSL stage" }
  & wsl.exe -d $Distro -u root -- cp -a "$datasetDirWsl/." "$stage/dataset/"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the public v3 dataset" }
  & wsl.exe -d $Distro -u root -- cp -a $sourceWsl "$stage/src/datagoblin"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the DataGoblin package" }
  & wsl.exe -d $Distro -u root -- cp $trainerWsl "$stage/train_ldt_array_confirmation.py"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the v3 LDT trainer" }
  & wsl.exe -d $Distro -u root -- cp $planWsl "$stage/trainer_plan.json"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the public v3 trainer plan" }
  $runUserLine = (& wsl.exe -d $Distro -u root -- getent passwd 1000 | Select-Object -Last 1)
  if ($LASTEXITCODE -ne 0 -or -not $runUserLine) { throw "WSL has no unprivileged UID 1000 run user" }
  $runUser = $runUserLine.Trim().Split(':')[0]
  & wsl.exe -d $Distro -u root -- chown -R "${runUser}:${runUser}" $stage
  if ($LASTEXITCODE -ne 0) { throw "could not assign the WSL stage to the run user" }
  $shardArgs = @()
  if ($CandidateId) {
    $shardArgs = @("--candidate-id", $CandidateId)
    if ($Seed -ge 0) {
      $shardArgs += @("--seed", [string]$Seed)
    }
  }
  & wsl.exe -d $Distro -u root -- bash $wrapperWsl $RunId /usr/bin/python3 $stage $stage "$stage/out" -- `
    /usr/bin/env PYTHONPATH="$stage/src" /usr/bin/python3 "$stage/train_ldt_array_confirmation.py" `
    --dataset-manifest "$stage/dataset/dataset_manifest.json" `
    --trainer-plan "$stage/trainer_plan.json" `
    --out "$stage/out" @shardArgs
  $status = $LASTEXITCODE
  & wsl.exe -d $Distro -u root -- cp -a "$stage/out/." "$outWsl/"
  if ($LASTEXITCODE -ne 0) { throw "could not copy capped v3 artifacts" }
} finally {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -eq 0 -and $resolvedStage -eq $stage -and $resolvedStage.StartsWith('/var/tmp/datagoblin-')) {
    & wsl.exe -d $Distro -u root -- rm -rf -- $resolvedStage
    $stageCleanupStatus = $LASTEXITCODE
  }
  & $cleanup -RunId $RunId -SummaryPath (Join-Path $out "cleanup_summary.json") | Out-Null
}

$wrapperSummaryPath = Join-Path $cappedOut "summary.json"
$trainerSummaryPath = Join-Path $cappedOut "trainer_summary.json"
$cleanupSummaryPath = Join-Path $out "cleanup_summary.json"
$receipt = [ordered]@{
  schema_version = "ldt-array-confirmation-run-receipt.v3"
  run_id = $RunId
  capped_process_exit_code = $status
  wsl_stage_cleanup_exit_code = $stageCleanupStatus
  shared_wsl_runtime_stopped = $false
  trainer_plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
  caps_receipt = if (Test-Path $wrapperSummaryPath) { Get-Content $wrapperSummaryPath -Raw | ConvertFrom-Json } else { $null }
  trainer_receipt = if (Test-Path $trainerSummaryPath) { Get-Content $trainerSummaryPath -Raw | ConvertFrom-Json } else { $null }
  cleanup_receipt = if (Test-Path $cleanupSummaryPath) { Get-Content $cleanupSummaryPath -Raw | ConvertFrom-Json } else { $null }
  heldout_pair_labels_staged = $false
  heldout_array_labels_staged = $false
  external_checkpoint_loaded = $false
  artifacts = [ordered]@{}
  promotion_eligible = $false
  next_gate = "one independent steward evaluation of pair, calibration, and array targets"
}
foreach ($name in @("events.jsonl", "summary.json", "trainer_events.jsonl", "trainer_summary.json", "heldout_pair_predictions.jsonl", "heldout_array_predictions.jsonl", "unit_properties.txt")) {
  $path = Join-Path $cappedOut $name
  if (Test-Path -LiteralPath $path -PathType Leaf) {
    $receipt.artifacts[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $out "run_receipt.json") -Encoding UTF8
if ($status -ne 0) { throw "capped v3 LDT trainer aborted with exit code $status; abort artifacts were preserved" }
Write-Output (Join-Path $out "run_receipt.json")
