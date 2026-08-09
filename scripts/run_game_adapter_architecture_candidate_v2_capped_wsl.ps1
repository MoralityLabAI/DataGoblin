param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$ReleaseManifest,
  [Parameter(Mandatory = $true)][string]$CandidateId,
  [Parameter(Mandatory = $true)][int]$Seed,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$Config = "configs/preregistrations/game_adapter_architecture_tournament_v2.yaml",
  [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataset = (Resolve-Path $DatasetManifest).Path
$release = (Resolve-Path $ReleaseManifest).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
$out = [IO.Path]::GetFullPath($OutDir)
$repoPrefix = $repo.TrimEnd('\') + '\'
if ($out.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Generated architecture candidate runs must stay outside the DataGoblin repository"
}
if ($CandidateId -notmatch '^[a-zA-Z0-9_.-]+$') { throw "CandidateId contains unsupported characters" }
$runId = "architecture-v2-$CandidateId-seed-$Seed"
if ($runId -notmatch '^[a-zA-Z0-9_.-]+$') { throw "RunId contains unsupported characters" }
if (Test-Path -LiteralPath $out) { throw "Architecture candidate output already exists: $out" }

function Convert-ToWslPath([string]$WindowsPath) {
  $full = [IO.Path]::GetFullPath($WindowsPath)
  if ($full -notmatch '^([a-zA-Z]):\\(.*)$') { throw "unsupported Windows path: $full" }
  $drive = $Matches[1].ToLowerInvariant()
  $tail = $Matches[2].Replace('\', '/')
  return "/mnt/$drive/$tail"
}

$driver = Join-Path $repo "scripts\game_adapter_architecture_tournament_v2.py"
$wrapper = Join-Path $repo "scripts\run_training_gym_capped_wsl.sh"
$cleanup = Join-Path $repo "scripts\post_run_hrm_cleanup.ps1"
$source = Join-Path $repo "src\datagoblin"
$benchmark = Join-Path $repo "configs\preregistrations\game_adapter_schema_drift_v1.yaml"
$policy = Join-Path $repo "configs\curation_adapter_verifier.yaml"
foreach ($required in @($driver, $wrapper, $cleanup, $source, $benchmark, $policy, $dataset, $release, $configPath)) {
  if (-not (Test-Path -LiteralPath $required)) { throw "missing required architecture file: $required" }
}

New-Item -ItemType Directory -Path $out | Out-Null
$cappedOut = Join-Path $out "capped_run"
New-Item -ItemType Directory -Path $cappedOut | Out-Null
$datasetRootWsl = Convert-ToWslPath (Split-Path -Parent $dataset)
$releaseRootWsl = Convert-ToWslPath (Split-Path -Parent $release)
$sourceWsl = Convert-ToWslPath $source
$driverWsl = Convert-ToWslPath $driver
$wrapperWsl = Convert-ToWslPath $wrapper
$configWsl = Convert-ToWslPath $configPath
$benchmarkWsl = Convert-ToWslPath $benchmark
$policyWsl = Convert-ToWslPath $policy
$outWsl = Convert-ToWslPath $cappedOut
$stage = "/var/tmp/datagoblin-$runId"
if ($stage -notmatch '^/var/tmp/datagoblin-[a-zA-Z0-9_.-]+$') { throw "unsafe WSL stage path" }

$status = 1
$stageCleanupStatus = 91
try {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -ne 0 -or $resolvedStage -ne $stage) { throw "WSL stage path verification failed" }
  & wsl.exe -d $Distro -u root -- rm -rf -- $stage
  if ($LASTEXITCODE -ne 0) { throw "could not clear the verified WSL stage" }
  & wsl.exe -d $Distro -u root -- mkdir -p `
    "$stage/repo/src" `
    "$stage/repo/scripts" `
    "$stage/repo/configs/preregistrations" `
    "$stage/public" `
    "$stage/release" `
    "$stage/out"
  if ($LASTEXITCODE -ne 0) { throw "could not create the architecture WSL stage" }
  & wsl.exe -d $Distro -u root -- cp -a $sourceWsl "$stage/repo/src/datagoblin"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the DataGoblin package" }
  & wsl.exe -d $Distro -u root -- cp $driverWsl "$stage/repo/scripts/game_adapter_architecture_tournament_v2.py"
  & wsl.exe -d $Distro -u root -- cp $configWsl "$stage/repo/configs/preregistrations/game_adapter_architecture_tournament_v2.yaml"
  & wsl.exe -d $Distro -u root -- cp $benchmarkWsl "$stage/repo/configs/preregistrations/game_adapter_schema_drift_v1.yaml"
  & wsl.exe -d $Distro -u root -- cp $policyWsl "$stage/repo/configs/curation_adapter_verifier.yaml"
  if ($LASTEXITCODE -ne 0) { throw "could not stage architecture configs" }
  & wsl.exe -d $Distro -u root -- cp -a "$datasetRootWsl/." "$stage/public/"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the public architecture dataset" }
  & wsl.exe -d $Distro -u root -- cp -a "$releaseRootWsl/." "$stage/release/"
  if ($LASTEXITCODE -ne 0) { throw "could not stage the development-only training release" }
  $runUserLine = (& wsl.exe -d $Distro -u root -- getent passwd 1000 | Select-Object -Last 1)
  if ($LASTEXITCODE -ne 0 -or -not $runUserLine) { throw "WSL has no unprivileged UID 1000 run user" }
  $runUser = $runUserLine.Trim().Split(':')[0]
  & wsl.exe -d $Distro -u root -- chown -R "${runUser}:${runUser}" $stage
  if ($LASTEXITCODE -ne 0) { throw "could not assign the architecture WSL stage" }
  & wsl.exe -d $Distro -u root -- bash $wrapperWsl $runId /usr/bin/python3 "$stage/repo" $stage "$stage/out" -- `
    /usr/bin/env PYTHONPATH="$stage/repo/src" /usr/bin/python3 `
    "$stage/repo/scripts/game_adapter_architecture_tournament_v2.py" train `
    --config "$stage/repo/configs/preregistrations/game_adapter_architecture_tournament_v2.yaml" `
    --dataset-manifest "$stage/public/tournament_manifest.json" `
    --release-manifest "$stage/release/release_manifest.json" `
    --candidate-id $CandidateId `
    --seed $Seed `
    --out "$stage/out"
  $status = $LASTEXITCODE
  & wsl.exe -d $Distro -u root -- cp -a "$stage/out/." "$outWsl/"
  if ($LASTEXITCODE -ne 0) { throw "could not copy capped architecture artifacts" }
} finally {
  $resolvedStage = (& wsl.exe -d $Distro -u root -- realpath -m $stage | Select-Object -Last 1).Trim()
  if ($LASTEXITCODE -eq 0 -and $resolvedStage -eq $stage -and $resolvedStage.StartsWith('/var/tmp/datagoblin-')) {
    & wsl.exe -d $Distro -u root -- rm -rf -- $resolvedStage
    $stageCleanupStatus = $LASTEXITCODE
  }
  & $cleanup -RunId $runId -SummaryPath (Join-Path $out "cleanup_summary.json") | Out-Null
}

$wrapperSummaryPath = Join-Path $cappedOut "summary.json"
$trainerSummaryPath = Join-Path $cappedOut "trainer_summary.json"
$candidateManifestPath = Join-Path $cappedOut "candidate_manifest.json"
$cleanupSummaryPath = Join-Path $out "cleanup_summary.json"
if (-not (Test-Path -LiteralPath $candidateManifestPath -PathType Leaf)) {
  throw "capped architecture candidate did not emit a candidate manifest"
}
$wrapperSummary = Get-Content $wrapperSummaryPath -Raw | ConvertFrom-Json
$trainerSummary = Get-Content $trainerSummaryPath -Raw | ConvertFrom-Json
$cleanupSummary = Get-Content $cleanupSummaryPath -Raw | ConvertFrom-Json
$receipt = [ordered]@{
  schema_version = "game-adapter-architecture-run-receipt.v2"
  run_id = $runId
  candidate_id = $CandidateId
  seed = $Seed
  capped_process_exit_code = $status
  wsl_stage_cleanup_exit_code = $stageCleanupStatus
  caps_enforced = [bool]$wrapperSummary.caps_enforced
  cleanup_passed = [bool]$cleanupSummary.cleanup_passed
  steps_completed = [int]$trainerSummary.steps_completed
  peak_ram_mb = [double]$wrapperSummary.peak_ram_mb
  avg_ram_mb = [double]$wrapperSummary.avg_ram_mb
  peak_io_mb_s = [double]$wrapperSummary.peak_io_mb_s
  cpu_pct = [double]$wrapperSummary.cpu_pct
  candidate_manifest_sha256 = (Get-FileHash -LiteralPath $candidateManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
  trainer_summary_sha256 = (Get-FileHash -LiteralPath $trainerSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
  wrapper_summary_sha256 = (Get-FileHash -LiteralPath $wrapperSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
  cleanup_summary_sha256 = (Get-FileHash -LiteralPath $cleanupSummaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
  development_training_labels_staged = $true
  development_evaluation_labels_staged = $false
  heldout_labels_staged = $false
  external_checkpoint_loaded = $false
  shared_wsl_runtime_stopped = $false
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $cappedOut "run_receipt.json") -Encoding UTF8
if ($status -ne 0) { throw "capped architecture candidate aborted with exit code $status; artifacts were preserved" }
if (-not $receipt.caps_enforced) { throw "architecture candidate wrapper did not verify hard caps" }
if (-not $receipt.cleanup_passed) { throw "architecture candidate cleanup audit failed" }
Write-Output (Join-Path $cappedOut "run_receipt.json")
