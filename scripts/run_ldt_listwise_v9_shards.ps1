param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$OutRoot,
  [string]$Preregistration = "configs/preregistrations/ldt_listwise_array_v9.yaml",
  [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wrapper = Join-Path $repo "scripts\run_ldt_array_confirmation_capped_wsl.ps1"
$merge = Join-Path $repo "scripts\merge_ldt_listwise_array_v9_shards.py"
$manifest = (Resolve-Path $DatasetManifest).Path
$out = [IO.Path]::GetFullPath($OutRoot)
$repoPrefix = $repo.TrimEnd('\') + '\'
if ($out.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Generated shard runs must stay outside the DataGoblin repository"
}
if (Test-Path -LiteralPath $out) {
  throw "Shard output root already exists: $out"
}
New-Item -ItemType Directory -Path $out | Out-Null

$candidateIds = @(
  "looped-h4-pair-control-v9",
  "looped-h4-pooled-listwise-v9"
)
$shardRuns = @()
foreach ($candidateId in $candidateIds) {
  $runId = "ldt-listwise-v9-$candidateId"
  $runDir = Join-Path $out $candidateId
  Write-Output "START $candidateId"
  & $wrapper `
    -DatasetManifest $manifest `
    -OutDir $runDir `
    -Preregistration $Preregistration `
    -RunId $runId `
    -Distro $Distro `
    -CandidateId $candidateId
  if ($LASTEXITCODE -ne 0) {
    throw "Capped shard failed: $candidateId"
  }
  $shardRuns += $runDir
  Write-Output "COMPLETE $candidateId"
}

$recurrentId = "looped-h4-recurrent-listwise-v9"
foreach ($seed in @(29, 53, 97)) {
  $runId = "ldt-listwise-v9-recurrent-seed-$seed"
  $runDir = Join-Path $out "$recurrentId-seed-$seed"
  Write-Output "START $recurrentId seed $seed"
  & $wrapper `
    -DatasetManifest $manifest `
    -OutDir $runDir `
    -Preregistration $Preregistration `
    -RunId $runId `
    -Distro $Distro `
    -CandidateId $recurrentId `
    -Seed $seed
  if ($LASTEXITCODE -ne 0) {
    throw "Capped shard failed: $recurrentId seed $seed"
  }
  $shardRuns += $runDir
  Write-Output "COMPLETE $recurrentId seed $seed"
}

$mergeArgs = @(
  $merge,
  "--preregistration",
  (Join-Path $repo $Preregistration),
  "--dataset-manifest",
  $manifest
)
foreach ($runDir in $shardRuns) {
  $mergeArgs += @("--shard-run", $runDir)
}
$merged = Join-Path $out "merged"
$mergeArgs += @("--out", $merged)
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = Join-Path $repo "src"
  & python @mergeArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Could not merge the complete v9 shard matrix"
  }
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
Write-Output $merged
