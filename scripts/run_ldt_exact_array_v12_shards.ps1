param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$OutRoot,
  [string]$Preregistration = "configs/preregistrations/ldt_exact_array_likelihood_v12.yaml",
  [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wrapper = Join-Path $repo "scripts\run_ldt_array_confirmation_capped_wsl.ps1"
$merge = Join-Path $repo "scripts\merge_ldt_listwise_array_v9_shards.py"
$select = Join-Path $repo "scripts\select_ldt_listwise_array_v9.py"
$manifest = (Resolve-Path $DatasetManifest).Path
$prereg = (Resolve-Path (Join-Path $repo $Preregistration)).Path
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
  "looped-h4-pair-control-v12",
  "looped-h4-recurrent-factorized-v12",
  "exact-array-h4-factorized-v12",
  "exact-array-h4-global-map-v12"
)
$seeds = @(41, 73, 113)
$shardRuns = @()
foreach ($candidateId in $candidateIds) {
  foreach ($seed in $seeds) {
    $runId = "ldt-exact-array-v12-$candidateId-seed-$seed"
    $runDir = Join-Path $out "$candidateId-seed-$seed"
    Write-Output "START $candidateId seed $seed"
    & $wrapper `
      -DatasetManifest $manifest `
      -OutDir $runDir `
      -Preregistration $Preregistration `
      -RunId $runId `
      -Distro $Distro `
      -CandidateId $candidateId `
      -Seed $seed
    if ($LASTEXITCODE -ne 0) {
      throw "Capped shard failed: $candidateId seed $seed"
    }
    $shardRuns += $runDir
    Write-Output "COMPLETE $candidateId seed $seed"
  }
}

$mergeArgs = @(
  $merge,
  "--preregistration", $prereg,
  "--dataset-manifest", $manifest
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
    throw "Could not merge the complete v12 shard matrix"
  }
  $selected = Join-Path $out "selected"
  & python $select `
    --preregistration $prereg `
    --trainer-summary (Join-Path $merged "trainer_summary.json") `
    --pair-predictions (Join-Path $merged "heldout_pair_predictions.jsonl") `
    --array-predictions (Join-Path $merged "heldout_array_predictions.jsonl") `
    --out $selected
  if ($LASTEXITCODE -ne 0) {
    throw "Could not freeze v12 development selection"
  }
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
Write-Output (Join-Path $out "selected\selection_receipt.json")
