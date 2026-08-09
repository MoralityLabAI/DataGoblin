param(
  [Parameter(Mandatory = $true)][string]$DatasetManifest,
  [Parameter(Mandatory = $true)][string]$ReleaseManifest,
  [Parameter(Mandatory = $true)][string]$OutRoot,
  [string]$Config = "configs/preregistrations/game_adapter_architecture_tournament_v2.yaml",
  [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$driver = Join-Path $repo "scripts\game_adapter_architecture_tournament_v2.py"
$wrapper = Join-Path $repo "scripts\run_game_adapter_architecture_candidate_v2_capped_wsl.ps1"
$dataset = (Resolve-Path $DatasetManifest).Path
$release = (Resolve-Path $ReleaseManifest).Path
$configPath = (Resolve-Path (Join-Path $repo $Config)).Path
$out = [IO.Path]::GetFullPath($OutRoot)
$repoPrefix = $repo.TrimEnd('\') + '\'
if ($out.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Generated architecture matrix must stay outside the DataGoblin repository"
}
if (Test-Path -LiteralPath $out) { throw "Architecture matrix output already exists: $out" }
New-Item -ItemType Directory -Path $out | Out-Null

$runs = @()
$baseline = Join-Path $out "frozen-exact-key-control-v2-seed-0"
$previousPythonPath = $env:PYTHONPATH
try {
  $env:PYTHONPATH = Join-Path $repo "src"
  & python $driver train `
    --config $configPath `
    --dataset-manifest $dataset `
    --release-manifest $release `
    --candidate-id frozen-exact-key-control-v2 `
    --seed 0 `
    --out $baseline | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "deterministic architecture control failed" }
} finally {
  $env:PYTHONPATH = $previousPythonPath
}
$runs += $baseline

$candidateIds = @(
  "lora-like-r4-v2",
  "recurrent-one-pass-h32-v2",
  "hrm-style-loop4-h32-v2"
)
$seeds = @(31, 73, 127)
foreach ($candidateId in $candidateIds) {
  foreach ($seed in $seeds) {
    $runRoot = Join-Path $out "$candidateId-seed-$seed"
    Write-Output "START $candidateId seed $seed"
    & $wrapper `
      -DatasetManifest $dataset `
      -ReleaseManifest $release `
      -CandidateId $candidateId `
      -Seed $seed `
      -OutDir $runRoot `
      -Config $Config `
      -Distro $Distro | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "capped architecture cell failed: $candidateId seed $seed" }
    $runs += (Join-Path $runRoot "capped_run")
    Write-Output "COMPLETE $candidateId seed $seed"
  }
}

$manifest = [ordered]@{
  schema_version = "game-adapter-architecture-matrix.v2"
  tournament_config_sha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
  tournament_manifest_sha256 = (Get-FileHash -LiteralPath $dataset -Algorithm SHA256).Hash.ToLowerInvariant()
  candidate_run_dirs = $runs
  candidate_cell_count = $runs.Count
  steward_labels_staged = $false
  matrix_complete = ($runs.Count -eq 10)
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out "matrix_manifest.json") -Encoding UTF8
if (-not $manifest.matrix_complete) { throw "architecture matrix is incomplete" }
Write-Output (Join-Path $out "matrix_manifest.json")
