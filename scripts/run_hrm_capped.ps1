param(
  [Parameter(Mandatory = $true)][string]$PreflightSummary,
  [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath $PreflightSummary -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$result = [ordered]@{
  schema_version = "1.0"
  run_id = $summary.run_id
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
  caps = $summary.caps
  child_started = $false
  owned_pids = @()
  status = "aborted"
  abort_reason = $null
  boundary = "This gate never substitutes monitor-and-kill behavior for an OS hard cap."
}

if ($summary.status -ne "ready_for_capped_wrapper") {
  $result.abort_reason = "preflight_not_permitted:$($summary.abort_reason)"
} else {
  # Windows Job Objects provide hard memory and CPU caps. Microsoft documents the
  # job I/O-rate API as unsupported on Windows 10 version 1607 and newer. A strict
  # run therefore requires a healthy WSL2/cgroup-v2 or container path.
  $result.abort_reason = "hard_io_cap_provider_unavailable_on_modern_windows"
}

$path = Join-Path $OutDir "wrapper_summary.json"
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
Write-Output $path
exit 3

