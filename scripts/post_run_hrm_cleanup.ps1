param(
  [Parameter(Mandatory = $true)][string]$RunId,
  [Parameter(Mandatory = $true)][string]$SummaryPath,
  [int[]]$OwnedPid = @(),
  [switch]$StopOwnedProcesses
)

$ErrorActionPreference = "Continue"

function Get-MemorySnapshot {
  $os = Get-CimInstance Win32_OperatingSystem
  $perf = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
  return [ordered]@{
    total_gb = [math]::Round(([double]$os.TotalVisibleMemorySize * 1KB) / 1GB, 3)
    free_gb = [math]::Round(([double]$os.FreePhysicalMemory * 1KB) / 1GB, 3)
    available_mb = if ($perf) { [int]$perf.AvailableMBytes } else { $null }
    committed_gb = if ($perf) { [math]::Round(([double]$perf.CommittedBytes) / 1GB, 3) } else { $null }
    commit_limit_gb = if ($perf) { [math]::Round(([double]$perf.CommitLimit) / 1GB, 3) } else { $null }
    cache_gb = if ($perf) { [math]::Round(([double]$perf.CacheBytes) / 1GB, 3) } else { $null }
    standby_gb = if ($perf) {
      [math]::Round((
        [double]$perf.StandbyCacheReserveBytes +
        [double]$perf.StandbyCacheNormalPriorityBytes +
        [double]$perf.StandbyCacheCoreBytes
      ) / 1GB, 3)
    } else { $null }
    page_reads_per_sec = if ($perf) { [int]$perf.PageReadsPersec } else { $null }
  }
}

function Get-GpuSnapshot {
  if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    return [ordered]@{ available = $false; compute_apps = @() }
  }
  return [ordered]@{
    available = $true
    gpu = @(& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null)
    compute_apps = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null)
  }
}

function Get-OwnedTree([int[]]$Roots) {
  $processes = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId)
  $seen = New-Object System.Collections.Generic.HashSet[int]
  $queue = New-Object System.Collections.Generic.Queue[int]
  foreach ($root in $Roots) {
    if ($root -gt 0 -and $seen.Add($root)) { $queue.Enqueue($root) }
  }
  while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    foreach ($child in ($processes | Where-Object { $_.ParentProcessId -eq $current })) {
      if ($seen.Add([int]$child.ProcessId)) { $queue.Enqueue([int]$child.ProcessId) }
    }
  }
  return @($seen)
}

$before = Get-MemorySnapshot
$gpuBefore = Get-GpuSnapshot
$ownedTree = @(Get-OwnedTree $OwnedPid)
$stopped = @()
if ($StopOwnedProcesses) {
  foreach ($ownedProcessId in ($ownedTree | Sort-Object -Descending)) {
    if ($ownedProcessId -eq $PID) { continue }
    if (Get-Process -Id $ownedProcessId -ErrorAction SilentlyContinue) {
      try {
        Stop-Process -Id $ownedProcessId -Force -ErrorAction Stop
        $stopped += $ownedProcessId
      } catch {
        $stopped += "failed:$ownedProcessId"
      }
    }
  }
}
$after = Get-MemorySnapshot
$gpuAfter = Get-GpuSnapshot
$lingering = @($ownedTree | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
$top = @(Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 12 Id,ProcessName,
  @{n="private_gb";e={[math]::Round($_.PrivateMemorySize64 / 1GB, 3)}},
  @{n="working_set_gb";e={[math]::Round($_.WorkingSet64 / 1GB, 3)}})
$summary = [ordered]@{
  run_id = $RunId
  ts_utc = (Get-Date).ToUniversalTime().ToString("o")
  owned_pids = @($OwnedPid)
  owned_process_tree = @($ownedTree)
  stopped = @($stopped)
  lingering_owned_pids = @($lingering)
  memory_before = $before
  memory_after = $after
  gpu_before = $gpuBefore
  gpu_after = $gpuAfter
  top_process_memory = $top
  global_standby_cache_purged = $false
  cleanup_passed = ($lingering.Count -eq 0)
}
$parent = Split-Path -Parent $SummaryPath
if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
$summary | ConvertTo-Json -Depth 8

