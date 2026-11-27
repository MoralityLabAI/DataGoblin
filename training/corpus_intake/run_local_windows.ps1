[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$CheckoutRoot = (Join-Path $env:TEMP "datagoblin-owned-math-checkouts"),
    [string]$OutputRoot = (Join-Path $env:TEMP "datagoblin-owned-math-audit"),
    [switch]$ReusePinnedCheckout
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$scanner = Join-Path $PSScriptRoot "audit_owned_math.py"
$roots = Join-Path $PSScriptRoot "proof_roots.py"

& $Python -c "import tiktoken"
if ($LASTEXITCODE -ne 0) {
    throw "Missing tiktoken. Install into this Python environment: python -m pip install tiktoken==0.12.0"
}

if (-not $ReusePinnedCheckout) {
    & $Python $scanner --clone-pinned --root $CheckoutRoot
    if ($LASTEXITCODE -ne 0) { throw "Pinned source checkout failed" }
}

& $Python $scanner --root $CheckoutRoot --out $OutputRoot --tokenizer "o200k_base"
if ($LASTEXITCODE -ne 0) { throw "Pinned source audit failed" }

$declarations = Join-Path $OutputRoot "lean_declarations.jsonl"
$clusters = Join-Path $OutputRoot "proof_root_candidates.json"
& $Python $roots --declarations $declarations --output $clusters
if ($LASTEXITCODE -ne 0) { throw "Theorem overlap clustering failed" }

Write-Host "Source-only evidence report: $(Join-Path $OutputRoot 'report.md')"
Write-Host "Proof root and train/eval overlap candidates: $clusters"
Write-Host "No GPU work or Lean kernel verification was invoked."
