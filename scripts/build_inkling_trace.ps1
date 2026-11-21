param(
    [Parameter(Mandatory)][string]$Runtime,
    [Parameter(Mandatory)][string]$Headers,
    [Parameter(Mandatory)][string]$Out
)

# Run from an x64 Developer PowerShell using the headers of the downloaded runtime.
$ErrorActionPreference = 'Stop'
$Runtime = (Resolve-Path -LiteralPath $Runtime).Path
$Headers = (Resolve-Path -LiteralPath $Headers).Path
$null = New-Item -ItemType Directory -Path $Out -Force
$Out = (Resolve-Path -LiteralPath $Out).Path
$importLibs = @()
foreach ($library in 'llama', 'ggml', 'ggml-base') {
    $dll = Join-Path $Runtime "$library.dll"
    $exports = & dumpbin.exe /nologo /exports $dll
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $names = foreach ($line in $exports) {
        if ($line -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z_][A-Za-z_0-9]*)\s*$') {
            $Matches[1]
        }
    }
    if (!$names) { throw "No C exports found in $dll" }
    $def = Join-Path $Out "$library.def"
    $lib = Join-Path $Out "$library.lib"
    @("LIBRARY $library.dll", 'EXPORTS') + $names | Set-Content -LiteralPath $def -Encoding ascii
    & lib.exe /nologo /machine:x64 "/def:$def" "/out:$lib"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $importLibs += $lib
}

$source = Join-Path $PSScriptRoot 'inkling_trace.cpp'
$program = Join-Path $Out 'inkling_trace.exe'
$compile = @('/nologo','/std:c++17','/EHsc','/O2','/W4','/D_CRT_SECURE_NO_WARNINGS',
             "/I$Headers","/Fo$Out/inkling_trace.obj","/Fe$program",$source,'/link') + $importLibs
& cl.exe @compile
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$previousPath = $env:PATH
try {
    $env:PATH = "$Runtime;$previousPath"
    & $program --check-binding
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    $env:PATH = $previousPath
}
