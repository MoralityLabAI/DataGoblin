# Streaming Inkling variable-loop Q4 weights on Windows

This runbook uses `TML-d24-varloop-research-IQ4_NL.gguf` with llama.cpp
`b12602` (`418fe97b5`) and its Windows Vulkan backend. CPU weights stay
memory-mapped. The loader reads GPU weights through an unbuffered file
handle in 4 MiB chunks and uploads them to Vulkan. This avoids holding a
second copy of the GPU payload in the Windows file cache.

## Build the runtime

Use an x64 Visual Studio Developer PowerShell with CMake and Ninja.
Keep the source, model, runtime, and generated artifacts outside Git.
Start with a clean local copy of the pinned source and the matching
portable Windows Vulkan runtime. From the DataGoblin checkout, set
`$Source`, `$Portable`, `$Headers`, `$Imports`, and `$Build` to your local
paths. Stage the exact-revision headers from `include` and `ggml/include`
together in `$Headers`.

```powershell
$Repo = (Get-Location).Path
git -C $Source apply "$Repo/scripts/inkling_no_prefetch.patch"
git -C $Source apply --unidiff-zero "$Repo/scripts/inkling_stream_gpu_weights.patch"
git -C $Source apply --unidiff-zero "$Repo/scripts/inkling_loop_graph_arena.patch"
& "$Repo/scripts/build_inkling_trace.ps1" -Runtime $Portable -Headers $Headers -Out $Imports
```

Apply the patches in that order and stop on an error. The last patch
sizes the GGML graph arena for repeated core layers and is required for
the six-pass configuration that exceeded the stock arena.

Point `ggml_DIR` at a directory containing `ggml-config.cmake`. The
following import file uses the portable DLLs and the import libraries
created in `$Imports`. Use absolute paths with forward slashes for
`PORTABLE`, `IMPORTS`, and `SOURCE`.

```cmake
add_library(ggml::base SHARED IMPORTED)
set_target_properties(ggml::base PROPERTIES
    IMPORTED_LOCATION "PORTABLE/ggml-base.dll"
    IMPORTED_IMPLIB "IMPORTS/ggml-base.lib"
    INTERFACE_INCLUDE_DIRECTORIES "SOURCE/ggml/include"
    INTERFACE_COMPILE_DEFINITIONS "GGML_SHARED")
add_library(ggml::ggml SHARED IMPORTED)
set_target_properties(ggml::ggml PROPERTIES
    IMPORTED_LOCATION "PORTABLE/ggml.dll"
    IMPORTED_IMPLIB "IMPORTS/ggml.lib"
    INTERFACE_LINK_LIBRARIES ggml::base)
```

Set `$GgmlConfig` to the directory containing that file. Configure and
build only `llama` against the packaged GGML libraries.

```powershell
cmake -S $Source -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-Dggml_DIR=$GgmlConfig" -DLLAMA_USE_SYSTEM_GGML=ON -DBUILD_SHARED_LIBS=ON `
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_NUMBER=12602 -DLLAMA_BUILD_COMMIT=418fe97b5
cmake --build $Build --target llama --parallel 2
```

Create a fresh runtime directory with the rebuilt `llama.dll` and the
original `ggml*.dll` and `libomp.dll`. Put that directory first on `PATH`.
Run `inkling_trace.exe --check-binding` using that DLL set before a GPU
launch. Record SHA-256 digests for the source archive, checkpoints, patch
files, tracer, and loaded runtime libraries.

## Choose a placement and launch

The completed six-pass configuration requests 85 GPU layers and keeps
physical blocks 0–5 and 20–23 on CPU. Put this single pattern in the
placement file for this checkpoint.

```text
^blk\.([0-5]|2[0-3])\.
```

All 12 physical recurrent core layers, blocks 6–17, can then reside on
the GPU. The RTX 5080 loader measured **8,633 MiB on Vulkan** and
**8,167 MiB mapped on CPU**. It streamed exactly **9,052,198,144 bytes**
and completed the [14-case holdout batch](six_pass_holdout.md).

An earlier placement kept only blocks 20–23 on CPU. It streamed
**12,672,091,648 bytes** and reached the system commit reserve during
context creation. Check host commit as well as VRAM when sizing placement.

Set the two opt-ins in the launcher environment. Use `mmap` as the
requested mode. Keep context at 1024 for this campaign.

```powershell
$env:PATH = "$PatchedRuntime;$env:PATH"
$env:DATAGOBLIN_SKIP_GGUF_PREFETCH = '1'
$env:DATAGOBLIN_STREAM_GPU_WEIGHTS = '1'
```

Pass `Vulkan1 6 85` to the tracer, followed by the placement file, frozen
prompt directory, new output directory, `mmap`, and the registered token
cap. Verify `Vulkan1` identifies the RTX 5080 on the host first. The
[gym guide](../inkling_variable_loop_gym.md#native-gguf-boundary-capture-on-windows)
describes how to stage prompts and the full tracer argument order.

Run through the Windows resource supervisor. Its launch guard requires
9.25 GiB of available physical RAM, 18 GiB of available system commit,
14,500 MiB of free VRAM, GPU utilization at or below 5%, and an empty GPU
compute process list. During inference it reserves 5 GiB of RAM, 4 GiB
of commit, and 1,800 MiB of VRAM. The job is capped at 19 GiB of commit
and a 5.75 GiB native working set. Stop the owned process if another GPU
compute job appears or fresh telemetry is unavailable. On a single-NVIDIA
Windows host, `WindowsNvmlMonitor` reads fresh VRAM, utilization, and
compute PIDs with a five-second deadline. It retries intermittent
NVML error 999 within that same deadline. A missing sample still stops
the owned process.

## Verify the capture

- Confirm the load log says `disabling whole-file GGUF prefetch` and
  `streamed … device weight bytes from unbuffered GGUF reads`. Record the
  actual streamed byte count and CPU and Vulkan buffer sizes. Allocation
  lines can appear before streaming finishes.
- Save the exact command, placement digest, native log, resource receipt,
  and raw activations in the run directory. Update rollouts only from the
  durable completion records and replay each answer with its own task
  verifier. Mark interrupted traces as partial.
- Compare depths within a cohort that shares the checkpoint, DLLs, load
  mode, and placement. Use separate cohorts for changes to placement or
  runtime. Keep large captures and model weights outside Git.
