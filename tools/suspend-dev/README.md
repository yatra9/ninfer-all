# Suspend development container

Use the locally built `ninfer-all:build` image and its existing BuildKit cache
(`id=ninfer-build`). That base image contains the product executables; the Ninja
objects and CMake cache live in the BuildKit cache mount. This Dockerfile imports
the current configuration into `/build` without changing source or build paths.
Keep the BuildKit cache until the development image has been created.

From the repository root:

```powershell
wslc build -f tools/suspend-dev/Dockerfile -t ninfer-suspend:dev .
wslc run -d --name ninfer-suspend-dev --gpus all -v "${PWD}:/workspace:ro" ninfer-suspend:dev
wslc exec ninfer-suspend-dev sh /workspace/tools/suspend-dev/prepare-build.sh
wslc exec ninfer-suspend-dev sh /workspace/tools/suspend-dev/build-test.sh
```

The build script uses eight jobs, builds the standalone suspend test and the
affected loading-test object, and records compiler output in
`/tmp/ninfer-suspend-build.log`. The build directory is recorded in
`/build/suspend-build-dir`. After the build, run
`ctest --test-dir <recorded-directory> -R '^ninfer_model_suspend_test$' --output-on-failure`.
The preparation script synchronizes by content and keeps unchanged source mtimes,
then enables test registration in the existing Ninja tree. Rebuilds required by
changed headers or sources still run normally.

The development image retains the master baseline. Subsequent incremental outputs
live in the container's writable layer; stop and restart that container to keep
them. Deleting the container loses those subsequent outputs.

## Real-model acceptance

Use a WSLC native named volume for the model rather than a Windows `/mnt` or
bind-mounted directory. Copy the explicitly selected artifact directory once:

```powershell
wslc volume create ninfer-suspend-models
wslc run --rm -v "${modelDirectory}:/source:ro" -v ninfer-suspend-models:/models ninfer-suspend:dev cp -a /source/. /models/
wslc run --rm -v ninfer-suspend-models:/models:ro ninfer-suspend:dev sh -c "cd /models && sha256sum -c SHA256SUMS"
```

Set `$modelDirectory` to the selected Windows model directory. The checksum
command applies to artifacts distributed with `SHA256SUMS`. Retain the original
files. Mount the native volume read-only at `/models` in the acceptance server.

Copy `acceptance.py` and `graph-trace.cpp` into the acceptance output directory
mounted at `/acceptance`. Compile the observation library before starting the
server:

```sh
g++ -shared -fPIC -O2 -std=c++20 -I/usr/local/cuda/include /acceptance/graph-trace.cpp -o /acceptance/graph-trace.so -ldl -pthread
```

Start `ninfer-serve` with `--enable-model-suspend` and the explicitly selected
model/options, setting `LD_PRELOAD=/acceptance/graph-trace.so` and
`NINFER_GRAPH_TRACE=/acceptance/graph-trace.log`. Use a new trace file for each
server instance. Do not overwrite the library while a server has it loaded.
Then run in that container:

```sh
python3 /acceptance/acceptance.py --model qwen3.8-27b --cycles 100 --trace /acceptance/graph-trace.log --report /acceptance/100-cycle-report.json
```

The script targets a single 24 GiB GPU and requires at least 20 GiB released and
free while suspended. It records timings, device memory, server process RSS and
high-water mark, continuation usage, and actual Graph calls. It compares rendered
continuation output and output token counts, rejects inference while suspended,
and verifies existing GraphExec reuse without recapture. It explicitly selects
`auto_resume: false` and uses a valid generation request for the suspended 503 check.
Shared automatic resume is covered by the Engine/HTTP tests listed in `tests/README.md`.
It writes progress after
each cycle; `complete: true` marks completion. Process RAM is not total host RAM,
and output comparison does not independently prove raw token-vector equality.
Vision overlay execution, workspace poisoning, and fault tests remain separate
acceptance tasks in `PLAN_SUSPEND.md`.
## Exact tokens, overlay loans and failures

Build the standalone `ninfer_qwen3_5_suspend_real_test` target in the existing
incremental tree. Supply an explicit artifact, PNG and KV capacity:

```sh
/acceptance/suspend-real /models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer /acceptance/colors.png 163840
```

The executable runs three text/Vision comparisons of exact generated token IDs.
Prefix reuse is disabled so an image frontier cannot bypass Vision execution.
A small KV capacity such as 2048 selects the existing weight-tail fallback when
one planned Vision window cannot fit in the lendable KV prefix. The actual
`overlay` and `exclusive` counters must confirm the route; capacity alone is
not evidence of route execution. With the optional `cache` argument, compare the
same cached frontier before and after suspend. A cold-versus-cached comparison
is reported separately and is not used as the restoration oracle. The optional
control argument performs only this cold/cached comparison with model suspend
disabled, to distinguish an existing execution-path difference from restoration.

Compile the acceptance-only probe and separate memory borrower:

```sh
g++ -shared -fPIC -O2 -std=c++20 -I/usr/local/cuda/include /acceptance/backing-probe.cpp -o /acceptance/backing-probe.so -ldl
nvcc -O2 -arch=sm_86 /acceptance/gpu-borrow.cu -o /acceptance/gpu-borrow
```

Set `LD_PRELOAD=/acceptance/graph-trace.so:/acceptance/backing-probe.so`,
`NINFER_GRAPH_TRACE` to a fresh file, and `NINFER_WORKSPACE_BYTES` to the measured
physical workspace size (218103808 on the accepted 160K configuration). The
probe fills every newly attached workspace backing with a different byte value;
initial backing uses the normal initialization. Verify poison records occurred,
exact token comparisons pass and Graph capture/instantiate do not increase.
`NINFER_BORROW_COMMAND=/acceptance/gpu-borrow` also runs a separate process while
the Engine is suspended: it allocates, fills, verifies and frees 20 GiB.

For one-shot failures, use `LD_PRELOAD=/acceptance/backing-probe.so` and set
`NINFER_COPY_FAILURE` to the final argument: `host`, `d2h`, `h2d`, `unmap`,
`release` or `access`. The test configures full-persistent and workspace sizes
after startup, before transition. Host/copy failures target the exact persistent
capacity; the `host` mode selects pageable snapshots to exercise allocation during
suspend, while other modes retain the default startup-pinned snapshots. Startup
pin-allocation failure is covered by the standalone model suspend test.
Unmap/access target the physical workspace extent. Release fails the
first physical release after injection is armed, testing incomplete pool detach.
These simulated API failures occur before the intercepted operation; cleanup
can retry because each injection fires once. The test checks Ready after Host
allocation failure, otherwise Error with inference blocked, preserved diagnostic
snapshots where available, and normal destruction. It does not simulate a
permanently lost CUDA context. Keep probes and diagnostics outside product builds.
