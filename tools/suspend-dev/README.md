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
