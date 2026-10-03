// Acceptance-only interposer: observe actual CUDA Graph calls without adding product diagnostics.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {
void record(const char* event, const void* handle, cudaError_t result) {
    static std::mutex mutex;
    static FILE* file = [] {
        const char* path = std::getenv("NINFER_GRAPH_TRACE");
        return path ? std::fopen(path, "a") : nullptr;
    }();
    std::lock_guard lock(mutex);
    if (file) {
        std::fprintf(file, "%s %p %d\n", event, handle, static_cast<int>(result));
        std::fflush(file);
    }
}
template <class Function> Function next(const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    if (!function) { std::fprintf(stderr, "missing CUDA symbol %s\n", name); std::abort(); }
    return function;
}
}
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    static const auto call = next<decltype(&cudaStreamBeginCapture)>("cudaStreamBeginCapture");
    const auto result = call(stream, mode);
    record("capture", stream, result);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* executable, cudaGraph_t graph,
                                                       unsigned long long flags) {
    static const auto call = next<decltype(&cudaGraphInstantiate)>("cudaGraphInstantiate");
    const auto result = call(executable, graph, flags);
    record("instantiate", result == cudaSuccess ? *executable : nullptr, result);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
    static const auto call = next<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");
    const auto result = call(executable, stream);
    record("launch", executable, result);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t executable) {
    static const auto call = next<decltype(&cudaGraphExecDestroy)>("cudaGraphExecDestroy");
    const auto result = call(executable);
    record("destroy", executable, result);
    return result;
}
