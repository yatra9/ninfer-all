// Acceptance-only fixed-size workspace poisoning and persistent-copy failure injection.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <new>

namespace {
template<class F> F next(const char* symbol) {
    const auto function = reinterpret_cast<F>(dlsym(RTLD_NEXT, symbol));
    if (!function) { std::fprintf(stderr, "missing symbol %s\n", symbol); std::abort(); }
    return function;
}
std::size_t configured(const char* name) {
    const auto value = std::getenv(name);
    return value ? std::strtoull(value, nullptr, 10) : 0;
}
bool fault(const char* kind, std::size_t bytes) {
    static std::atomic<bool> injected = false;
    const auto mode = std::getenv("NINFER_COPY_FAILURE");
    const bool backing = std::strcmp(kind, "access") == 0 || std::strcmp(kind, "unmap") == 0;
    if (!mode || std::strcmp(mode, kind) != 0 ||
        bytes != configured(backing ? "NINFER_WORKSPACE_BYTES" : "NINFER_SNAPSHOT_BYTES")) {
        return false;
    }
    if (injected.exchange(true)) { return false; }
    std::fprintf(stderr, "backing-probe injected %s bytes=%zu\n", kind, bytes);
    return true;
}
}

extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr pointer, std::size_t bytes,
                                          const CUmemAccessDesc* descriptors, std::size_t count) {
    static const auto call = next<decltype(&cuMemSetAccess)>("cuMemSetAccess");
    if (fault("access", bytes)) { return CUDA_ERROR_INVALID_VALUE; }
    const auto result = call(pointer, bytes, descriptors, count);
    if (result == CUDA_SUCCESS && bytes == configured("NINFER_WORKSPACE_BYTES")) {
        static std::atomic<unsigned int> generation = 0;
        const auto current = ++generation;
        // Initial storage uses its normal initialization. Poison every newly attached backing.
        if (current > 1) {
            static const auto fill = next<decltype(&cuMemsetD8)>("cuMemsetD8_v2");
            const auto value = static_cast<unsigned char>(0x51U + current);
            const auto filled = fill(pointer, value, bytes);
            std::fprintf(stderr, "backing-probe poison generation=%u pointer=%llx bytes=%zu value=%u result=%d\n",
                         current, static_cast<unsigned long long>(pointer), bytes, value, int(filled));
            if (filled != CUDA_SUCCESS) { return filled; }
        }
    }
    return result;
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr pointer, std::size_t bytes) {
    static const auto call = next<decltype(&cuMemUnmap)>("cuMemUnmap");
    if (fault("unmap", bytes)) { return CUDA_ERROR_INVALID_VALUE; }
    return call(pointer, bytes);
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    static const auto call = next<decltype(&cuMemRelease)>("cuMemRelease");
    // The test enables injection after startup. Fail the first physical release after unmap;
    // the one-shot fault lets destructor cleanup retry the retained handle.
    const auto armed = configured("NINFER_SNAPSHOT_BYTES");
    if (armed && fault("release", armed)) { return CUDA_ERROR_INVALID_VALUE; }
    return call(handle);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t bytes, cudaMemcpyKind kind) {
    static const auto call = next<decltype(&cudaMemcpy)>("cudaMemcpy");
    if ((kind == cudaMemcpyDeviceToHost && fault("d2h", bytes)) ||
        (kind == cudaMemcpyHostToDevice && fault("h2d", bytes))) {
        return cudaErrorInvalidValue;
    }
    return call(destination, source, bytes, kind);
}

void* operator new[](std::size_t bytes) {
    if (fault("host", bytes)) { throw std::bad_alloc(); }
    static const auto call = next<void*(*)(std::size_t)>("_Znam");
    return call(bytes);
}
