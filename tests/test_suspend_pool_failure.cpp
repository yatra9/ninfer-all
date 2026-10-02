#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include <cuda.h>
#include <iostream>
#include <stdexcept>

namespace {
enum class Fault { None, Create, Map, Access, Unmap, Release };
Fault attach_fault = Fault::None, cleanup_fault = Fault::None;
int calls = 0, live_handles = 0, live_maps = 0;
bool fail(Fault stage) {
    if (stage == attach_fault && ++calls == 2) { attach_fault = Fault::None; return true; }
    if (stage == cleanup_fault) { cleanup_fault = Fault::None; return true; }
    return false;
}
void require(bool value) {
    if (!value) { throw std::runtime_error("pool attach rollback invariant failed"); }
}
template<class Pool> void check(Pool& pool, Fault stage, Fault cleanup) {
    const auto address = pool.arena().data;
    const auto bytes = pool.physical_bytes();
    pool.detach_backing();
    require(live_handles == 0 && live_maps == 0);
    pool.attach_backing();
    require(pool.arena().data == address && pool.physical_bytes() == bytes);
    pool.detach_backing();
    require(live_handles == 0 && live_maps == 0);
    calls = 0;
    attach_fault = stage;
    cleanup_fault = cleanup;
    bool failed = false;
    try { pool.attach_backing(); } catch (const std::runtime_error&) { failed = true; }
    require(failed && attach_fault == Fault::None && cleanup_fault == Fault::None);
    require(pool.poisoned() && pool.arena().data == address);
    if (cleanup == Fault::None) {
        require(pool.physical_bytes() == 0 && live_handles == 0 && live_maps == 0);
    } else {
        require(pool.physical_bytes() > 0 && live_handles == 1);
        require(live_maps == (cleanup == Fault::Unmap ? 1 : 0));
    }
}
}

extern "C" {
CUresult CUDAAPI __real_cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
CUresult CUDAAPI __wrap_cuMemCreate(CUmemGenericAllocationHandle* h, size_t n, const CUmemAllocationProp* p, unsigned long long f) {
    if (fail(Fault::Create)) { return CUDA_ERROR_OUT_OF_MEMORY; }
    const auto r = __real_cuMemCreate(h, n, p, f);
    if (r == CUDA_SUCCESS) { ++live_handles; }
    return r;
}
CUresult CUDAAPI __real_cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
CUresult CUDAAPI __wrap_cuMemMap(CUdeviceptr p, size_t n, size_t o, CUmemGenericAllocationHandle h, unsigned long long f) {
    if (fail(Fault::Map)) { return CUDA_ERROR_INVALID_VALUE; }
    const auto r = __real_cuMemMap(p, n, o, h, f);
    if (r == CUDA_SUCCESS) { ++live_maps; }
    return r;
}
CUresult CUDAAPI __real_cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
CUresult CUDAAPI __wrap_cuMemSetAccess(CUdeviceptr p, size_t n, const CUmemAccessDesc* d, size_t c) {
    return fail(Fault::Access) ? CUDA_ERROR_INVALID_VALUE : __real_cuMemSetAccess(p, n, d, c);
}
CUresult CUDAAPI __real_cuMemUnmap(CUdeviceptr, size_t);
CUresult CUDAAPI __wrap_cuMemUnmap(CUdeviceptr p, size_t n) {
    if (fail(Fault::Unmap)) { return CUDA_ERROR_INVALID_VALUE; }
    const auto r = __real_cuMemUnmap(p, n);
    if (r == CUDA_SUCCESS) { --live_maps; }
    return r;
}
CUresult CUDAAPI __real_cuMemRelease(CUmemGenericAllocationHandle);
CUresult CUDAAPI __wrap_cuMemRelease(CUmemGenericAllocationHandle h) {
    if (fail(Fault::Release)) { return CUDA_ERROR_INVALID_VALUE; }
    const auto r = __real_cuMemRelease(h);
    if (r == CUDA_SUCCESS) { --live_handles; }
    return r;
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        ninfer::DeviceContext device(0);
        if (!ninfer::EvictableKVPool::supported(device)) { return 77; }
        for (const auto stage : {Fault::Create, Fault::Map, Fault::Access}) {
            for (const auto cleanup : {Fault::None, Fault::Unmap, Fault::Release}) {
                {
                    constexpr auto chunk = ninfer::EvictableWeightPool::kChunkBytes;
                    ninfer::EvictableWeightPool pool(device, {.arena_bytes = 3 * chunk, .evictable_tail_bytes = 2 * chunk});
                    check(pool, stage, cleanup);
                }
                require(live_handles == 0 && live_maps == 0);
                {
                    const auto g = ninfer::EvictableKVPool::device_granularity(device);
                    ninfer::EvictableKVPool pool(device, {.arena_bytes = 4 * g, .lendable_prefix_bytes = 2 * g, .window_capacity_bytes = g});
                    check(pool, stage, cleanup);
                }
                require(live_handles == 0 && live_maps == 0);
            }
        }
        std::cout << "PASS: 18 pool attach/cleanup failures; destructor retries retained resources\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
