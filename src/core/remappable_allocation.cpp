#include "core/remappable_allocation.h"
#include "core/device.h"

#include <cuda.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {
void checked(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver error"));
}
void checked(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}
}
struct RemappableDeviceAllocation::Impl {
    int device = 0;
    CUcontext context = nullptr;
    CUdeviceptr va = 0;
    CUmemGenericAllocationHandle handle = 0;
    std::size_t bytes = 0;
    bool mapped = false;
    bool poisoned = false;
    AttachFault fault = AttachFault::None;
    CUmemAllocationProp prop{};
    void require_context() const {
        CUcontext current = nullptr;
        checked(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
        if (current != context) { throw std::logic_error("remappable allocation used in a different CUDA context"); }
    }
    void fail(AttachFault stage) {
        if (fault != stage) { return; }
        fault = AttachFault::None;
        throw std::runtime_error("injected VMM attach failure");
    }
    ~Impl() {
        if (!va && !handle) { return; }
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) { return; }
        if (mapped) { (void)cuMemUnmap(va, bytes); }
        if (handle) { (void)cuMemRelease(handle); }
        if (va) { (void)cuMemAddressFree(va, bytes); }
        CUcontext popped = nullptr;
        (void)cuCtxPopCurrent(&popped);
    }
};
RemappableDeviceAllocation::RemappableDeviceAllocation(std::size_t bytes)
    : impl_(std::make_unique<Impl>()) {
    if (!bytes) { throw std::invalid_argument("remappable allocation must be nonempty"); }
    checked(cudaGetDevice(&impl_->device), "cudaGetDevice");
    checked(cudaFree(nullptr), "initialize CUDA context");
    checked(cuCtxGetCurrent(&impl_->context), "cuCtxGetCurrent");
    impl_->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    impl_->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    impl_->prop.location.id = impl_->device;
    std::size_t granularity = 0;
    checked(cuMemGetAllocationGranularity(&granularity, &impl_->prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM), "cuMemGetAllocationGranularity");
    if (bytes > std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
        throw std::overflow_error("remappable allocation size overflow");
    }
    impl_->bytes = (bytes + granularity - 1) / granularity * granularity;
    checked(cuMemAddressReserve(&impl_->va, impl_->bytes, granularity, 0, 0), "cuMemAddressReserve");
    attach();
}
RemappableDeviceAllocation::~RemappableDeviceAllocation() = default;
void* RemappableDeviceAllocation::data() const noexcept { return reinterpret_cast<void*>(impl_->va); }
std::size_t RemappableDeviceAllocation::physical_bytes() const noexcept {
    return impl_->handle ? impl_->bytes : 0;
}
bool RemappableDeviceAllocation::attached() const noexcept { return impl_->mapped; }
void RemappableDeviceAllocation::attach() {
    impl_->require_context();
    if (impl_->poisoned) { throw std::logic_error("remappable allocation cleanup failed"); }
    if (impl_->mapped) { return; }
    if (impl_->handle) { throw std::logic_error("remappable allocation has incomplete detach"); }
    impl_->fail(AttachFault::Create);
    checked(cuMemCreate(&impl_->handle, impl_->bytes, &impl_->prop, 0), "cuMemCreate");
    try {
        impl_->fail(AttachFault::Map);
        checked(cuMemMap(impl_->va, impl_->bytes, 0, impl_->handle, 0), "cuMemMap");
        impl_->mapped = true;
        CUmemAccessDesc access{};
        access.location = impl_->prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        impl_->fail(AttachFault::Access);
        checked(cuMemSetAccess(impl_->va, impl_->bytes, &access, 1), "cuMemSetAccess");
    } catch (...) {
        if (impl_->mapped && cuMemUnmap(impl_->va, impl_->bytes) == CUDA_SUCCESS) { impl_->mapped = false; }
        if (!impl_->mapped && cuMemRelease(impl_->handle) == CUDA_SUCCESS) { impl_->handle = 0; }
        impl_->poisoned = impl_->mapped || impl_->handle != 0;
        throw;
    }
}
void RemappableDeviceAllocation::detach() {
    impl_->require_context();
    if (impl_->mapped) {
        checked(cuMemUnmap(impl_->va, impl_->bytes), "cuMemUnmap");
        impl_->mapped = false;
    }
    if (impl_->handle) {
        checked(cuMemRelease(impl_->handle), "cuMemRelease");
        impl_->handle = 0;
    }
}
void RemappableDeviceAllocation::inject_attach_fault(AttachFault fault) noexcept { impl_->fault = fault; }
} // namespace ninfer
