#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include "core/remappable_allocation.h"

#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool ok) { if (!ok) { throw std::runtime_error("backing invariant failed"); } }
template<class Pool> void roundtrip(Pool& pool, ninfer::DeviceContext& device) {
    const auto storage = pool.arena();
    const auto physical = pool.physical_bytes();
    std::vector<unsigned char> expected(storage.bytes, 0x37), actual(storage.bytes);
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaMemcpy(storage.data, expected.data(), storage.bytes, cudaMemcpyHostToDevice));
        device.synchronize();
        pool.detach_backing();
        require(pool.physical_bytes() == 0 && pool.arena().data == storage.data);
        pool.attach_backing();
        require(pool.physical_bytes() == physical && pool.arena().data == storage.data);
        CUDA_CHECK(cudaMemcpy(storage.data, expected.data(), storage.bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(actual.data(), storage.data, storage.bytes, cudaMemcpyDeviceToHost));
        require(actual == expected);
    }
}
}
int main() {
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || (err == cudaSuccess && !count)) { return 77; }
    try {
        CUDA_CHECK(err);
        ninfer::DeviceContext device(0);
        if (!ninfer::EvictableKVPool::supported(device)) { return 77; }
        using Allocation = ninfer::RemappableDeviceAllocation;
        for (auto fault : {Allocation::AttachFault::Create, Allocation::AttachFault::Map, Allocation::AttachFault::Access}) {
            Allocation allocation(4096);
            const auto address = allocation.data();
            allocation.detach();
            allocation.inject_attach_fault(fault);
            bool failed = false;
            try { allocation.attach(); } catch (const std::runtime_error&) { failed = true; }
            require(failed && !allocation.attached() && allocation.physical_bytes() == 0);
            allocation.attach();
            require(allocation.attached() && allocation.data() == address);
        }
        ninfer::DeviceArena arena(1024 * 1024 + 1, true);
        const auto span = arena.alloc_bytes(1024);
        const auto physical = arena.physical_bytes();
        require(physical >= arena.capacity());
        arena.detach_backing();
        ninfer::DeviceArena moved(std::move(arena));
        require(moved.base() == span.data && moved.used() == 1024 && moved.physical_bytes() == 0);
        moved.attach_backing();
        require(moved.physical_bytes() == physical);
        ninfer::DeviceArena assigned(1024, true);
        assigned = std::move(moved);
        require(assigned.base() == span.data && assigned.used() == 1024);
        assigned.detach_backing(); // destructor must only free the VA reservation

        constexpr auto chunk = ninfer::EvictableWeightPool::kChunkBytes;
        ninfer::EvictableWeightPool weights(device, {.arena_bytes = 2 * chunk, .evictable_tail_bytes = chunk});
        CUDA_CHECK(cudaMemset(weights.arena().data, 0x37, weights.arena().bytes));
        weights.capture_window_mirror(chunk, device.stream);
        roundtrip(weights, device);
        {
            auto loan = weights.evict(chunk, device.stream);
            bool rejected = false;
            try { weights.detach_backing(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected);
            CUDA_CHECK(cudaMemsetAsync(loan.leased().data, 0x91, loan.leased().bytes, device.stream));
            device.synchronize();
        }
        require(!weights.poisoned());
        std::vector<unsigned char> tail(chunk);
        CUDA_CHECK(cudaMemcpy(tail.data(), static_cast<unsigned char*>(weights.arena().data) + chunk,
                              chunk, cudaMemcpyDeviceToHost));
        for (auto byte : tail) { require(byte == 0x37); }

        const auto granule = ninfer::EvictableKVPool::lending_granularity(device);
        ninfer::EvictableKVPool kv(device, {.arena_bytes = 4 * granule, .lendable_prefix_bytes = 2 * granule,
                                          .window_capacity_bytes = granule});
        roundtrip(kv, device);
        {
            const std::size_t index = 0;
            auto loan = kv.lease({&index, 1}, device.stream);
            bool rejected = false;
            try { kv.detach_backing(); } catch (const std::logic_error&) { rejected = true; }
            require(rejected);
            CUDA_CHECK(cudaMemsetAsync(loan.leased().data, 0x19, loan.leased().bytes, device.stream));
            device.synchronize();
        }
        require(!kv.poisoned());
        weights.detach_backing(); kv.detach_backing();
        std::cout << "PASS: arena move/lifetime; pool new backing and subsequent overlay loans\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
