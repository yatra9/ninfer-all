// A separate process must be able to use the memory released by the suspended Engine.
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    constexpr std::size_t bytes = 20ULL << 30;
    void* memory = nullptr;
    const auto allocated = cudaMalloc(&memory, bytes);
    if (allocated != cudaSuccess) {
        std::fprintf(stderr, "20 GiB allocation: %s\n", cudaGetErrorString(allocated));
        return 1;
    }
    auto result = cudaMemset(memory, 0x6d, bytes);
    if (result == cudaSuccess) { result = cudaDeviceSynchronize(); }
    unsigned char first = 0, last = 0;
    if (result == cudaSuccess) { result = cudaMemcpy(&first, memory, 1, cudaMemcpyDeviceToHost); }
    if (result == cudaSuccess) {
        result = cudaMemcpy(&last, static_cast<unsigned char*>(memory) + bytes - 1,
                            1, cudaMemcpyDeviceToHost);
    }
    const auto freed = cudaFree(memory);
    if (result != cudaSuccess || freed != cudaSuccess || first != 0x6d || last != 0x6d) {
        std::fprintf(stderr, "20 GiB fill/read/free validation failed\n");
        return 1;
    }
    std::puts("separate process allocated, filled, verified and freed 20 GiB");
    return 0;
}
