// Suspend PoC: new physical allocations at stable addresses, with one GraphExec.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(CUresult result) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    throw std::runtime_error(name ? name : "CUDA driver error");
}
void check(cudaError_t result) {
    if (result != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(result)); }
}

struct Backing {
    CUdeviceptr va = 0;
    std::size_t bytes = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool mapped = false;
    CUmemAllocationProp prop{};

    explicit Backing(std::size_t requested) {
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        std::size_t granularity = 0;
        check(cuMemGetAllocationGranularity(&granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        bytes = (requested + granularity - 1) / granularity * granularity;
        check(cuMemAddressReserve(&va, bytes, granularity, 0, 0));
        try { attach(); } catch (...) { cuMemAddressFree(va, bytes); throw; }
    }
    Backing(const Backing&) = delete;
    Backing& operator=(const Backing&) = delete;
    ~Backing() {
        if (mapped) { cuMemUnmap(va, bytes); }
        if (handle) { cuMemRelease(handle); }
        if (va) { cuMemAddressFree(va, bytes); }
    }
    void* data() const { return reinterpret_cast<void*>(va); }
    void attach() {
        check(cuMemCreate(&handle, bytes, &prop, 0));
        try {
            check(cuMemMap(va, bytes, 0, handle, 0));
            mapped = true;
            CUmemAccessDesc access{};
            access.location = prop.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            check(cuMemSetAccess(va, bytes, &access, 1));
        } catch (...) {
            if (mapped) { cuMemUnmap(va, bytes); mapped = false; }
            cuMemRelease(handle);
            handle = 0;
            throw;
        }
    }
    void detach() {
        check(cuMemUnmap(va, bytes));
        mapped = false;
        check(cuMemRelease(handle));
        handle = 0;
    }
};

struct Persistent {
    const std::uint32_t* weights;
    std::uint32_t* workspace;
    std::uint64_t rounds;
    std::uint64_t checksum;
};

__global__ void initialize_workspace(Persistent* state, std::size_t words) {
    for (std::size_t i = threadIdx.x; i < words; i += blockDim.x) {
        state->workspace[i] = state->weights[i] ^ static_cast<std::uint32_t>(state->rounds);
    }
}
__global__ void consume_workspace(Persistent* state, std::size_t words) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < words; ++i) { sum += state->workspace[i]; }
    state->checksum = sum;
    ++state->rounds;
}

struct Graph {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() {
        if (stream) { cudaStreamSynchronize(stream); }
        if (exec) { cudaGraphExecDestroy(exec); }
        if (graph) { cudaGraphDestroy(graph); }
        if (stream) { cudaStreamDestroy(stream); }
    }
};
} // namespace

int main() {
    int count = 0;
    const auto available = cudaGetDeviceCount(&count);
    if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver ||
        (available == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        check(available);
        check(cudaSetDevice(0));
        check(cudaFree(nullptr));
        check(cuInit(0));
        int vmm = 0;
        check(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0));
        if (!vmm) { std::cout << "SKIP: VMM unsupported\n"; return 77; }
        constexpr std::size_t words = 4096;
        constexpr int cycles = 1000;
        Backing weights(words * sizeof(std::uint32_t));
        Backing persistent(sizeof(Persistent));
        Backing workspace(words * sizeof(std::uint32_t));
        const auto weight_va = weights.va, state_va = persistent.va, work_va = workspace.va;
        std::vector<std::uint32_t> source(weights.bytes / sizeof(std::uint32_t));
        for (std::size_t i = 0; i < source.size(); ++i) { source[i] = i * 2654435761U; }
        std::vector<unsigned char> snapshot(persistent.bytes), restored(persistent.bytes);
        Persistent initial{static_cast<const std::uint32_t*>(weights.data()),
                           static_cast<std::uint32_t*>(workspace.data()), 0, 0};
        std::memcpy(snapshot.data(), &initial, sizeof(initial));
        check(cudaMemcpy(weights.data(), source.data(), weights.bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(persistent.data(), snapshot.data(), persistent.bytes, cudaMemcpyHostToDevice));
        Graph graph;
        check(cudaStreamCreateWithFlags(&graph.stream, cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeGlobal));
        initialize_workspace<<<1, 256, 0, graph.stream>>>(static_cast<Persistent*>(persistent.data()), words);
        consume_workspace<<<1, 1, 0, graph.stream>>>(static_cast<Persistent*>(persistent.data()), words);
        check(cudaStreamEndCapture(graph.stream, &graph.graph));
        check(cudaGraphInstantiate(&graph.exec, graph.graph, nullptr, nullptr, 0));
        const auto original_exec = graph.exec;
        for (int cycle = 0; cycle <= cycles; ++cycle) {
            if (cycle != 0) {
                check(cudaMemcpy(snapshot.data(), persistent.data(), persistent.bytes, cudaMemcpyDeviceToHost));
                weights.detach(); persistent.detach(); workspace.detach();
                weights.attach(); persistent.attach(); workspace.attach();
                check(cudaMemcpy(weights.data(), source.data(), weights.bytes, cudaMemcpyHostToDevice));
                check(cudaMemcpy(persistent.data(), snapshot.data(), persistent.bytes, cudaMemcpyHostToDevice));
                check(cudaMemcpy(restored.data(), persistent.data(), persistent.bytes, cudaMemcpyDeviceToHost));
                if (restored != snapshot) { throw std::runtime_error("whole persistent snapshot mismatch"); }
            }
            check(cudaMemsetAsync(workspace.data(), cycle % 256, workspace.bytes, graph.stream));
            check(cudaGraphLaunch(graph.exec, graph.stream));
            check(cudaStreamSynchronize(graph.stream));
            Persistent actual{};
            check(cudaMemcpy(&actual, persistent.data(), sizeof(actual), cudaMemcpyDeviceToHost));
            std::uint64_t expected = 0;
            for (std::size_t i = 0; i < words; ++i) { expected += source[i] ^ static_cast<std::uint32_t>(cycle); }
            if (actual.checksum != expected || actual.rounds != static_cast<std::uint64_t>(cycle + 1) ||
                actual.weights != initial.weights || actual.workspace != initial.workspace ||
                weights.va != weight_va || persistent.va != state_va || workspace.va != work_va ||
                graph.exec != original_exec) {
                throw std::runtime_error("graph replay / pointer / continuation mismatch at cycle " + std::to_string(cycle));
            }
        }
        std::cout << "PASS: 1000 new-backing cycles; exact persistent restore; stable pointers; "
                     "fresh poisoned workspace; one capture and instantiate\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "suspend VMM PoC failed: " << error.what() << '\n';
        return 1;
    }
}
