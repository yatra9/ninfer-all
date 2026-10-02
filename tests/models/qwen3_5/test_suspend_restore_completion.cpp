#include "ninfer/engine.h"
#include "models/qwen3_5/execution_fixture.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>
#include <utility>

namespace {
using namespace ninfer;
using ninfer::test::artifact_fixture::require;
struct TransferGate {
    std::size_t bytes;
    bool fail_completion;
    bool armed = true;
    bool synchronized = false;
    void* staging = nullptr;
    std::promise<void> enqueued, release;
    std::shared_future<void> released = release.get_future().share();
};
thread_local TransferGate* transfer_gate = nullptr;
}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpy(void* destination, const void* source,
                                        std::size_t bytes, cudaMemcpyKind kind) {
    auto* gate = transfer_gate;
    if (!gate || !gate->armed || bytes != gate->bytes || kind != cudaMemcpyHostToDevice) {
        return __real_cudaMemcpy(destination, source, bytes, kind);
    }
    gate->armed = false;
    auto status = cudaMallocHost(&gate->staging, bytes);
    if (status != cudaSuccess) { return status; }
    // Pageable cudaMemcpy may finish Host staging before the actual DMA finishes.
    std::memcpy(gate->staging, source, bytes);
    status = cudaLaunchHostFunc(nullptr, [](void* data) {
        static_cast<TransferGate*>(data)->released.wait(); // No CUDA calls in callback.
    }, gate);
    if (status != cudaSuccess) { return status; }
    status = cudaMemcpyAsync(destination, gate->staging, bytes, kind, nullptr);
    gate->enqueued.set_value();
    return status;
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto status = __real_cudaStreamSynchronize(stream);
    auto* gate = transfer_gate;
    if (gate && !gate->armed && stream == nullptr) {
        gate->synchronized = true;
        if (status == cudaSuccess && std::exchange(gate->fail_completion, false)) {
            return cudaErrorInvalidValue;
        }
    }
    return status;
}

namespace {
void check(bool failure) {
    ninfer::test::qwen_fixture::ModelFixture fixture;
    ninfer::test::qwen_fixture::execution_fixture(fixture);
    EngineOptions options;
    options.artifact_path = fixture.file.entry;
    options.enable_model_suspend = true;
    options.max_context = 128;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk = 128;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.host_state_slots = 0;
    Engine engine(options);
    const auto suspended = engine.suspend();
    TransferGate gate{suspended.persistent_snapshot_bytes, failure};
    auto enqueued = gate.enqueued.get_future();
    std::exception_ptr error;
    std::jthread worker([&] {
        transfer_gate = &gate;
        try { (void)engine.resume(); } catch (...) { error = std::current_exception(); }
        transfer_gate = nullptr;
    });
    bool pending_is_closed = false;
    const bool reached = enqueued.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    if (reached) {
        // Give an incorrectly unsynchronized resume time to publish READY.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto pending = engine.residency();
        pending_is_closed = pending.state == ModelResidencyState::Resuming &&
                            pending.persistent_snapshot_bytes == gate.bytes && !engine.is_available();
    }
    gate.release.set_value();
    worker.join();
    // Also safe when testing a mutant that omitted the completion synchronization.
    (void)__real_cudaStreamSynchronize(nullptr);
    if (gate.staging) { (void)cudaFreeHost(gate.staging); }
    require(reached && pending_is_closed, "resume published READY before persistent H2D completed");
    require(gate.synchronized, "persistent H2D completion was not checked");
    const auto result = engine.residency();
    if (failure) {
        require(error != nullptr, "persistent completion failure was ignored");
        try { std::rethrow_exception(error); }
        catch (const ModelResidencyError& e) {
            require(e.kind() == ModelResidencyErrorKind::Failure, "wrong completion failure kind");
        }
        require(result.state == ModelResidencyState::Error && !engine.is_available() &&
                result.persistent_snapshot_bytes == gate.bytes &&
                result.last_error.find("complete persistent arena restore") != std::string::npos,
                "completion failure lost ERROR state or snapshot");
        bool rejected = false;
        try { (void)engine.submit(engine.prepare_tokens({65, 66}), RequestOptions{}); }
        catch (const RequestError& e) { rejected = e.kind() == RequestErrorKind::Unavailable; }
        require(rejected, "completion failure admitted inference");
    } else {
        if (error) { std::rethrow_exception(error); }
        require(result.state == ModelResidencyState::Ready && result.persistent_snapshot_bytes == 0 &&
                engine.is_available(), "completed restore did not publish READY");
        RequestOptions request;
        request.execution.requested_output_tokens = 3;
        require(engine.generate(engine.prepare_tokens({65, 66}), request).generated_token_ids.size() == 3,
                "generation failed after completed restore");
    }
}
}
int main() {
    int count = 0, vmm = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0) != CUDA_SUCCESS || !vmm) {
        return 77;
    }
    try {
        check(false); check(true);
        std::cout << "PASS persistent restore completion and asynchronous failure\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
