#include "models/qwen3_5/execution_fixture.h"
#include "serve/generation_service.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <future>
#include <iostream>
#include <thread>

namespace {
struct RestoreGate {
    std::promise<void> reached;
    std::promise<void> release;
    bool fail = false;
};
std::atomic<RestoreGate*> restore_gate = nullptr;
std::atomic<unsigned> restores = 0;
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (!stream) {
        if (auto* gate = restore_gate.exchange(nullptr)) {
            ++restores;
            auto release = gate->release.get_future();
            gate->reached.set_value();
            release.wait();
            const auto result = __real_cudaStreamSynchronize(stream);
            return gate->fail && result == cudaSuccess ? cudaErrorInvalidValue : result;
        }
    }
    return __real_cudaStreamSynchronize(stream);
}

namespace {
using namespace ninfer;
using namespace ninfer::serve;
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
void run() {
    ninfer::test::qwen_fixture::ModelFixture fixture;
    ninfer::test::qwen_fixture::execution_fixture(fixture);
    for (int mode : {0, 1, 2}) { // Cancelled leader, wait deadline, restore failure.
        ServeOptions options;
        options.artifact_path = fixture.file.entry;
        options.enable_model_suspend = true;
        options.max_context = 128;
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(128);
        options.prefill_chunk = 128;
        options.context_cache.host_kv_capacity_bytes = 0;
        options.context_cache.host_state_slots = 0;
        options.max_pending_requests = 1;
        options.pending_timeout_ms = mode == 1 ? 250 : 5000;
        GenerationService service(options);
        GenerationRequest request;
        request.messages.push_back(ChatTurn{.role = ChatRole::User,
            .content = {ContentPart{.kind = ContentKind::Text, .text = "hello"}}});
        request.max_tokens = 3;
        request.ignore_eos = true;
        require(service.suspend().auto_resume, "default policy is not automatic");
        RestoreGate gate;
        gate.fail = mode == 2;
        auto reached = gate.reached.get_future();
        restores = 0;
        std::atomic<bool> cancel = false;
        const auto prepare = [&](bool leader) {
            try {
                auto prepared = service.prepare(request, GenerationConsumerMode::Aggregate, {},
                    [&] { return leader && cancel.load(); });
                require(service.run(prepared, nullptr).completion_tokens == 3, "generation did not run after restore");
                return std::string("ok");
            } catch (const ApiException& error) { return error.error().code; }
        };
        restore_gate = &gate;
        auto leader = std::async(std::launch::async, prepare, true);
        std::future<std::string> follower;
        try {
            require(reached.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
                    "automatic resume did not reach restore gate");
            require(service.residency().state == ModelResidencyState::Resuming,
                    "restore did not publish RESUMING");
            follower = std::async(std::launch::async, prepare, false);
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (service.admitted_requests() != 2 && std::chrono::steady_clock::now() < limit) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(service.admitted_requests() == 2, "resume waiter did not reserve capacity");
            require(prepare(false) == "server_overloaded", "resume waiters bypassed capacity limit");
            try {
                (void)service.suspend(false);
                throw std::runtime_error("policy changed during RESUMING");
            } catch (const ModelResidencyError& error) {
                require(error.kind() == ModelResidencyErrorKind::Busy, "wrong policy update error");
            }
            if (mode == 0) { cancel = true; }
            if (mode == 1) {
                require(follower.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
                        "resume waiter did not obey pending deadline");
            }
        } catch (...) {
            restore_gate = nullptr;
            gate.release.set_value();
            leader.wait();
            if (follower.valid()) { follower.wait(); }
            throw;
        }
        gate.release.set_value();
        const auto first = leader.get();
        const auto second = follower.get();
        require(restores == 1, "concurrent requests performed multiple restores");
        require(service.admitted_requests() == 0, "resume requests leaked reservations");
        if (mode == 0) {
            require(first == "client_disconnected" && second == "ok", "leader cancellation cancelled shared restore");
        } else if (mode == 1) {
            require(first == "request_queue_timeout" && second == "request_queue_timeout",
                    "restore wait did not account for request deadline");
        } else {
            require(first == "model_residency_error" && second == "model_residency_error",
                    "restore failure was not shared by waiting requests");
            require(service.residency().state == ModelResidencyState::Error,
                    "restore failure did not retain ERROR");
            require(prepare(false) == "model_error", "later request retried failed automatic resume");
        }
        if (mode != 2) {
            require(service.residency().state == ModelResidencyState::Ready && prepare(false) == "ok",
                    "cancelled or timed out request damaged shared restore");
        }
    }
}
}
int main() {
    int count = 0, vmm = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0) != CUDA_SUCCESS || !vmm) {
        return 77;
    }
    try { run(); std::cout << "PASS bounded shared auto resume, cancellation, deadline and failure\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
