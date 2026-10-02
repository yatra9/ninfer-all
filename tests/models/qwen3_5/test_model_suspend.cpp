#include "ninfer/engine.h"
#include "models/qwen3_5/execution_fixture.h"
#include "runtime/engine/model_instance.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <atomic>
#include <barrier>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::test::artifact_fixture;
using ninfer::test::qwen_fixture::ModelFixture;
using ninfer::test::qwen_fixture::execution_fixture;

EngineOptions options(const ModelFixture& fixture, bool enabled) {
    EngineOptions out;
    out.artifact_path = fixture.file.entry;
    out.enable_model_suspend = enabled;
    out.max_context = 128;
    out.kv_capacity = KvCapacityPolicy::explicit_capacity(128);
    out.prefill_chunk = 128;
    out.context_cache.host_kv_capacity_bytes = 0;
    out.context_cache.host_state_slots = 0;
    return out;
}
void expect_residency_error(Engine& engine, ModelResidencyErrorKind expected) {
    try { (void)engine.suspend(); }
    catch (const ModelResidencyError& error) { require(error.kind() == expected, "wrong residency error"); return; }
    throw std::runtime_error("expected residency error");
}
void run() {
    EngineOptions unsupported;
    unsupported.enable_model_suspend = true;
    unsupported.purpose = EnginePurpose::CausalScoring;
    rejects<std::invalid_argument>([&] { (void)runtime::normalize_engine_options(unsupported); }, "scoring suspend accepted");
    unsupported.purpose = EnginePurpose::Generation;
    unsupported.devices = {0, 1};
    rejects<std::invalid_argument>([&] { (void)runtime::normalize_engine_options(unsupported); }, "multi-GPU suspend accepted");
    unsupported.devices.clear();
    unsupported.wddm_evictable_budget = true;
    rejects<std::invalid_argument>([&] { (void)runtime::normalize_engine_options(unsupported); }, "WDDM budget suspend accepted");
    ModelFixture fixture;
    execution_fixture(fixture);
    {
        Engine disabled(options(fixture, false));
        require(!disabled.residency().enabled, "suspend must be opt-in");
        expect_residency_error(disabled, ModelResidencyErrorKind::Unsupported);
    }
    {
        Engine engine(options(fixture, true));
        const auto ready = engine.residency();
        require(ready.enabled && ready.retained_device_bytes > 0, "missing READY accounting");
        RequestOptions baseline_request;
        baseline_request.execution.requested_output_tokens = 3;
        const auto baseline = engine.generate(engine.prepare_tokens({65, 66}), baseline_request).generated_token_ids;
        RequestOptions busy_request;
        busy_request.execution.requested_output_tokens = 100;
        auto active = engine.submit(engine.prepare_tokens({65, 66}), busy_request);
        auto queued = engine.submit(engine.prepare_tokens({67, 68}), busy_request);
        expect_residency_error(engine, ModelResidencyErrorKind::Busy);
        require(engine.is_available(), "busy suspend closed admission permanently");
        require(active.wait().generated_token_ids.size() == 100 && queued.wait().generated_token_ids.size() == 100,
                "busy suspend discarded active or queued request");
        for (int iteration = 0; iteration < 10; ++iteration) {
            auto prompt = engine.prepare_tokens({65, 66});
            const auto suspended = engine.suspend();
            require(suspended.state == ModelResidencyState::Suspended &&
                    suspended.retained_device_bytes == 0 && suspended.persistent_snapshot_bytes > 0 &&
                    suspended.released_device_bytes == ready.retained_device_bytes, "incomplete suspend");
            require(engine.suspend().state == ModelResidencyState::Suspended, "suspend not idempotent");
            require(!engine.is_available(), "suspended engine accepts requests");
            RequestOptions request;
            request.execution.requested_output_tokens = 0;
            rejects<RequestError>([&] { (void)engine.submit(std::move(prompt), request); }, "immediate submit bypassed residency");
            const auto resumed = engine.resume();
            require(resumed.state == ModelResidencyState::Ready && resumed.persistent_snapshot_bytes == 0 &&
                    resumed.retained_device_bytes == ready.retained_device_bytes && resumed.weight_h2d_bytes > 0,
                    "incomplete resume");
            require(engine.resume().state == ModelResidencyState::Ready && engine.is_available(), "resume not idempotent");
            request.execution.requested_output_tokens = 3;
            const auto result = engine.generate(engine.prepare_tokens({65, 66}), request);
            require(result.generated_token_ids == baseline && baseline.size() == 3,
                    "generation changed with fresh workspace");
        }
        std::atomic<bool> failed_race = false;
        std::barrier start_race(3);
        const auto race = [&] {
            start_race.arrive_and_wait();
            for (int iteration = 0; iteration < 20; ++iteration) {
                try { (void)engine.suspend(); (void)engine.residency(); (void)engine.resume(); }
                catch (const ModelResidencyError& error) {
                    if (error.kind() != ModelResidencyErrorKind::Busy) { failed_race = true; }
                }
                catch (...) { failed_race = true; }
            }
        };
        std::thread first(race), second(race);
        start_race.arrive_and_wait();
        first.join(); second.join();
        require(!failed_race && engine.resume().state == ModelResidencyState::Ready,
                "concurrent management damaged residency");
        (void)engine.suspend();
        const auto timestamp = std::filesystem::last_write_time(fixture.file.entry);
        std::filesystem::last_write_time(fixture.file.entry, timestamp + std::chrono::seconds(10));
        rejects<ModelResidencyError>([&] { (void)engine.resume(); }, "changed source was accepted");
        const auto failed = engine.residency();
        require(failed.state == ModelResidencyState::Error && failed.persistent_snapshot_bytes > 0 &&
                !failed.last_error.empty() && !engine.is_available(), "resume failure lost diagnostics/snapshot");
        expect_residency_error(engine, ModelResidencyErrorKind::Failure);
        std::filesystem::last_write_time(fixture.file.entry, timestamp);
    }
    {
        Engine engine(options(fixture, true));
        (void)engine.suspend(); // Destruction must not implicitly restore or touch unmapped state.
    }
}
}
int main() {
    int count = 0, vmm = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 ||
        cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0) != CUDA_SUCCESS || !vmm) {
        return 77;
    }
    try { run(); std::cout << "PASS model suspend\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
