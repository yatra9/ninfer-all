#include "ninfer/engine.h"
#include "models/qwen3_5/execution_fixture.h"
#include "runtime/engine/model_instance.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <atomic>
#include <barrier>
#include <future>
#include <iostream>
#include <thread>
#include <utility>

#ifdef NINFER_TEST_WRAP_SAMPLING
namespace {
struct SubmissionGate {
    std::promise<void> reached;
    std::promise<void> release;
};
thread_local SubmissionGate* submission_gate = nullptr;
}
extern "C" ninfer::ResolvedSamplingParameters real_sampling(
    const ninfer::ModelSamplingDefaults&, ninfer::SamplingMode, const ninfer::SamplingOverrides&)
    asm("__real__ZN6ninfer7runtime16resolve_samplingERKNS_21ModelSamplingDefaultsENS_12SamplingModeERKNS_17SamplingOverridesE");
extern "C" ninfer::ResolvedSamplingParameters wrapped_sampling(
    const ninfer::ModelSamplingDefaults&, ninfer::SamplingMode, const ninfer::SamplingOverrides&)
    asm("__wrap__ZN6ninfer7runtime16resolve_samplingERKNS_21ModelSamplingDefaultsENS_12SamplingModeERKNS_17SamplingOverridesE");
extern "C" ninfer::ResolvedSamplingParameters wrapped_sampling(
    const ninfer::ModelSamplingDefaults& defaults, ninfer::SamplingMode mode,
    const ninfer::SamplingOverrides& overrides) {
    auto result = real_sampling(defaults, mode, overrides);
    if (auto* gate = std::exchange(submission_gate, nullptr)) {
        auto release = gate->release.get_future();
        gate->reached.set_value();
        release.wait();
    }
    return result;
}
#endif

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
void check_hybrid_shutdown(const ModelFixture& fixture) {
    const std::vector<TokenId> prompt(130, 65);
    RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.stop.include_model_defaults = false;
    for (const int mode : {0, 1, 2}) { // Ready, Suspended, failed resume.
        auto configured = options(fixture, true);
        configured.max_context = 256;
        configured.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
        configured.context_cache.mode = ContextCacheMode::Hybrid;
        configured.context_cache.host_cache_budget_bytes = 64ULL << 20;
        const auto file = fixture.file.directory / ("hybrid-shutdown-" + std::to_string(mode) + ".cache");
        configured.context_cache.hybrid.persistent_file = file;
        configured.context_cache.hybrid.persistent_identity = "suspend-shutdown-test";
        std::vector<TokenId> reference;
        {
            Engine engine(configured);
            reference = engine.generate(engine.prepare_tokens(prompt), request).generated_token_ids;
            require(reference.size() == 3 && engine.runtime_stats().hybrid_host_image_writes > 0,
                    "hybrid shutdown fixture did not produce Host snapshots");
            if (mode != 0) {
                bool suspended = false;
                for (int attempt = 0; attempt < 200 && !suspended; ++attempt) {
                    try {
                        const auto status = engine.suspend();
                        require(status.state == ModelResidencyState::Suspended && status.retained_device_bytes == 0,
                                "hybrid shutdown fixture retained Device backing");
                        suspended = true;
                    } catch (const ModelResidencyError& error) {
                        if (error.kind() != ModelResidencyErrorKind::Busy) { throw; }
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                require(suspended, "hybrid Host writes did not become idle");
            }
            if (mode == 2) {
                const auto timestamp = std::filesystem::last_write_time(fixture.file.entry);
                std::filesystem::last_write_time(fixture.file.entry, timestamp + std::chrono::seconds(10));
                try {
                    rejects<ModelResidencyError>([&] { (void)engine.resume(); }, "changed hybrid source was accepted");
                } catch (...) {
                    std::filesystem::last_write_time(fixture.file.entry, timestamp);
                    throw;
                }
                std::filesystem::last_write_time(fixture.file.entry, timestamp);
                require(engine.residency().state == ModelResidencyState::Error, "hybrid resume failure lost ERROR");
            }
        }
        if (mode == 2) {
            require(!std::filesystem::exists(file), "ERROR shutdown persisted potentially invalid cache");
            continue;
        }
        require(std::filesystem::exists(file) && std::filesystem::file_size(file) > 0,
                "clean hybrid shutdown did not persist Host cache");
        Engine loader(configured);
        const auto summary = loader.load_summary();
        require(summary.prefix_cache.restored && summary.prefix_cache.snapshots > 0 && summary.prefix_cache.blocks > 0,
                "hybrid Host snapshots were not restored after shutdown");
        require(loader.generate(loader.prepare_tokens(prompt), request).generated_token_ids == reference,
                "persisted hybrid cache changed generation");
    }
    std::cout << "PASS hybrid Ready/Suspended persistence and ERROR exclusion\n";
}
void run() {
    EngineOptions unsupported;
    unsupported.suspend_snapshot_memory = static_cast<SuspendSnapshotMemory>(255);
    rejects<std::invalid_argument>([&] { (void)runtime::normalize_engine_options(unsupported); }, "invalid snapshot mode accepted");
    unsupported.suspend_snapshot_memory = SuspendSnapshotMemory::Pinned;
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
        require(disabled.residency().persistent_snapshot_capacity_bytes == 0 &&
                !disabled.residency().persistent_snapshot_pinned, "disabled suspend allocated snapshot storage");
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
        // CPU-only submissions must share the admission boundary with suspend/resume.
        std::barrier immediate_start(2);
        std::atomic<bool> immediate_failed = false;
        std::jthread immediate_worker([&] {
            immediate_start.arrive_and_wait();
            for (int i = 0; i < 1000; ++i) {
                try {
                    RequestOptions zero;
                    zero.execution.requested_output_tokens = 0;
                    auto result = engine.submit(engine.prepare_tokens({65, 66}), zero).wait();
                    if (!result.generated_token_ids.empty() || result.finish_reason != FinishReason::OutputLimit) {
                        immediate_failed = true;
                    }
                } catch (const RequestError& error) {
                    if (error.kind() != RequestErrorKind::Unavailable) { immediate_failed = true; }
                } catch (...) { immediate_failed = true; }
            }
        });
        immediate_start.arrive_and_wait();
        for (int i = 0; i < 10; ++i) {
            (void)engine.suspend();
            RequestOptions zero;
            zero.execution.requested_output_tokens = 0;
            rejects<RequestError>([&] { (void)engine.submit(engine.prepare_tokens({65, 66}), zero); },
                                  "suspended CPU-only submission was admitted");
            (void)engine.resume();
        }
        immediate_worker.join();
        require(!immediate_failed, "CPU-only admission race failed");
#ifdef NINFER_TEST_WRAP_SAMPLING
        // Pause this submission after its initial availability check but before final admission.
        // Link wrapping is test-only; no hook or scheduling policy enters the product.
        SubmissionGate gate;
        auto reached = gate.reached.get_future();
        auto prompt = engine.prepare_tokens({65, 66});
        bool rejected = false;
        std::exception_ptr submission_error;
        std::jthread paused_submit([&, prompt = std::move(prompt)]() mutable {
            submission_gate = &gate;
            try {
                RequestOptions zero;
                zero.execution.requested_output_tokens = 0;
                (void)engine.submit(std::move(prompt), zero);
            } catch (const RequestError& error) {
                rejected = error.kind() == RequestErrorKind::Unavailable;
            } catch (...) { submission_error = std::current_exception(); }
            submission_gate = nullptr;
        });
        try {
            require(reached.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
                    "submission did not reach the test synchronization point");
            require(engine.suspend().state == ModelResidencyState::Suspended,
                    "suspend did not finish before final admission");
        } catch (...) {
            gate.release.set_value();
            paused_submit.join();
            throw;
        }
        gate.release.set_value();
        paused_submit.join();
        if (submission_error) { std::rethrow_exception(submission_error); }
        require(rejected, "zero-output submission bypassed final residency admission");
        (void)engine.resume();
        std::cout << "PASS deterministic zero-output admission rejection\n";
#endif
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
    auto& position_capacity = fixture.file.root["components"]["text"]["config"]["max_position_embeddings"];
    const auto original_position_capacity = position_capacity;
    position_capacity = 32768;
    fixture.file.write();
    for (const auto memory : {SuspendSnapshotMemory::Pinned, SuspendSnapshotMemory::Pageable}) {
        auto configured = options(fixture, true);
        configured.suspend_snapshot_memory = memory;
        configured.max_context = 32768;
        configured.kv_capacity = KvCapacityPolicy::explicit_capacity(32768);
        Engine engine(configured);
        const auto ready_bytes = engine.residency().retained_device_bytes;
        const bool pinned = memory == SuspendSnapshotMemory::Pinned;
        const auto snapshot_capacity = engine.memory_summary().sequence.capacity_bytes;
        require(engine.residency().persistent_snapshot_bytes == 0 &&
                engine.residency().persistent_snapshot_pinned == pinned &&
                engine.residency().persistent_snapshot_capacity_bytes == (pinned ? snapshot_capacity : 0),
                "startup snapshot allocation disagrees with selected mode");
        RequestOptions request;
        request.execution.requested_output_tokens = 3;
        request.execution.allow_prefix_reuse = false;
        const auto baseline = engine.generate(engine.prepare_tokens({65, 66}), request).generated_token_ids;
        for (int cycle = 0; cycle < 3; ++cycle) {
            const auto suspended = engine.suspend();
            require(suspended.state == ModelResidencyState::Suspended &&
                    suspended.persistent_snapshot_bytes >= 64ULL * 1024 * 1024 &&
                    suspended.persistent_snapshot_capacity_bytes == snapshot_capacity &&
                    suspended.retained_device_bytes == 0, "large snapshot did not release device backing");
            const auto resumed = engine.resume();
            require(resumed.state == ModelResidencyState::Ready && resumed.persistent_snapshot_bytes == 0 &&
                    resumed.retained_device_bytes == ready_bytes, "large snapshot restore retained RAM or lost backing");
            require(resumed.persistent_snapshot_capacity_bytes == (pinned ? snapshot_capacity : 0) &&
                    resumed.persistent_snapshot_pinned == pinned, "resume lost pinned buffer or retained pageable storage");
            require(engine.generate(engine.prepare_tokens({65, 66}), request).generated_token_ids == baseline &&
                    baseline.size() == 3, "large snapshot restore changed generation");
        }
        std::cout << "PASS large persistent snapshot\n";
    }
    position_capacity = original_position_capacity;
    fixture.file.write();
    check_hybrid_shutdown(fixture);
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
