#include "ninfer/engine.h"
#include "models/qwen3_5/model_fixture.h"
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

void execution_fixture(ModelFixture& fixture) {
    // Keep the byte tokenizer; use one attention and one GDN layer with supported dense geometry.
    std::vector<std::pair<std::string, std::string>> resources;
    for (const auto& object : fixture.file.root["objects"]) {
        if (object["kind"] != "resource") { continue; }
        const auto offset = object["offset"].get<std::size_t>();
        const auto bytes = object["bytes"].get<std::size_t>();
        resources.emplace_back(object["id"].get<std::string>(),
            std::string(reinterpret_cast<const char*>(fixture.file.payload.data() + offset), bytes));
    }
    fixture.file.payload.clear();
    fixture.file.root["objects"] = Json::array();
    fixture.file.root["uses"] = Json::array();
    fixture.file.root["bindings"] = Json::object();
    auto& config = fixture.file.root["components"]["text"]["config"];
    config["hidden_size"] = 5120;
    config["num_hidden_layers"] = 2;
    config["layer_types"] = {"full_attention", "linear_attention"};
    config["linear_num_key_heads"] = 16;
    config["linear_key_head_dim"] = 128;
    config["linear_num_value_heads"] = 48;
    config["linear_value_head_dim"] = 128;
    config["linear_conv_kernel_dim"] = 4;
    config["num_attention_heads"] = 24;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 256;
    config["intermediate_size"] = 17408;
    config["rope_parameters"]["mrope_section"] = {22, 21, 21};
    fixture.parameter("text/token_embedding", {272, 5120});
    fixture.file.root["bindings"]["text/output_head"] = {{"object", "text/token_embedding"}};
    fixture.use("text/output_head", "text/final_hidden");
    fixture.parameter("text/final_norm", {5120});
    const std::string p = "text/layers/0/";
    fixture.parameter(p + "input_norm", {5120});
    fixture.parameter(p + "post_attention_norm", {5120});
    fixture.tensor("attention-storage", {14336, 5120});
    const std::array roles = {"query", "key", "gate", "value"};
    const std::array rows = {0, 6144, 7168, 13312, 14336};
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto name = p + "attention/" + roles[i];
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "attention-storage"},
                                     {"range", {rows[i] * 5120, rows[i + 1] * 5120}}}})}};
        fixture.use(name, p + "mixer_input");
    }
    fixture.parameter(p + "attention/query_norm", {256});
    fixture.parameter(p + "attention/key_norm", {256});
    fixture.parameter(p + "attention/output", {5120, 6144}, p + "attention/gated_output");
    fixture.tensor("gate-up-storage", {34816, 5120}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    for (int i = 0; i < 2; ++i) {
        const auto name = p + (i == 0 ? "mlp/gate" : "mlp/up");
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gate-up-storage"},
                                     {"range", {i * 17408 * 5120, (i + 1) * 17408 * 5120}}}})}};
        fixture.use(name, p + "ffn_input");
    }
    fixture.tensor("down-storage", {5120, 17408}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    fixture.file.root["bindings"][p + "mlp/down"] = {{"object", "down-storage"}};
    fixture.use(p + "mlp/down", p + "mlp/product");
    const std::string g = "text/layers/1/";
    fixture.parameter(g + "input_norm", {5120});
    fixture.parameter(g + "post_attention_norm", {5120});
    fixture.tensor("gdn-input-storage", {16384, 5120}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    const std::array gdn_roles = {"query", "key", "value", "z"};
    const std::array gdn_rows = {0, 2048, 4096, 10240, 16384};
    for (std::size_t i = 0; i < gdn_roles.size(); ++i) {
        const auto name = g + "gdn/" + gdn_roles[i];
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gdn-input-storage"},
                                     {"range", {gdn_rows[i] * 5120, gdn_rows[i + 1] * 5120}}}})}};
        fixture.use(name, g + "mixer_input");
    }
    fixture.tensor("gdn-control-storage", {96, 5120});
    for (int i = 0; i < 2; ++i) {
        const auto name = g + (i == 0 ? "gdn/a_projection" : "gdn/b_projection");
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gdn-control-storage"},
                                     {"range", {i * 48 * 5120, (i + 1) * 48 * 5120}}}})}};
        fixture.use(name, g + "mixer_input");
    }
    for (const auto* name : {"a_log", "dt_bias"}) {
        fixture.tensor(g + "gdn/" + name, {48}, QType::FP32);
        fixture.file.root["bindings"][g + "gdn/" + name] = {{"object", g + "gdn/" + name}};
    }
    fixture.parameter(g + "gdn/convolution", {4, 10240});
    fixture.parameter(g + "gdn/norm", {128});
    fixture.parameter(g + "gdn/output", {5120, 6144}, g + "gdn/gated_output");
    for (const auto* name : {"gate", "up"}) {
        fixture.file.root["bindings"][g + "mlp/" + name] = fixture.file.root["bindings"][p + "mlp/" + name];
        fixture.use(g + "mlp/" + name, g + "ffn_input");
    }
    fixture.file.root["bindings"][g + "mlp/down"] = fixture.file.root["bindings"][p + "mlp/down"];
    fixture.use(g + "mlp/down", g + "mlp/product");
    for (const auto& [role, bytes] : resources) {
        if (role == "tokenizer_config.json") {
            auto tokenizer = Json::parse(bytes);
            tokenizer["add_bos_token"] = false;
            tokenizer["add_prefix_space"] = false;
            tokenizer["pad_token"] = "<|endoftext|>";
            for (auto& token : tokenizer["added_tokens_decoder"]) {
                if (token["content"] == "<think>" || token["content"] == "</think>") {
                    token["special"] = false;
                }
            }
            fixture.resource(role, tokenizer.dump());
        } else if (role == "tokenizer.json") {
            auto tokenizer = Json::parse(bytes);
            for (auto& token : tokenizer["added_tokens"]) {
                if (token["content"] == "<think>" || token["content"] == "</think>") {
                    token["special"] = false;
                }
            }
            fixture.resource(role, tokenizer.dump());
        } else {
            fixture.resource(role, bytes);
        }
    }
    fixture.file.root["files"][0]["payload_bytes"] = fixture.file.payload.size();
    fixture.file.write();
}
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
