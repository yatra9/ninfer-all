#include "models/qwen3_5/execution_fixture.h"
#include "serve/http_server.h"
#include "serve/model_residency.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/logger.h>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
using namespace ninfer::serve;
using Json = nlohmann::json;
void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Listener {
    HttpServer& server;
    std::thread worker;
    explicit Listener(HttpServer& owner) : server(owner), worker([&owner] { (void)owner.listen(); }) {}
    ~Listener() { server.stop(); worker.join(); }
};
void run(bool enabled, const std::string& alias) {
    ninfer::test::qwen_fixture::ModelFixture fixture;
    ninfer::test::qwen_fixture::execution_fixture(fixture);
    ServeOptions options;
    options.artifact_path = fixture.file.entry;
    options.enable_model_suspend = enabled;
    options.max_context = 128;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk = 128;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.host_state_slots = 0;
    options.api_key = "test-key";
    options.model_id_override = alias;
    options.log_stats_interval_ms = 0;
    options.enable_webui = false;
    GenerationService service(options);
    auto logger = std::make_shared<spdlog::logger>("residency-test", std::make_shared<spdlog::sinks::null_sink_mt>());
    options.port = 28179;
    HttpServer server(options, logger);
    require(server.bind(), "cannot bind residency test listener");
    server.attach(service);
    Listener listener(server);
    httplib::Client client(options.host, options.port);
    client.set_connection_timeout(5);
    client.set_read_timeout(30);
    const httplib::Headers auth = {{"Authorization", "Bearer test-key"}};
    const std::string base = "/v1/models/" + alias;
    const auto get = [&](const std::string& path, int status) {
        auto response = client.Get(path, auth);
        require(response && response->status == status, "unexpected GET status for " + path +
                (response ? ": " + std::to_string(response->status) + " " + response->body : ": no response"));
        return Json::parse(response->body);
    };
    const auto post = [&](const std::string& path, const std::string& body, int status) {
        auto response = client.Post(path, auth, body, "application/json");
        require(response && response->status == status, "unexpected POST status for " + path +
                (response ? ": " + response->body : ": no response"));
        return Json::parse(response->body);
    };
    auto unauthenticated = client.Post(base + "/suspend", "{}", "application/json");
    require(unauthenticated && unauthenticated->status == 401, "management bypassed auth");
    const auto detail = get(base, 200);
    require(detail["object"] == "model" && detail["id"] == alias, "management shadowed model detail");
    const auto ready = get(base + "/residency", 200);
    require(ready["object"] == "model.residency" && ready["model"] == alias &&
            ready["state"] == "ready" && ready["enabled"] == enabled, "wrong alias residency status");
    require(get("/v1/models", 200)["data"][0]["id"] == alias, "model list lost public alias");
    get("/v1/models/unknown/residency", 404);
    post("/v1/models/unknown/suspend", "{}", 404);
    auto generation_body = Json::parse(
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":3,"temperature":0,"ignore_eos":true})");
    generation_body["model"] = alias;
    if (!enabled) {
        for (const auto* operation : {"/suspend", "/resume"}) {
            require(post(base + operation, "{}", 400)["error"]["code"] == "model_suspend_disabled",
                    "disabled management did not preserve opt-in contract");
        }
        require(post("/v1/chat/completions", generation_body.dump(), 200)["usage"]["completion_tokens"] == 3,
                "disabled suspend broke alias generation");
        return;
    }
    for (const auto* body : {"", "{", "[]", "null", "{\"memory_class\":\"weights\"}"}) {
        post(base + "/suspend", body, 400);
    }
    // A completed zero-output generation still has a service response reservation until released.
    GenerationRequest request;
    request.messages.push_back(ChatTurn{.role = ChatRole::User,
                                       .content = {ContentPart{.kind = ContentKind::Text, .text = "hello"}}});
    auto prepared = service.prepare(request, GenerationConsumerMode::Aggregate);
    (void)service.run(prepared, nullptr);
    require(post(base + "/suspend", "{}", 409)["error"]["code"] == "model_busy",
            "suspend ignored service response reservation");
    prepared = {};
    const auto suspended = post(base + "/suspend", "{}", 200);
    require(suspended["state"] == "suspended" && suspended["retained_device_bytes"] == 0 &&
            suspended["persistent_snapshot_bytes"].get<std::size_t>() > 0, "incomplete HTTP suspend");
    require(post(base + "/suspend", "{}", 200)["state"] == "suspended", "HTTP suspend not idempotent");
    get(base + "/residency", 200);
    get("/v1/models", 200);
    get("/stats", 200);
    get("/health", 503);
    for (const auto* path : {"/v1/chat/completions", "/v1/responses", "/v1/responses/input_tokens",
                            "/v1/messages", "/v1/messages/count_tokens"}) {
        const auto error = post(path, "{}", 503);
        // Anthropic uses its protocol-specific error shape; all gates run before body/SSE handling.
        require(error.contains("error"), "missing unavailable error");
        if (std::string_view(path).find("messages") == std::string_view::npos) {
            require(error["error"]["code"] == "model_suspended", "wrong unavailable code");
        }
    }
    require(service.residency().state == ModelResidencyState::Suspended, "inference implicitly resumed model");
    const auto resumed = post(base + "/resume", "{}", 200);
    require(resumed["state"] == "ready" && resumed["persistent_snapshot_bytes"] == 0 &&
            resumed["weight_h2d_bytes"].get<std::uint64_t>() > 0, "incomplete HTTP resume");
    post(base + "/resume", "{}", 200);
    get("/health", 200);
    const auto generation = post("/v1/chat/completions", generation_body.dump(), 200);
    require(generation["usage"]["completion_tokens"] == 3, "HTTP generation did not recover after resume");
    post(base + "/suspend", "{}", 200);
    const auto timestamp = std::filesystem::last_write_time(fixture.file.entry);
    std::filesystem::last_write_time(fixture.file.entry, timestamp + std::chrono::seconds(10));
    require(post(base + "/resume", "{}", 500)["error"]["code"] == "model_residency_error",
            "resume failure misclassified");
    const auto failed = get(base + "/residency", 200);
    require(failed["state"] == "error" && !failed["last_error"].get<std::string>().empty() &&
            failed["persistent_snapshot_bytes"].get<std::size_t>() > 0, "ERROR diagnostics unavailable");
    post(base + "/suspend", "{}", 500);
    post("/v1/responses", "{}", 503);
}
}
int main() {
    int count = 0, vmm = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0) != CUDA_SUCCESS || !vmm) {
        return 77;
    }
    try {
        for (const auto* alias : {"deployment/alias", "deployment/residency", "deployment/residency/residency"}) {
            for (const bool enabled : {true, false}) { run(enabled, alias); }
        }
        std::cout << "PASS residency HTTP aliases with suspend enabled/disabled\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
