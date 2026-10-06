#include <fastmcpp/server/streamable_http_server.hpp>
#include <atomic>
#include <future>
#include <iostream>

using Server = fastmcpp::server::StreamableHttpServerWrapper;
using Json = fastmcpp::Json;
static void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        std::atomic<int> seconds{0};
        std::promise<void> started, release;
        auto released = release.get_future().share();
        Server server([&](const Json& request) {
            if (request["method"] == "blocked") { started.set_value(); released.wait(); }
            return Json{{"jsonrpc", "2.0"}, {"id", request.value("id", Json())}, {"result", Json::object()}};
        }, "127.0.0.1", 0);
        Server::SessionPolicy policy;
        policy.now = [&] { return Server::SessionClock::time_point{} + std::chrono::seconds(seconds.load()); };
        server.set_session_policy(policy);
        expect(server.start(), "transport failed to start");
        auto post = [&](std::string method, std::string session = "") {
            httplib::Client client("127.0.0.1", server.port());
            httplib::Headers headers{{"Accept", "application/json, text/event-stream"},
                {"MCP-Protocol-Version", "2025-11-25"}};
            if (!session.empty()) headers.emplace("Mcp-Session-Id", session);
            Json body{{"jsonrpc", "2.0"}, {"method", method}, {"id", 1}};
            if (method == "initialize") body["params"] = {{"protocolVersion", "2025-11-25"},
                {"capabilities", Json::object()}, {"clientInfo", {{"name", "expiry"}, {"version", "1"}}}};
            if (method == "notifications/initialized") body.erase("id");
            return client.Post("/mcp", headers, body.dump(), "application/json");
        };
        auto init = [&] {
            auto result = post("initialize");
            expect(result && result->status == 200, "initialize failed");
            return result->get_header_value("Mcp-Session-Id");
        };
        const auto abandoned = init();
        for (int i = 1; i < 1000; ++i) (void)init();
        expect(post("initialize")->status == 503, "session cap was not enforced");
        seconds = 60;
        const auto idle = init();
        expect(server.session_count() == 1, "abandoned sessions were not reclaimed at capacity");
        expect(post("ping", abandoned)->status == 404, "expired session was accepted");
        expect(post("notifications/initialized", idle)->status == 202, "initialized failed");
        seconds = 1859;
        expect(post("ping", idle)->status == 200, "live idle session expired early");
        seconds = 3659;
        expect(post("ping", idle)->status == 404, "idle timeout did not refresh from completed activity");
        const auto active = init();
        expect(post("notifications/initialized", active)->status == 202, "initialized failed");
        auto blocked = std::async(std::launch::async, [&] { return post("blocked", active); });
        started.get_future().wait();
        seconds = 10000;
        const auto another = init();
        const bool pinned = post("ping", active)->status == 200;
        release.set_value();
        expect(blocked.get()->status == 200, "active handler failed");
        expect(pinned, "active session expired during tool call");
        expect(post("ping", active)->status == 200, "session expired immediately after active tool call");
        expect(post("ping", another)->status == 200, "new session failed after expiry");
        server.stop();
        std::cout << "PASS: capacity recovery, initialization/idle expiry, activity refresh, active request pinning\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
