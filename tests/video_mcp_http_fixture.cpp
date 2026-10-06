#include "serve/video_mcp.h"
#include <httplib.h>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ninfer::serve::VideoMcpServer mcp(argv[1], "127.0.0.1");
    httplib::Server server;
    server.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        return mcp.reject_request(req, res) ? httplib::Server::HandlerResponse::Handled :
                                            httplib::Server::HandlerResponse::Unhandled;
    });
    mcp.register_routes(server);
    server.Get("/health", [](const auto&, auto& res) { res.set_content("ok", "text/plain"); });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port < 0) return 3;
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    std::cout << port << std::endl;
    std::string line;
    std::getline(std::cin, line);
    mcp.stop();
    server.stop();
    listener.join();
}
