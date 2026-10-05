#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <httplib.h>
namespace spdlog { class logger; }

namespace ninfer::serve {
// CPU-only tools; attaches to the existing listener and shares native video sources.
class VideoMcpServer {
public:
    explicit VideoMcpServer(std::filesystem::path root, std::string host,
                            std::shared_ptr<spdlog::logger> logger = {});
    ~VideoMcpServer();
    void register_routes(httplib::Server& server);
    bool reject_request(const httplib::Request& request, httplib::Response& response) const;
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
