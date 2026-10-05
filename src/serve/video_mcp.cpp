#include "serve/video_mcp.h"
#include "video_mcp_schema.h"
#include "media/local_video/video_source_service.h"
#include "product/local_video/local_video_url.h"
#include <fastmcpp/protocol.hpp>
#include <fastmcpp/server/streamable_http_server.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <semaphore>
#include <unordered_map>

namespace ninfer::serve {
namespace {
using Json = nlohmann::json;
namespace lv = media::local_video;
namespace url = product::local_video;
struct ToolError : std::runtime_error {
    std::string code;
    ToolError(std::string c, std::string message) : std::runtime_error(message), code(std::move(c)) {}
};
// Validate the vocabulary used by the embedded tool schemas, without type coercion.
void validate(const Json& value, const Json& schema, const std::string& at = "arguments") {
    auto matches = [&](const std::string& type) {
        if (type == "null") return value.is_null();
        if (type == "object") return value.is_object();
        if (type == "array") return value.is_array();
        if (type == "string") return value.is_string();
        if (type == "boolean") return value.is_boolean();
        if (type == "number") return value.is_number();
        if (type == "integer") return value.is_number_integer();
        throw std::logic_error("Unsupported schema type: " + type);
    };
    if (schema.contains("type")) {
        const auto& types = schema["type"];
        bool valid = false;
        if (types.is_string()) valid = matches(types.get<std::string>());
        else for (const auto& type : types) valid |= matches(type.get<std::string>());
        if (!valid) throw ToolError("invalid_arguments", at + ": incorrect type");
    }
    if (schema.contains("enum")) {
        bool found = false;
        for (const auto& item : schema["enum"]) found |= item == value;
        if (!found) throw ToolError("invalid_arguments", at + ": unsupported value");
    }
    if (value.is_number()) {
        const auto number = value.get<double>();
        if (!std::isfinite(number) ||
            (schema.contains("minimum") && number < schema["minimum"].get<double>()) ||
            (schema.contains("maximum") && number > schema["maximum"].get<double>()) ||
            (schema.contains("exclusiveMinimum") && number <= schema["exclusiveMinimum"].get<double>()))
            throw ToolError("invalid_arguments", at + ": out of range");
        if (value.is_number_integer() && value.is_number_unsigned() &&
            value.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
            throw ToolError("invalid_arguments", at + ": integer overflow");
    }
    if (value.is_string() && schema.contains("minLength") &&
        value.get_ref<const std::string&>().size() < schema["minLength"].get<std::size_t>())
        throw ToolError("invalid_arguments", at + ": empty string");
    if (value.is_object()) {
        for (const auto& key : schema.value("required", Json::array()))
            if (!value.contains(key.get<std::string>()))
                throw ToolError("invalid_arguments", at + ": missing " + key.get<std::string>());
        const auto properties = schema.value("properties", Json::object());
        for (const auto& [key, item] : value.items()) {
            if (properties.contains(key)) validate(item, properties[key], at + "." + key);
            else if (!schema.value("additionalProperties", true))
                throw ToolError("invalid_arguments", at + ": unknown " + key);
        }
    }
    if (value.is_array() && schema.contains("items"))
        for (const auto& item : value) validate(item, schema["items"], at + "[]");
}
Json rate(int numerator, int denominator) {
    if (numerator <= 0 || denominator <= 0) return nullptr;
    return double(numerator) / denominator;
}
Json result(const Json& output) {
    return {{"content", Json::array({{{"type", "text"}, {"text", output.dump()}}})},
            {"structuredContent", output}, {"isError", false}};
}
Json tool_error(const std::string& code, const std::string& message) {
    return {{"content", Json::array({{{"type", "text"}, {"text", code + ": " + message}}})},
            {"isError", true}};
}
}

struct VideoMcpServer::Impl {
    std::filesystem::path root;
    std::string host;
    std::shared_ptr<spdlog::logger> logger;
    Json tools = Json::parse(video_mcp_schema)["tools"];
    std::shared_ptr<lv::VideoSourceService> sources = lv::shared_video_source_service();
    std::atomic<bool> stopping{false};
    std::counting_semaphore<4> slots{4};
    std::mutex calls_mutex;
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> calls;
    fastmcpp::server::HostOriginGuard guard;
    fastmcpp::server::StreamableHttpServerWrapper transport;

    Impl(std::filesystem::path r, std::string h, std::shared_ptr<spdlog::logger> log)
        : root(std::move(r)), host(std::move(h)), logger(std::move(log)),
          guard(fastmcpp::server::HostOriginGuardOptions{
              .mode = fastmcpp::server::HostOriginProtectionMode::Strict,
              .allowed_hosts = std::vector<std::string>{"localhost", "127.0.0.1", "::1",
                  (host == "0.0.0.0" || host == "::") ? "localhost" : host}}),
          transport([this](const Json& request) { return handle(request); }, host) {}

    Json call(const std::string& name, const Json& args, const lv::Options& options) {
        const auto text = args.at("path").get<std::string>();
        if (text.find("://") != std::string::npos || text.find('\0') != std::string::npos ||
            !std::filesystem::path(text).is_absolute())
            throw ToolError("invalid_path", "Use an absolute local path, not a URI");
        const auto path = url::authorize_local_path(text, root);
        options.checkpoint();
        const auto source = sources->acquire(path, options);
        struct Diagnostics {
            Impl& owner; std::filesystem::path path; std::shared_ptr<lv::VideoSource> source;
            ~Diagnostics() {
                if (!owner.logger || !owner.logger->should_log(spdlog::level::debug)) return;
                const auto cache = owner.sources->stats();
                const auto index = source->source_stats();
                owner.logger->debug("video MCP source={} metadata_hits={} metadata_probes={} evictions={} "
                    "index_builds={} index_reuses={} indexed_frames={} decoder_opens={} index_seconds={}",
                    path.string(), cache.metadata_hits, cache.metadata_probes, cache.evictions,
                    index.index_builds, index.index_reuses, index.index_scanned_frames, index.decoder_opens,
                    index.index_seconds);
            }
        } diagnostics{*this, path, source};
        options.checkpoint();
        if (name == "resolve_video_time") {
            const auto resolved = source->resolve_time(args.at("time_seconds").get<double>(),
                args.value("radius_frames", 2), options);
            if (logger) logger->debug("video MCP resolve target={} nearest_frame={} timestamp={}",
                resolved.requested_time_seconds, resolved.nearest.source_index, resolved.nearest.timestamp_seconds);
            Json frames = Json::array();
            for (const auto& frame : resolved.frames)
                frames.push_back({{"frame_number", frame.source_index},
                    {"timestamp_seconds", frame.timestamp_seconds},
                    {"delta_seconds", frame.timestamp_seconds - resolved.requested_time_seconds},
                    {"is_nearest", frame.source_index == resolved.nearest.source_index}});
            return result({{"requested_time_seconds", resolved.requested_time_seconds},
                {"nearest_frame_number", resolved.nearest.source_index},
                {"nearest_timestamp_seconds", resolved.nearest.timestamp_seconds}, {"frames", frames}});
        }
        const auto metadata = source->metadata(options);
        const auto& info = metadata.info;
        if (name == "get_video_metadata") {
            return result({{"path", path.generic_string()}, {"width", info.width}, {"height", info.height},
                {"duration_seconds", info.duration_seconds},
                {"frame_count", metadata.frame_count ? Json(*metadata.frame_count) : Json(nullptr)},
                {"frame_count_exact", metadata.frame_count.has_value()},
                {"average_fps", rate(info.average_fps_num, info.average_fps_den)},
                {"nominal_fps", rate(info.nominal_fps_num, info.nominal_fps_den)},
                {"variable_frame_rate", metadata.variable_frame_rate ? Json(*metadata.variable_frame_rate) : Json(nullptr)},
                {"has_audio", info.audio_stream_count > 0}, {"audio_stream_count", info.audio_stream_count},
                {"interlace_mode", metadata.interlace_mode},
                {"field_order", metadata.field_order ? Json(*metadata.field_order) : Json(nullptr)}});
        }
        url::LocalVideoSpec selection;
        selection.path = path;
        if (args.contains("frame")) {
            if (args.contains("start_frame") || args.contains("end_frame") || args.contains("skip_frame"))
                throw ToolError("invalid_frame_selection", "frame cannot be combined with start_frame, end_frame or skip_frame");
            selection.frame = args["frame"].get<std::int64_t>();
        }
        selection.start_frame = args.value("start_frame", std::int64_t(0));
        selection.skip_frame = args.value("skip_frame", std::int64_t(0));
        if (args.contains("end_frame")) selection.end_frame = args["end_frame"].get<std::int64_t>();
        selection.scale = args.value("scale", 1.0);
        const auto deint = args.value("deinterlace", std::string("auto"));
        selection.deinterlace = deint == "on" ? url::DeinterlaceMode::On :
                               deint == "off" ? url::DeinterlaceMode::Off : url::DeinterlaceMode::Auto;
        if (args.contains("bbox")) {
            const auto& box = args["bbox"];
            for (const auto& item : box.items())
                if (item.value().get<std::int64_t>() > std::numeric_limits<int>::max())
                    throw ToolError("invalid_bbox", "Crop coordinate overflow");
            selection.bbox = url::CropRect{box["x"], box["y"], box["width"], box["height"]};
            const auto& b = *selection.bbox;
            if (std::int64_t(b.x) + b.width > info.width || std::int64_t(b.y) + b.height > info.height)
                throw ToolError("invalid_bbox", "Crop extends outside source dimensions");
        }
        if (metadata.frame_count && (selection.frame.value_or(selection.start_frame) >= *metadata.frame_count ||
            (selection.end_frame && *selection.end_frame >= *metadata.frame_count)))
            throw ToolError("invalid_frame_range", "Selection exceeds exact source frame count");
        auto geometry_options = options;
        geometry_options.scale = selection.scale;
        geometry_options.alignment = 32;
        if (selection.bbox) {
            const auto& b = *selection.bbox;
            geometry_options.crop = media::local_video::Rect{b.x, b.y, b.width, b.height};
        }
        (void)media::local_video::output_geometry(info, geometry_options);
        const auto uri = url::build_local_video_url(selection);
        if (logger) logger->debug("video MCP inspect uri={}", uri);
        // Custom native resource: no encoded body is served. MIME selects the client's
        // native-video attachment route; FFmpeg discovers the actual source container.
        return {{"content", Json::array({
            {{"type", "text"}, {"text", args["instruction"].get<std::string>()}},
            {{"type", "resource_link"}, {"uri", uri}, {"name", path.filename().string()},
             {"description", (selection.frame ? "Single source frame as an image: " : "Selected native video frames: ") + args["instruction"].get<std::string>()},
             {"mimeType", "video/mp4"}}})}, {"isError", false}};
    }

    Json handle(const Json& request) {
        const auto id = request.value("id", Json(nullptr));
        const auto params = request.value("params", Json::object());
        const auto method = request.at("method").get<std::string>();
        auto reply = [&](const Json& body) { return Json{{"jsonrpc", "2.0"}, {"id", id}, {"result", body}}; };
        auto error = [&](int code, const std::string& message) {
            return Json{{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}};
        };
        const auto session = params.at("_meta").at("session_id").get<std::string>();
        if (method == "initialize") return reply({
            {"protocolVersion", fastmcpp::protocol::negotiate(params.at("protocolVersion").get<std::string>())},
            {"capabilities", {{"tools", {{"listChanged", false}}}}},
            {"serverInfo", {{"name", "ninfer-video-mcp"}, {"version", "1.0.0"}}}});
        if (method == "notifications/cancelled") {
            if (params.contains("requestId")) {
                std::lock_guard lock(calls_mutex);
                const auto found = calls.find(session + ":" + params["requestId"].dump());
                if (found != calls.end()) found->second->store(true);
            }
            return Json::object();
        }
        if (method.starts_with("notifications/")) return Json::object();
        if (method == "ping") return reply(Json::object());
        if (method == "tools/list") return reply({{"tools", tools}});
        if (method != "tools/call") return error(-32601, "Method not found");
        if (!params.contains("name") || !params["name"].is_string() ||
            (params.contains("arguments") && !params["arguments"].is_object()))
            return error(-32602, "Expected tool name and arguments object");
        const auto name = params["name"].get<std::string>();
        const Json* schema = nullptr;
        for (const auto& tool : tools) if (tool["name"] == name) schema = &tool;
        if (!schema) return error(-32602, "Unknown tool: " + name);
        const auto args = params.value("arguments", Json::object());
        if (!slots.try_acquire()) return reply(tool_error("busy", "Four video tool calls are already active"));
        struct Release { std::counting_semaphore<4>& slots; ~Release() { slots.release(); } } release{slots};
        const auto cancelled = std::make_shared<std::atomic<bool>>(false);
        const auto key = session + ":" + id.dump();
        {
            std::lock_guard lock(calls_mutex);
            if (!calls.emplace(key, cancelled).second) return error(-32600, "Duplicate active request id");
        }
        struct Remove {
            Impl& owner; std::string key;
            ~Remove() { std::lock_guard lock(owner.calls_mutex); owner.calls.erase(key); }
        } remove{*this, key};
        lv::Options options;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        options.checkpoint = [this, cancelled, deadline] {
            if (stopping.load() || cancelled->load()) throw ToolError("cancelled", "Video tool call cancelled");
            if (std::chrono::steady_clock::now() > deadline) throw ToolError("deadline_exceeded", "Video tool deadline exceeded");
        };
        try {
            validate(args, schema->at("inputSchema"));
            options.checkpoint();
            auto output = call(name, args, options);
            if (schema->contains("outputSchema")) validate(output.at("structuredContent"), schema->at("outputSchema"), "result");
            return reply(output);
        } catch (const ToolError& e) { return reply(tool_error(e.code, e.what())); }
          catch (const url::PathError& e) {
            const auto code = e.kind() == url::PathErrorKind::Disabled ? "local_media_disabled" :
                e.kind() == url::PathErrorKind::OutsideRoot ? "path_outside_root" :
                e.kind() == url::PathErrorKind::NotFound ? "video_not_found" : "invalid_path";
            return reply(tool_error(code, e.what()));
        } catch (const lv::Error& e) {
            return reply(tool_error(e.kind() == lv::ErrorKind::SourceChanged ? "source_changed" :
                e.kind() == lv::ErrorKind::ResourceLimit ? "resource_limit" : "invalid_video", e.what()));
        } catch (const std::invalid_argument& e) { return reply(tool_error("invalid_arguments", e.what())); }
          catch (const std::exception& e) { return reply(tool_error("video_processing_failed", e.what())); }
    }
};
VideoMcpServer::VideoMcpServer(std::filesystem::path root, std::string host,
                               std::shared_ptr<spdlog::logger> logger)
    : impl_(std::make_unique<Impl>(std::move(root), std::move(host), std::move(logger))) {}
VideoMcpServer::~VideoMcpServer() { stop(); }
void VideoMcpServer::register_routes(httplib::Server& server) { impl_->transport.register_routes(server); }
void VideoMcpServer::stop() { impl_->stopping.store(true); }
bool VideoMcpServer::reject_request(const httplib::Request& req, httplib::Response& res) const {
    if (req.path != "/mcp") return false;
    auto verdict = impl_->guard.check(req.get_header_value("Host"),
        req.has_header("Origin") ? std::optional<std::string>(req.get_header_value("Origin")) : std::nullopt,
        "http", impl_->host);
    if (req.has_header("Origin") &&
        fastmcpp::server::host_origin::normalize_origin(req.get_header_value("Origin")) !=
        fastmcpp::server::host_origin::normalize_origin("http://" + req.get_header_value("Host")))
        verdict = fastmcpp::server::HostOriginVerdict::RejectOrigin403;
    if (verdict == fastmcpp::server::HostOriginVerdict::Allow) return false;
    res.status = verdict == fastmcpp::server::HostOriginVerdict::RejectHost421 ? 421 : 403;
    res.set_content("{\"error\":\"Untrusted MCP Host or Origin\"}", "application/json");
    return true;
}
}
