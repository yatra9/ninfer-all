#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
// Host paths are lexical references: they need not exist in the server's filesystem.
// Filesystem canonicalization and local-media-root authorization follow translation.
class ReferencePathMaps {
    struct Path { std::string root; std::vector<std::string> parts; bool windows = false; };
    struct Mapping { Path host; std::string container; };
    std::vector<Mapping> maps_;
    static std::string fold(std::string value) {
        for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        return value;
    }
    static Path parse(std::string text) {
        if (text.empty() || text.find('\0') != std::string::npos)
            throw std::invalid_argument("Reference path must be an absolute local path");
        Path result;
        const bool drive = text.size() >= 3 &&
            ((text[0] >= 'a' && text[0] <= 'z') || (text[0] >= 'A' && text[0] <= 'Z')) &&
            text[1] == ':' && (text[2] == '/' || text[2] == '\\');
        const bool unc = text.starts_with("\\\\");
        result.windows = drive || unc;
        if (result.windows) std::replace(text.begin(), text.end(), '\\', '/');
        std::size_t begin = 0;
        if (drive) { result.root = fold(text.substr(0, 2)); begin = 3; }
        else if (unc) {
            const auto server_end = text.find('/', 2);
            const auto share_end = server_end == std::string::npos ? server_end : text.find('/', server_end + 1);
            const auto end = share_end == std::string::npos ? text.size() : share_end;
            if (server_end == std::string::npos || server_end == 2 || end == server_end + 1 ||
                text.substr(2, server_end - 2) == "?" || text.substr(2, server_end - 2) == ".")
                throw std::invalid_argument("Reference UNC path must name a server and share");
            result.root = fold(text.substr(0, end)); begin = end;
        } else if (text.front() == '/') { result.root = "/"; begin = 1; }
        else throw std::invalid_argument("Reference path must be an absolute local path");
        while (begin < text.size()) {
            const auto end = text.find('/', begin);
            auto part = text.substr(begin, end == std::string::npos ? end : end - begin);
            if (part == "..") {
                if (result.parts.empty()) throw std::invalid_argument("Reference path escapes its root");
                result.parts.pop_back();
            } else if (!part.empty() && part != ".") result.parts.push_back(std::move(part));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return result;
    }
    static bool prefix(const Path& path, const Path& host) {
        if (path.windows != host.windows || path.root != host.root || path.parts.size() < host.parts.size())
            return false;
        for (std::size_t i = 0; i < host.parts.size(); ++i)
            if ((path.windows ? fold(path.parts[i]) : path.parts[i]) !=
                (host.windows ? fold(host.parts[i]) : host.parts[i])) return false;
        return true;
    }
    static std::string append(std::string root, const Path& path, std::size_t first) {
        for (std::size_t i = first; i < path.parts.size(); ++i) {
            if (root.back() != '/') root += '/';
            root += path.parts[i];
        }
        return root;
    }
public:
    void add(std::string host, std::string container) {
        auto source = parse(std::move(host));
        auto target = parse(std::move(container));
        if (target.windows) throw std::invalid_argument("CONTAINER_DIR must be an absolute POSIX path");
        for (const auto& mapping : maps_)
            if (source.parts.size() == mapping.host.parts.size() && prefix(source, mapping.host))
                throw std::invalid_argument("Duplicate reference path mapping");
        maps_.push_back({std::move(source), append("/", target, 0)});
    }
    [[nodiscard]] std::string translate(std::string_view text) const {
        const auto path = parse(std::string(text));
        const Mapping* best = nullptr;
        for (const auto& mapping : maps_)
            if (prefix(path, mapping.host) && (!best || mapping.host.parts.size() > best->host.parts.size()))
                best = &mapping;
        if (best) return append(best->container, path, best->host.parts.size());
        if (path.windows) throw std::invalid_argument("Windows reference path has no matching --reference-path-map");
        return append("/", path, 0);
    }
};
} // namespace ninfer::serve
