#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::product::local_video {

enum class DeinterlaceMode { Auto, On, Off };
enum class PathErrorKind { Disabled, InvalidRoot, NotFound, OutsideRoot, NotRegularFile };
class PathError final : public std::invalid_argument {
public:
    PathError(PathErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}
    [[nodiscard]] PathErrorKind kind() const noexcept { return kind_; }
private:
    PathErrorKind kind_;
};

struct CropRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct LocalVideoSpec {
    std::filesystem::path path;
    std::int64_t start_frame = 0;
    std::optional<std::int64_t> end_frame;
    std::int64_t skip_frame = 0;
    std::optional<CropRect> bbox;
    double scale = 1.0;
    bool autotone = false;
    DeinterlaceMode deinterlace = DeinterlaceMode::Auto;
};

[[nodiscard]] bool is_local_video_url(std::string_view value) noexcept;

// Parses and validates every URL field and query value. The returned path is still the requested
// absolute path; authorize_local_path must be called before opening it.
[[nodiscard]] LocalVideoSpec parse_local_video_url(std::string_view value);

// Requires an explicitly configured root, resolves symlinks, requires a regular file, and returns
// the canonical path only when it is a component-wise descendant of the canonical root.
[[nodiscard]] std::filesystem::path authorize_local_path(
    const std::filesystem::path& requested, const std::filesystem::path& configured_root);

} // namespace ninfer::product::local_video
