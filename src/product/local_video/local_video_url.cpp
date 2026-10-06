#include "product/local_video/local_video_url.h"

#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ninfer::product::local_video {
namespace {

constexpr std::string_view kScheme = "ninfer-video://";

int hex(char value) {
    if (value >= '0' && value <= '9') { return value - '0'; }
    if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
    if (value >= 'A' && value <= 'F') { return value - 'A' + 10; }
    return -1;
}

std::string percent_decode(std::string_view encoded) {
    std::string decoded;
    decoded.reserve(encoded.size());
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        char value = encoded[i];
        if (value == '%') {
            if (i + 2 >= encoded.size()) {
                throw std::invalid_argument("invalid ninfer-video percent encoding");
            }
            const int high = hex(encoded[i + 1]);
            const int low  = hex(encoded[i + 2]);
            if (high < 0 || low < 0) {
                throw std::invalid_argument("invalid ninfer-video percent encoding");
            }
            value = static_cast<char>((high << 4) | low);
            i += 2;
        }
        if (value == '\0') { throw std::invalid_argument("ninfer-video URL contains NUL"); }
        decoded.push_back(value);
    }
    return decoded;
}

std::int64_t integer(std::string_view name, std::string_view value) {
    if (value.empty()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    std::int64_t result = 0;
    const auto parsed   = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return result;
}

int crop_integer(std::string_view name, std::string_view value) {
    const std::int64_t parsed = integer(name, value);
    if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return static_cast<int>(parsed);
}

CropRect parse_bbox(std::string_view value) {
    std::array<std::string_view, 4> fields;
    std::size_t begin = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const std::size_t comma = value.find(',', begin);
        if ((i + 1 < fields.size() && comma == std::string_view::npos) ||
            (i + 1 == fields.size() && comma != std::string_view::npos)) {
            throw std::invalid_argument("bbox must be x,y,width,height");
        }
        fields[i] = value.substr(begin, comma == std::string_view::npos ? comma : comma - begin);
        begin = comma == std::string_view::npos ? value.size() : comma + 1;
    }
    CropRect result{crop_integer("bbox x", fields[0]), crop_integer("bbox y", fields[1]),
                    crop_integer("bbox width", fields[2]), crop_integer("bbox height", fields[3])};
    if (result.width == 0 || result.height == 0) {
        throw std::invalid_argument("bbox width and height must be positive");
    }
    return result;
}

double positive_scale(std::string_view value) {
    if (value.empty()) { throw std::invalid_argument("scale must be a finite positive number"); }
    double result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result,
                                        std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !std::isfinite(result) || result <= 0) {
        throw std::invalid_argument("scale must be a finite positive number");
    }
    return result;
}

bool is_descendant(const std::filesystem::path& path, const std::filesystem::path& root) {
    auto path_part = path.begin();
    for (auto root_part = root.begin(); root_part != root.end(); ++root_part, ++path_part) {
        if (path_part == path.end() || *path_part != *root_part) { return false; }
    }
    return true;
}

} // namespace

bool is_local_video_url(std::string_view value) noexcept { return value.starts_with(kScheme); }

LocalVideoSpec parse_local_video_url(std::string_view value) {
    if (!is_local_video_url(value)) { throw std::invalid_argument("invalid ninfer-video URL"); }
    value.remove_prefix(kScheme.size());
    if (value.empty() || value.front() != '/') {
        throw std::invalid_argument("ninfer-video URL must not contain an authority");
    }
    if (value.find('#') != std::string_view::npos) {
        throw std::invalid_argument("ninfer-video URL must not contain a fragment");
    }

    const std::size_t query_at = value.find('?');
    const std::string path_text = percent_decode(value.substr(0, query_at));
    LocalVideoSpec result;
    result.path = std::filesystem::path(path_text);
    if (path_text.empty() || !result.path.is_absolute()) {
        throw std::invalid_argument("ninfer-video path must be absolute");
    }
    if (query_at == std::string_view::npos) { return result; }

    value.remove_prefix(query_at + 1);
    if (value.empty()) { throw std::invalid_argument("ninfer-video query must not be empty"); }
    std::unordered_set<std::string> seen;
    while (!value.empty()) {
        const std::size_t amp = value.find('&');
        const std::string_view pair = value.substr(0, amp);
        if (pair.empty()) { throw std::invalid_argument("ninfer-video query contains an empty parameter"); }
        const std::size_t equals = pair.find('=');
        if (equals == std::string_view::npos || equals == 0) {
            throw std::invalid_argument("ninfer-video parameter must contain a value");
        }
        const std::string name = percent_decode(pair.substr(0, equals));
        const std::string decoded = percent_decode(pair.substr(equals + 1));
        if (!seen.insert(name).second) {
            throw std::invalid_argument("duplicate ninfer-video parameter: " + name);
        }
        const std::string_view parameter(decoded);
        if (name == "frame") {
            result.frame = integer(name, parameter);
            if (*result.frame < 0) { throw std::invalid_argument("frame must be nonnegative"); }
        } else if (name == "start_frame") {
            result.start_frame = integer(name, parameter);
            if (result.start_frame < 0) { throw std::invalid_argument("start_frame must be nonnegative"); }
        } else if (name == "end_frame") {
            result.end_frame = integer(name, parameter);
            if (*result.end_frame < 0) { throw std::invalid_argument("end_frame must be nonnegative"); }
        } else if (name == "skip_frame") {
            result.skip_frame = integer(name, parameter);
            if (result.skip_frame < 0 || result.skip_frame == std::numeric_limits<std::int64_t>::max()) {
                throw std::invalid_argument("skip_frame is out of range");
            }
        } else if (name == "bbox") {
            result.bbox = parse_bbox(parameter);
        } else if (name == "scale") {
            result.scale = positive_scale(parameter);
        } else if (name == "autotone") {
            if (parameter == "0" || parameter == "false") {
                result.autotone = false;
            } else if (parameter == "1" || parameter == "true") {
                throw std::invalid_argument("autotone is not supported");
            } else {
                throw std::invalid_argument("autotone must be one of: 0, false");
            }
        } else if (name == "deinterlace") {
            if (parameter == "auto") result.deinterlace = DeinterlaceMode::Auto;
            else if (parameter == "on") result.deinterlace = DeinterlaceMode::On;
            else if (parameter == "off") result.deinterlace = DeinterlaceMode::Off;
            else throw std::invalid_argument("deinterlace must be one of: auto, on, off");
        } else {
            throw std::invalid_argument("unknown ninfer-video parameter: " + name);
        }
        if (amp == std::string_view::npos) { break; }
        value.remove_prefix(amp + 1);
        if (value.empty()) { throw std::invalid_argument("ninfer-video query contains an empty parameter"); }
    }
    if (result.frame && (seen.contains("start_frame") || seen.contains("end_frame") ||
                         seen.contains("skip_frame"))) {
        throw std::invalid_argument("frame cannot be combined with start_frame, end_frame or skip_frame");
    }
    if (result.end_frame && *result.end_frame < result.start_frame) {
        throw std::invalid_argument("end_frame must be >= start_frame");
    }
    return result;
}

std::filesystem::path authorize_local_path(const std::filesystem::path& requested,
                                           const std::filesystem::path& configured_root) {
    if (configured_root.empty()) {
        throw PathError(PathErrorKind::Disabled,
                        "ninfer-video is disabled without --local-media-root");
    }
    std::error_code error;
    const std::filesystem::path root = std::filesystem::canonical(configured_root, error);
    if (error || !std::filesystem::is_directory(root, error) || error) {
        throw PathError(PathErrorKind::InvalidRoot,
                        "local media root is not an accessible directory");
    }
    const std::filesystem::path path = std::filesystem::canonical(requested, error);
    if (error) { throw PathError(PathErrorKind::NotFound, "local video path does not exist"); }
    if (!is_descendant(path, root)) {
        throw PathError(PathErrorKind::OutsideRoot,
                        "local video path is outside configured media root");
    }
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw PathError(PathErrorKind::NotRegularFile,
                        "local video path is not a regular file");
    }
    return path;
}

std::string build_local_video_url(const LocalVideoSpec& spec) {
    if (spec.frame && (spec.start_frame != 0 || spec.end_frame || spec.skip_frame != 0))
        throw std::invalid_argument("frame cannot be combined with start_frame, end_frame or skip_frame");
    if (!spec.path.is_absolute()) throw std::invalid_argument("local video path must be absolute");
    std::string result(kScheme);
    constexpr char digits[]="0123456789ABCDEF";
    for (const unsigned char c:spec.path.generic_string()) {
        if ((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') ||
            c=='/' || c=='-' || c=='_' || c=='.' || c=='~') result+=char(c);
        else { result+='%'; result+=digits[c>>4]; result+=digits[c&15]; }
    }
    char separator='?';
    const auto add=[&](std::string_view name,const std::string& value) {
        result+=separator; separator='&'; result+=name; result+='='; result+=value;
    };
    if (spec.frame) add("frame",std::to_string(*spec.frame));
    if (spec.start_frame!=0) add("start_frame",std::to_string(spec.start_frame));
    if (spec.end_frame) add("end_frame",std::to_string(*spec.end_frame));
    if (spec.skip_frame!=0) add("skip_frame",std::to_string(spec.skip_frame));
    if (spec.bbox) {
        const auto& b=*spec.bbox;
        add("bbox",std::to_string(b.x)+","+std::to_string(b.y)+","+
                   std::to_string(b.width)+","+std::to_string(b.height));
    }
    if (spec.scale!=1) {
        std::array<char,64> buffer;
        const auto encoded=std::to_chars(buffer.data(),buffer.data()+buffer.size(),spec.scale);
        if (encoded.ec!=std::errc{}) throw std::invalid_argument("scale could not be encoded");
        add("scale",std::string(buffer.data(),encoded.ptr));
    }
    switch (spec.deinterlace) {
    case DeinterlaceMode::Auto: break;
    case DeinterlaceMode::On: add("deinterlace","on"); break;
    case DeinterlaceMode::Off: add("deinterlace","off"); break;
    default: throw std::invalid_argument("invalid deinterlace mode");
    }
    if (spec.autotone) throw std::invalid_argument("autotone is not supported");
    (void)parse_local_video_url(result);
    return result;
}

} // namespace ninfer::product::local_video
