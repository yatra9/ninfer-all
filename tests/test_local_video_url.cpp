#include "product/local_video/local_video_url.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace local_video = ninfer::product::local_video;

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Function> void rejects(Function&& function, std::string_view expected) {
    try {
        [[maybe_unused]] auto ignored = function();
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(expected) != std::string_view::npos,
                "unexpected validation error");
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

class Fixture final {
public:
    Fixture()
        : base(std::filesystem::temp_directory_path() /
               ("ninfer-local-video-url-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))),
          root(base / "media"), sibling(base / "media-other"), video(root / "clip one.mp4"),
          outside(sibling / "outside.mp4") {
        std::filesystem::create_directories(root);
        std::filesystem::create_directories(sibling);
        std::ofstream(video).put('v');
        std::ofstream(outside).put('x');
    }
    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(base, ignored);
    }

    std::filesystem::path base, root, sibling, video, outside;
};

} // namespace

int main() {
    try {
        require(!local_video::is_local_video_url("https://example/video.mp4"),
                "ordinary URL was routed locally");
        require(local_video::is_local_video_url("ninfer-video:///videos/a.mp4"),
                "local URL was not detected");

        const auto spec = local_video::parse_local_video_url(
            "ninfer-video:///videos/clip%20one.mp4?start_frame=100&end_frame=1000&"
            "skip_frame=2&bbox=320,180,640,360&scale=1.5&autotone=false&deinterlace=on");
        require(spec.path == "/videos/clip one.mp4" && spec.start_frame == 100 &&
                    spec.end_frame == 1000 && spec.skip_frame == 2 && spec.bbox &&
                    spec.bbox->x == 320 && spec.bbox->height == 360 && spec.scale == 1.5 &&
                    spec.deinterlace == local_video::DeinterlaceMode::On && !spec.autotone,
                "valid URL parsed incorrectly");

        rejects([] { return local_video::parse_local_video_url("ninfer-video://server/a.mp4"); },
                "authority");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a.mp4#fragment"); },
                "fragment");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a%2.mp4"); },
                "percent");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a%00.mp4"); }, "NUL");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?autoton=0"); },
                "unknown");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?scale=nan"); },
                "scale");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?skip_frame=9223372036854775807"); },
                "skip_frame");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?start_frame=2&end_frame=1"); },
                "end_frame");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?bbox=0,0,0,1"); },
                "positive");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?autotone=true"); },
                "not supported");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?deinterlace=yes"); },
                "deinterlace");
        rejects([] { return local_video::parse_local_video_url("ninfer-video:///a?scale=1&scale=2"); },
                "duplicate");

        Fixture fixture;
        require(local_video::authorize_local_path(fixture.video, fixture.root) ==
                    std::filesystem::canonical(fixture.video),
                "authorized path was not canonicalized");
        rejects([&] { return local_video::authorize_local_path(fixture.video, {}); }, "disabled");
        rejects([&] { return local_video::authorize_local_path(fixture.outside, fixture.root); },
                "outside");
        rejects([&] { return local_video::authorize_local_path(fixture.sibling, fixture.root); },
                "outside");

        std::error_code error;
        const auto escape = fixture.root / "escape.mp4";
        std::filesystem::create_symlink(fixture.outside, escape, error);
        require(!error, "failed to create symlink fixture");
        rejects([&] { return local_video::authorize_local_path(escape, fixture.root); }, "outside");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
