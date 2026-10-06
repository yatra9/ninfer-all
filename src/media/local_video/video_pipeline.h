#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::media::local_video {
enum class ErrorKind { InvalidInput, ResourceLimit, SourceChanged };
class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}
    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }
private:
    ErrorKind kind_;
};
enum class Deinterlace { Auto, On, Off };
enum class ReadMode { Sequential, IndexedSeek };
struct Rect { int x, y, width, height; };
struct Options {
    std::int64_t start = 0;
    std::optional<std::int64_t> end; // inclusive, source display-frame order
    std::int64_t skip = 0;
    std::optional<Rect> crop;
    double scale = 1;
    int alignment = 1; // generic default; use 32 for the current Qwen geometry
    Deinterlace deinterlace = Deinterlace::Auto;
    ReadMode mode = ReadMode::IndexedSeek;
    std::int64_t max_scan_frames = 2'000'000;
    std::int64_t max_selected_frames = 10'000;
    std::int64_t max_pixels = 64LL * 1024 * 1024;
    // Called during indexing, decoding and output. Throw to cancel/deadline.
    std::function<void()> checkpoint;
};
struct Info {
    int width = 0, height = 0, stream_index = 0;
    int time_base_num = 0, time_base_den = 1;
    int fps_num = 0, fps_den = 1;
    int average_fps_num = 0, average_fps_den = 1;
    int nominal_fps_num = 0, nominal_fps_den = 1;
    int audio_stream_count = 0;
    double duration_seconds = 0;
    bool duration_known = false;
    std::int64_t reported_frames = 0;
    std::string codec, pixel_format, field_order, container;
    int sample_aspect_num = 0, sample_aspect_den = 1;
    bool has_display_transform = false;
};
struct Frame {
    std::int64_t source_index = 0, source_pts = 0;
    double timestamp_seconds = 0;
    int width = 0, height = 0;
    bool deinterlaced = false;
    std::vector<std::uint8_t> rgb;
};
struct OutputGeometry { int width, height; };
// Metadata-only calculation: does not decode or build a frame index.
OutputGeometry output_geometry(const Info& info, const Options& options);
struct Stats {
    std::int64_t indexed_frames = 0, decoded_frames = 0, selected_frames = 0;
    std::int64_t first_decoded_index = -1;
    bool seek_used = false, index_reached_eof = false;
    std::string timing = "not_scanned", seek_note;
    double index_seconds = 0, processing_seconds = 0;
};
struct Chunk {
    std::vector<Frame> frames;
    bool eof = false;
    Stats cumulative_stats;
};
struct SourceStats {
    std::int64_t metadata_probes = 1;
    std::int64_t decoder_opens = 0;
    std::int64_t index_builds = 0;
    std::int64_t index_scanned_frames = 0;
    std::int64_t index_reuses = 0;
    double index_seconds = 0;
};
struct FrameTiming {
    std::int64_t source_index = 0;
    std::int64_t source_pts = 0;
    double timestamp_seconds = 0;
};
struct Metadata {
    Info info;
    std::optional<std::int64_t> frame_count;
    std::optional<bool> variable_frame_rate;
    std::string interlace_mode = "unknown";
    std::optional<std::string> field_order;
};
struct TimeResolution {
    double requested_time_seconds = 0;
    FrameTiming nearest;
    std::vector<FrameTiming> frames;
};
struct VideoPlan {
    int width = 0, height = 0;
    std::vector<FrameTiming> selected_frames;
    Stats index_stats;
};
class VideoReader;
class VideoSource {
public:
    explicit VideoSource(std::filesystem::path path, const Options& options = {});
    ~VideoSource();
    VideoSource(VideoSource&&) noexcept;
    VideoSource& operator=(VideoSource&&) noexcept;
    VideoSource(const VideoSource&) = delete;
    VideoSource& operator=(const VideoSource&) = delete;
    const Info& info() const noexcept;
    Metadata metadata(const Options& options = {});
    TimeResolution resolve_time(double seconds, int radius_frames = 2,
                                const Options& options = {});
    VideoPlan plan(const Options& options);
    VideoReader create_reader(Options options);
    SourceStats source_stats() const;
private:
    friend class VideoReader;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
class VideoReader {
public:
    ~VideoReader();
    VideoReader(VideoReader&&) noexcept;
    VideoReader& operator=(VideoReader&&) noexcept;
    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;
    Chunk read_chunk(std::size_t max_frames);
private:
    friend class VideoSource;
    struct Impl;
    explicit VideoReader(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
// Local regular files only. No NInfer/CUDA, URL parser, subprocess, or JSON dependency.
Info inspect(const std::filesystem::path& path);
// Callback runs synchronously with backpressure. Ownership can be moved out by the caller.
// The library retains only decoder/filter history and a single output frame, not RGB history.
Stats process(const std::filesystem::path& path, const Options& options,
              const std::function<void(Frame&&)>& consume);
} // namespace ninfer::media::local_video
