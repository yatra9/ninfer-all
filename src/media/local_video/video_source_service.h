#pragma once

#include "media/local_video/video_pipeline.h"

#include <cstddef>
#include <memory>

namespace ninfer::media::local_video {

// Shares source metadata and immutable indexes; readers remain request-local.
// Callers authorize the path before acquire(), including on cache hits.
class VideoSourceService {
public:
    explicit VideoSourceService(std::size_t maximum_entries = 8);
    ~VideoSourceService();
    VideoSourceService(const VideoSourceService&) = delete;
    VideoSourceService& operator=(const VideoSourceService&) = delete;
    [[nodiscard]] std::shared_ptr<VideoSource> acquire(const std::filesystem::path& path);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::shared_ptr<VideoSourceService> shared_video_source_service();

} // namespace ninfer::media::local_video
