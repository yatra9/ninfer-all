#include "media/local_video/video_source_service.h"

#include <mutex>
#include <unordered_map>

namespace ninfer::media::local_video {

struct VideoSourceService::Impl {
    struct Entry {
        std::filesystem::path path;
        std::uintmax_t size;
        std::filesystem::file_time_type modified;
        std::mutex mutex;
        std::weak_ptr<VideoSource> source;
        std::shared_ptr<VideoSource> retained;
        std::uint64_t last_use = 0;
    };
    std::size_t maximum_entries;
    std::uint64_t clock = 0;
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    explicit Impl(std::size_t maximum) : maximum_entries(maximum) {
        if (!maximum) throw std::invalid_argument("video source cache must retain at least one entry");
    }
};

VideoSourceService::VideoSourceService(std::size_t maximum_entries)
    : impl_(std::make_unique<Impl>(maximum_entries)) {}
VideoSourceService::~VideoSourceService() = default;

std::shared_ptr<VideoSource> VideoSourceService::acquire(const std::filesystem::path& requested) {
    const auto path = std::filesystem::canonical(requested);
    const auto size = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path);
    std::shared_ptr<Impl::Entry> entry;
    {
        std::lock_guard lock(impl_->mutex);
        auto& slot = impl_->entries[path.string()];
        if (!slot || slot->size != size || slot->modified != modified) {
            slot = std::make_shared<Impl::Entry>();
            slot->path = path;
            slot->size = size;
            slot->modified = modified;
        }
        entry = slot;
    }
    std::shared_ptr<VideoSource> source;
    {
        // Probe only while holding this source's lock, never the global LRU lock.
        std::lock_guard lock(entry->mutex);
        {
            std::lock_guard cache_lock(impl_->mutex);
            source = entry->source.lock();
        }
        if (!source) {
            source = std::make_shared<VideoSource>(path);
            if (std::filesystem::file_size(path) != size ||
                std::filesystem::last_write_time(path) != modified)
                throw Error(ErrorKind::SourceChanged, "video changed during metadata probe");
            std::lock_guard cache_lock(impl_->mutex);
            entry->source = source;
        }
    }
    {
        std::lock_guard lock(impl_->mutex);
        entry->last_use = ++impl_->clock;
        entry->retained = source;
        std::size_t retained = 0;
        std::shared_ptr<Impl::Entry> oldest;
        for (auto it = impl_->entries.begin(); it != impl_->entries.end();) {
            auto& candidate = it->second;
            if (candidate != entry && candidate.use_count() == 1 &&
                !candidate->retained && candidate->source.expired()) {
                it = impl_->entries.erase(it);
                continue;
            }
            if (candidate->retained) {
                ++retained;
                if (!oldest || candidate->last_use < oldest->last_use) oldest = candidate;
            }
            ++it;
        }
        if (retained > impl_->maximum_entries) oldest->retained.reset();
    }
    return source;
}

std::shared_ptr<VideoSourceService> shared_video_source_service() {
    static auto service = std::make_shared<VideoSourceService>();
    return service;
}

} // namespace ninfer::media::local_video
