#pragma once

#include "models/qwen3_5/frontend/local_video_plan.h"

#include <ninfer/types.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

namespace ninfer::models::qwen3_5 {
struct PreparedMediaPayload;
}

namespace ninfer::models::qwen3_5::frontend {
class MediaPreprocessCache;

// Process-wide frontend cache for immutable local-video sources. Entries are fingerprinted by
// size and modification time; replacing a file publishes a new source while active requests keep
// their old source alive long enough to fail its unchanged check safely.
class LocalVideoSourceCache {
public:
    explicit LocalVideoSourceCache(std::size_t maximum_entries = 8);
    ~LocalVideoSourceCache();
    [[nodiscard]] std::shared_ptr<media::local_video::VideoSource>
    acquire(const std::filesystem::path& path);

private:
    struct Entry;
    std::size_t maximum_entries_;
    std::uint64_t clock_ = 0;
    std::mutex mutex_;
    std::vector<Entry> entries_;
};

struct PreparedLocalVideoInput {
    std::shared_ptr<media::local_video::VideoSource> source;
    std::shared_ptr<MediaPreprocessCache> payload_account;
    media::local_video::Options reader_options;
    media::local_video::Stats index_stats;
    media::local_video::SourceStats source_stats;
    LocalVideoPromptPlan prompt;
};

class LocalVideoPayloadReader {
public:
    explicit LocalVideoPayloadReader(const PreparedLocalVideoInput& input,
                                     PreparationControl control = {});
    ~LocalVideoPayloadReader();
    LocalVideoPayloadReader(LocalVideoPayloadReader&&) noexcept;
    LocalVideoPayloadReader& operator=(LocalVideoPayloadReader&&) noexcept;

    LocalVideoPayloadReader(const LocalVideoPayloadReader&) = delete;
    LocalVideoPayloadReader& operator=(const LocalVideoPayloadReader&) = delete;

    // Chunks must be requested once in plan order. The returned payload contains patch rows only
    // for that temporal range and can be released as soon as its Vision output is encoded.
    [[nodiscard]] std::shared_ptr<qwen3_5::PreparedMediaPayload>
    read_chunk(std::size_t chunk_index);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Performs the one metadata/index pass shared by token counting and execution. It does not decode
// RGB frames or allocate a BF16 patch payload.
[[nodiscard]] PreparedLocalVideoInput
prepare_local_video_input(const OwnedLocalVideo& input, const PreparationControl& control,
                          std::uint64_t maximum_total_tokens,
                          std::uint64_t maximum_chunk_tokens,
                          const std::shared_ptr<LocalVideoSourceCache>& source_cache = {},
                          const std::shared_ptr<MediaPreprocessCache>& payload_account = {});

} // namespace ninfer::models::qwen3_5::frontend
