#pragma once

#include "models/qwen3_5/frontend/local_video_plan.h"

#include <ninfer/types.h>
#include "media/local_video/video_source_service.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5 {
struct PreparedMediaPayload;
}

namespace ninfer::models::qwen3_5::frontend {
class MediaPreprocessCache;
class MediaPayloadReservation;

struct PreparedLocalVideoInput {
    std::shared_ptr<media::local_video::VideoSource> source;
    std::shared_ptr<MediaPreprocessCache> payload_account;
    std::shared_ptr<MediaPayloadReservation> payload_reservation;
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
                          const std::shared_ptr<media::local_video::VideoSourceService>& source_cache = {},
                          const std::shared_ptr<MediaPreprocessCache>& payload_account = {});

} // namespace ninfer::models::qwen3_5::frontend
