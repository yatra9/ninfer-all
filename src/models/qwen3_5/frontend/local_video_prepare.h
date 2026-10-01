#pragma once

#include "models/qwen3_5/frontend/local_video_plan.h"

#include <ninfer/types.h>

#include <cstdint>
#include <memory>

namespace ninfer::models::qwen3_5::frontend {

struct PreparedLocalVideoInput {
    std::shared_ptr<media::local_video::VideoSource> source;
    media::local_video::Options reader_options;
    LocalVideoPromptPlan prompt;
};

// Performs the one metadata/index pass shared by token counting and execution. It does not decode
// RGB frames or allocate a BF16 patch payload.
[[nodiscard]] PreparedLocalVideoInput
prepare_local_video_input(const OwnedLocalVideo& input, const PreparationControl& control,
                          std::uint64_t maximum_total_tokens,
                          std::uint64_t maximum_chunk_tokens);

} // namespace ninfer::models::qwen3_5::frontend
