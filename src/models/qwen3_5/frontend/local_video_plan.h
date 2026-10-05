#pragma once

#include "media/local_video/video_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

enum class LocalVideoPlanErrorKind { InvalidInput, BudgetExceeded };

class LocalVideoPlanError final : public std::invalid_argument {
public:
    LocalVideoPlanError(LocalVideoPlanErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}
    [[nodiscard]] LocalVideoPlanErrorKind kind() const noexcept { return kind_; }

private:
    LocalVideoPlanErrorKind kind_;
};

struct LocalVideoChunkPlan {
    std::size_t temporal_begin = 0;
    std::size_t temporal_count = 0;
    std::size_t frame_begin = 0;
    std::size_t frame_count = 0;
    std::uint64_t token_begin = 0;
    std::uint64_t token_count = 0;
};

// Immutable prompt geometry derived without decoding RGB frames. Frame timings remain in source
// display order so execution can read only the ranges named by chunks.
struct LocalVideoPromptPlan {
    bool image = false;
    int width = 0;
    int height = 0;
    int grid_height = 0;
    int grid_width = 0;
    std::uint64_t tokens_per_temporal_group = 0;
    std::uint64_t total_tokens = 0;
    std::vector<media::local_video::FrameTiming> selected_frames;
    std::vector<double> timestamps;
    std::vector<LocalVideoChunkPlan> chunks;
};

// Qwen3.5 uses 16x16 spatial patches, two-frame temporal groups and 2x2 spatial merge.
// output dimensions must therefore be aligned to 32. The total and chunk limits are independent:
// the former is a request budget, while the latter bounds one Vision execution.
[[nodiscard]] LocalVideoPromptPlan
plan_local_video_prompt(int output_width, int output_height,
                        std::span<const media::local_video::FrameTiming> selected_frames,
                        std::uint64_t maximum_total_tokens,
                        std::uint64_t maximum_chunk_tokens);

} // namespace ninfer::models::qwen3_5::frontend
