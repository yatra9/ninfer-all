#pragma once
#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/program/vision_control.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct VisionUseSpan {
    std::uint32_t begin               = 0;
    std::uint32_t end                 = 0;
    std::uint32_t prepared_item_index = 0;
    std::uint32_t control_index       = 0;
    // Zero temporal_count denotes the complete ordinary media item. Local-video uses name one
    // planned temporal slice and its sequential payload chunk.
    std::int32_t temporal_begin       = 0;
    std::int32_t temporal_count       = 0;
    std::uint32_t local_chunk_index   = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_5::VisionControlPlan> control_plan;
    std::shared_ptr<const qwen3_5::VisionControl> control;
    std::vector<VisionUseSpan> uses;
    std::size_t max_merged_count = 0;
};

} // namespace ninfer::models::qwen3_5::detail
