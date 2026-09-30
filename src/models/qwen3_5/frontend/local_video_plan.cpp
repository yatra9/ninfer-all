#include "models/qwen3_5/frontend/local_video_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::frontend {
namespace {

constexpr int kPatch = 16;
constexpr int kTemporal = 2;
constexpr int kMerge = 2;
constexpr int kAlignment = kPatch * kMerge;

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::invalid_argument(std::string("local video ") + label + " overflow");
    }
    return left * right;
}

} // namespace

LocalVideoPromptPlan
plan_local_video_prompt(int output_width, int output_height,
                        std::span<const media::local_video::FrameTiming> selected_frames,
                        std::uint64_t maximum_total_tokens,
                        std::uint64_t maximum_chunk_tokens) {
    if (output_width < kAlignment || output_height < kAlignment ||
        output_width % kAlignment != 0 || output_height % kAlignment != 0) {
        throw std::invalid_argument("local video output dimensions must be positive multiples of 32");
    }
    if (selected_frames.empty()) {
        throw std::invalid_argument("local video selection contains no frames");
    }
    if (maximum_total_tokens == 0 || maximum_chunk_tokens == 0) {
        throw std::invalid_argument("local video token limits must be positive");
    }
    for (std::size_t index = 0; index < selected_frames.size(); ++index) {
        const auto& frame = selected_frames[index];
        if (!std::isfinite(frame.timestamp_seconds)) {
            throw std::invalid_argument("local video frame timestamp must be finite");
        }
        if (index != 0 &&
            (frame.source_index <= selected_frames[index - 1].source_index ||
             frame.timestamp_seconds < selected_frames[index - 1].timestamp_seconds)) {
            throw std::invalid_argument("local video frame timings must be in display order");
        }
    }

    LocalVideoPromptPlan plan;
    plan.width = output_width;
    plan.height = output_height;
    plan.grid_height = output_height / kPatch;
    plan.grid_width = output_width / kPatch;
    plan.tokens_per_temporal_group =
        checked_mul(static_cast<std::uint64_t>(plan.grid_height / kMerge),
                    static_cast<std::uint64_t>(plan.grid_width / kMerge), "spatial token count");
    if (plan.tokens_per_temporal_group > maximum_chunk_tokens) {
        throw std::invalid_argument("one local video temporal group exceeds Vision chunk capacity");
    }

    const std::uint64_t temporal_groups =
        (static_cast<std::uint64_t>(selected_frames.size()) + kTemporal - 1) / kTemporal;
    plan.total_tokens =
        checked_mul(temporal_groups, plan.tokens_per_temporal_group, "total token count");
    if (plan.total_tokens > maximum_total_tokens) {
        throw std::invalid_argument("local video tokens exceed request budget");
    }

    plan.selected_frames.assign(selected_frames.begin(), selected_frames.end());
    plan.timestamps.reserve(static_cast<std::size_t>(temporal_groups));
    for (std::size_t group = 0; group < temporal_groups; ++group) {
        const std::size_t first = group * kTemporal;
        const std::size_t second = std::min(first + 1, selected_frames.size() - 1);
        plan.timestamps.push_back((selected_frames[first].timestamp_seconds +
                                   selected_frames[second].timestamp_seconds) /
                                  2.0);
    }

    const std::uint64_t groups_per_chunk =
        maximum_chunk_tokens / plan.tokens_per_temporal_group;
    for (std::uint64_t begin = 0; begin < temporal_groups; begin += groups_per_chunk) {
        const std::uint64_t count = std::min(groups_per_chunk, temporal_groups - begin);
        const std::size_t frame_begin = static_cast<std::size_t>(begin * kTemporal);
        const std::size_t frame_end = std::min(
            selected_frames.size(), static_cast<std::size_t>((begin + count) * kTemporal));
        plan.chunks.push_back(LocalVideoChunkPlan{
            .temporal_begin = static_cast<std::size_t>(begin),
            .temporal_count = static_cast<std::size_t>(count),
            .frame_begin = frame_begin,
            .frame_count = frame_end - frame_begin,
            .token_begin = begin * plan.tokens_per_temporal_group,
            .token_count = count * plan.tokens_per_temporal_group,
        });
    }
    return plan;
}

} // namespace ninfer::models::qwen3_5::frontend
