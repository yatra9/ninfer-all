#include "models/qwen3_5/program/vision_control.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5 {
namespace {

std::int32_t checked_i32(std::size_t value, const char* label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

float coordinate(std::int32_t index, std::int32_t size, std::int32_t position_side) {
    return size <= 1 ? 0.0F
                     : static_cast<float>(index) * static_cast<float>(position_side - 1) /
                           static_cast<float>(size - 1);
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(std::string("vision control ") + label + " exceeds size_t");
    }
    return left * right;
}

} // namespace

VisionControlPlan plan_vision_control(const PreparedPromptData& prompt,
                                      const VisionConfig& config) {
    const auto merge         = checked_i32(config.spatial_merge_size, "merge size");
    const auto position_side = checked_i32(config.position_grid_side, "position grid side");
    if (merge <= 0 || position_side <= 0) {
        throw std::invalid_argument("Vision control geometry must be positive");
    }
    if (prompt.token_ids.size() != prompt.token_types.size()) {
        throw std::invalid_argument("vision control token types must cover the prompt");
    }
    VisionControlPlan plan{.spatial_merge_size = merge, .position_grid_side = position_side};
    plan.items.reserve(prompt.vision_items.size());
    std::size_t patch_cursor    = 0;
    std::size_t token_cursor    = 0;
    std::size_t next_span_begin = 0;
    for (const VisionItem& item : prompt.vision_items) {
        const std::int32_t t = item.grid.temporal;
        const std::int32_t h = item.grid.height;
        const std::int32_t w = item.grid.width;
        if (t <= 0 || h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0) {
            throw std::invalid_argument(
                "vision control grid must be positive and merge-aligned: " + std::to_string(t) +
                "x" + std::to_string(h) + "x" + std::to_string(w));
        }
        const std::size_t item_patches =
            checked_mul(checked_mul(static_cast<std::size_t>(t), static_cast<std::size_t>(h),
                                    "temporal/spatial patch product"),
                        static_cast<std::size_t>(w), "patch product");
        if (item.patch_begin != patch_cursor || item.patch_count != item_patches) {
            throw std::invalid_argument("vision control patch ranges are not canonical");
        }
        const std::size_t expected_spans =
            item.modality == PromptModality::Video ? static_cast<std::size_t>(t) : 1;
        if (item.token_spans.size() != expected_spans) {
            throw std::invalid_argument("vision control token spans do not match modality grid");
        }
        std::size_t item_tokens = 0;
        for (const TokenSpan& span : item.token_spans) {
            if (span.count == 0 || span.begin > prompt.token_types.size() ||
                span.count > prompt.token_types.size() - span.begin) {
                throw std::invalid_argument("vision control token span exceeds prompt");
            }
            if (span.begin < next_span_begin) {
                throw std::invalid_argument("vision control token spans are not ordered");
            }
            if (span.count > std::numeric_limits<std::size_t>::max() - item_tokens) {
                throw std::overflow_error("vision control merged token count exceeds size_t");
            }
            item_tokens += span.count;
            next_span_begin = span.begin + span.count;
        }
        if (item_tokens != item_patches / static_cast<std::size_t>(merge * merge)) {
            throw std::invalid_argument("vision control token spans do not cover merged patches");
        }
        if (item.token_spans.front().begin > std::numeric_limits<std::uint32_t>::max() ||
            item.token_spans.back().begin + item.token_spans.back().count >
                std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("vision control token frontier exceeds uint32");
        }
        plan.items.push_back(VisionItemControlPlan{
            .token_begin  = static_cast<std::uint32_t>(item.token_spans.front().begin),
            .token_end    = static_cast<std::uint32_t>(item.token_spans.back().begin +
                                                       item.token_spans.back().count),
            .merged_count = item_tokens,
        });
        token_cursor += item_tokens;
        patch_cursor += item_patches;
    }
    if (static_cast<std::uint64_t>(patch_cursor) != prompt.prepare.raw_patches ||
        static_cast<std::uint64_t>(token_cursor) != prompt.prepare.vision_tokens ||
        plan.items.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("vision control metadata does not cover prepared prompt");
    }
    return plan;
}

VisionControl build_vision_control(const PreparedPromptData& prompt, const VisionControlPlan& plan,
                                   std::uint32_t prepared_item_begin) {
    if (plan.items.size() != prompt.vision_items.size() ||
        prepared_item_begin > prompt.vision_items.size()) {
        throw std::invalid_argument("vision control plan does not cover the prepared media");
    }
    const auto merge         = plan.spatial_merge_size;
    const auto position_side = plan.position_grid_side;
    if (merge <= 0 || position_side <= 0) {
        throw std::invalid_argument("Vision control plan has no geometry");
    }
    VisionControl out;
    out.prepared_item_begin = prepared_item_begin;
    out.items.reserve(prompt.vision_items.size() - prepared_item_begin);
    for (std::size_t item_index = prepared_item_begin; item_index < prompt.vision_items.size();
         ++item_index) {
        const VisionItem& item             = prompt.vision_items[item_index];
        const VisionItemControlPlan& input = plan.items[item_index];
        const std::int32_t t               = item.grid.temporal;
        const std::int32_t h               = item.grid.height;
        const std::int32_t w               = item.grid.width;
        const std::size_t item_patches     = item.patch_count;

        VisionItemControl control;
        control.modality       = item.modality;
        control.grid           = item.grid;
        control.patch_begin    = item.patch_begin;
        control.patch_count    = item.patch_count;
        control.merged_count   = input.merged_count;
        control.segment_length = checked_i32(
            checked_mul(static_cast<std::size_t>(h), static_cast<std::size_t>(w), "segment length"),
            "segment length");
        control.segment_count = t;
        control.position_ids.resize(checked_mul(item_patches, 2, "position id count"));
        control.position_table_indices.reserve(
            checked_mul(item_patches, 4, "position table index count"));
        control.position_table_weights.reserve(
            checked_mul(item_patches, 4, "position table weight count"));
        const auto expected = static_cast<std::uint8_t>(item.modality);
        for (const TokenSpan& span : item.token_spans) {
            if (!std::all_of(prompt.token_types.begin() + static_cast<std::ptrdiff_t>(span.begin),
                             prompt.token_types.begin() +
                                 static_cast<std::ptrdiff_t>(span.begin + span.count),
                             [expected](std::uint8_t value) { return value == expected; })) {
                throw std::invalid_argument("vision control token span modality mismatch");
            }
            for (std::size_t i = 0; i < span.count; ++i) {
                control.scatter_indices.push_back(checked_i32(span.begin + i, "scatter index"));
            }
        }

        std::size_t position_cursor = 0;
        for (std::int32_t temporal = 0; temporal < t; ++temporal) {
            for (std::int32_t block_y = 0; block_y < h / merge; ++block_y) {
                for (std::int32_t block_x = 0; block_x < w / merge; ++block_x) {
                    for (std::int32_t inner_y = 0; inner_y < merge; ++inner_y) {
                        for (std::int32_t inner_x = 0; inner_x < merge; ++inner_x) {
                            const std::int32_t y                  = block_y * merge + inner_y;
                            const std::int32_t x                  = block_x * merge + inner_x;
                            control.position_ids[position_cursor] = y;
                            control.position_ids[item_patches + position_cursor] = x;
                            ++position_cursor;

                            const float yf        = coordinate(y, h, position_side);
                            const float xf        = coordinate(x, w, position_side);
                            const auto y0         = static_cast<std::int32_t>(yf);
                            const auto x0         = static_cast<std::int32_t>(xf);
                            const std::int32_t y1 = std::min(y0 + 1, position_side - 1);
                            const std::int32_t x1 = std::min(x0 + 1, position_side - 1);
                            const float wy        = yf - static_cast<float>(y0);
                            const float wx        = xf - static_cast<float>(x0);
                            control.position_table_indices.insert(
                                control.position_table_indices.end(),
                                {y0 * position_side + x0, y0 * position_side + x1,
                                 y1 * position_side + x0, y1 * position_side + x1});
                            control.position_table_weights.insert(
                                control.position_table_weights.end(),
                                {(1.0F - wy) * (1.0F - wx), (1.0F - wy) * wx, wy * (1.0F - wx),
                                 wy * wx});
                        }
                    }
                }
            }
        }

        if (position_cursor != item_patches || control.position_ids.size() != item_patches * 2 ||
            control.position_table_indices.size() != item_patches * 4 ||
            control.position_table_weights.size() != item_patches * 4 ||
            control.scatter_indices.size() != input.merged_count ||
            checked_mul(static_cast<std::size_t>(control.segment_length),
                        static_cast<std::size_t>(control.segment_count),
                        "segmented patch count") != item_patches) {
            throw std::invalid_argument("vision item control metadata is incomplete");
        }
        out.items.push_back(std::move(control));
    }
    if (out.items.size() != prompt.vision_items.size() - prepared_item_begin) {
        throw std::invalid_argument("vision control metadata does not cover prepared prompt");
    }
    return out;
}

VisionItemControl slice_vision_control(const VisionItemControl& item,
                                       std::int32_t temporal_begin,
                                       std::int32_t temporal_count) {
    if (item.segment_length <= 0 || item.segment_count <= 0 || item.grid.temporal <= 0 ||
        item.grid.temporal != item.segment_count || temporal_begin < 0 || temporal_count <= 0 ||
        temporal_begin > item.segment_count - temporal_count) {
        throw std::invalid_argument("Vision temporal slice is outside the item grid");
    }
    const std::size_t segment_length = static_cast<std::size_t>(item.segment_length);
    const std::size_t segment_count  = static_cast<std::size_t>(item.segment_count);
    const std::size_t patch_begin =
        checked_mul(static_cast<std::size_t>(temporal_begin), segment_length, "slice patch begin");
    const std::size_t patch_count =
        checked_mul(static_cast<std::size_t>(temporal_count), segment_length, "slice patch count");
    if (item.patch_count != checked_mul(segment_count, segment_length, "item patch count") ||
        item.position_ids.size() != checked_mul(item.patch_count, 2, "position id count") ||
        item.position_table_indices.size() !=
            checked_mul(item.patch_count, 4, "position table index count") ||
        item.position_table_weights.size() !=
            checked_mul(item.patch_count, 4, "position table weight count") ||
        item.merged_count == 0 || item.merged_count % segment_count != 0 ||
        item.scatter_indices.size() != item.merged_count) {
        throw std::invalid_argument("Vision temporal slice source metadata is incomplete");
    }
    if (patch_begin > std::numeric_limits<std::size_t>::max() - item.patch_begin) {
        throw std::overflow_error("Vision temporal slice patch offset exceeds size_t");
    }
    const std::size_t merged_per_segment = item.merged_count / segment_count;
    const std::size_t merged_begin = checked_mul(static_cast<std::size_t>(temporal_begin),
                                                 merged_per_segment, "slice merged begin");
    const std::size_t merged_count = checked_mul(static_cast<std::size_t>(temporal_count),
                                                 merged_per_segment, "slice merged count");

    VisionItemControl out;
    out.modality       = item.modality;
    out.grid           = item.grid;
    out.grid.temporal  = temporal_count;
    out.patch_begin    = item.patch_begin + patch_begin;
    out.patch_count    = patch_count;
    out.merged_count   = merged_count;
    out.segment_length = item.segment_length;
    out.segment_count  = temporal_count;
    out.position_ids.reserve(checked_mul(patch_count, 2, "slice position id count"));
    out.position_ids.insert(out.position_ids.end(), item.position_ids.begin() + patch_begin,
                            item.position_ids.begin() + patch_begin + patch_count);
    out.position_ids.insert(out.position_ids.end(),
                            item.position_ids.begin() + item.patch_count + patch_begin,
                            item.position_ids.begin() + item.patch_count + patch_begin + patch_count);
    out.scatter_indices.insert(out.scatter_indices.end(),
                               item.scatter_indices.begin() + merged_begin,
                               item.scatter_indices.begin() + merged_begin + merged_count);
    const std::size_t table_begin = checked_mul(patch_begin, 4, "slice table begin");
    const std::size_t table_count = checked_mul(patch_count, 4, "slice table count");
    out.position_table_indices.insert(
        out.position_table_indices.end(), item.position_table_indices.begin() + table_begin,
        item.position_table_indices.begin() + table_begin + table_count);
    out.position_table_weights.insert(
        out.position_table_weights.end(), item.position_table_weights.begin() + table_begin,
        item.position_table_weights.begin() + table_begin + table_count);
    return out;
}

} // namespace ninfer::models::qwen3_5
