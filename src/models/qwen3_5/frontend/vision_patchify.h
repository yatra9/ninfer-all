#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::frontend {

struct RgbFrameView {
    int width = 0;
    int height = 0;
    std::span<const std::uint8_t> rgb;
};

// Appends one Qwen temporal patch pair in channel/temporal/y/x order. Inputs are already resized
// to the final aligned geometry. Values use the checkpoint's exact value/127.5-1 BF16 conversion.
void append_vision_patch_pair(RgbFrameView first, RgbFrameView second, int grid_y, int grid_x,
                              std::span<std::uint16_t> output, std::size_t& cursor);

} // namespace ninfer::models::qwen3_5::frontend
