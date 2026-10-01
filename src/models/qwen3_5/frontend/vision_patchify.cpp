#include "models/qwen3_5/frontend/vision_patchify.h"

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <bit>
#include <stdexcept>

namespace ninfer::models::qwen3_5::frontend {
namespace {

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

const std::array<std::uint16_t, 256>& normalization_lut() {
    static const std::array<std::uint16_t, 256> values = [] {
        std::array<std::uint16_t, 256> out{};
        for (std::size_t value = 0; value < out.size(); ++value) {
            out[value] = to_bf16(static_cast<float>(value) / 127.5f - 1.0f);
        }
        return out;
    }();
    return values;
}

void validate_frame(RgbFrameView frame, int grid_y, int grid_x) {
    if (frame.width <= 0 || frame.height <= 0 || grid_y < 0 || grid_x < 0 ||
        grid_y > frame.height / 16 - 1 || grid_x > frame.width / 16 - 1 ||
        frame.rgb.size() != static_cast<std::size_t>(frame.width) * frame.height * 3) {
        throw std::invalid_argument("Vision RGB patch source has an invalid shape");
    }
}

} // namespace

void append_vision_patch_pair(RgbFrameView first, RgbFrameView second, int grid_y, int grid_x,
                              std::span<std::uint16_t> output, std::size_t& cursor) {
    validate_frame(first, grid_y, grid_x);
    validate_frame(second, grid_y, grid_x);
    if (first.width != second.width || first.height != second.height) {
        throw std::invalid_argument("Vision temporal pair dimensions do not match");
    }
    if (cursor > output.size() || output.size() - cursor < kPreparedVisionPatchFeatures) {
        throw std::logic_error("Vision patch writer exceeded its allocation");
    }
    const std::array<RgbFrameView, 2> frames{first, second};
    const auto& lut = normalization_lut();
    std::uint16_t* destination = output.data() + cursor;
    std::size_t local = 0;
    for (int channel = 0; channel < 3; ++channel) {
        for (const RgbFrameView frame : frames) {
            for (int y = 0; y < 16; ++y) {
                const std::uint8_t* source =
                    frame.rgb.data() +
                    (static_cast<std::size_t>(grid_y * 16 + y) * frame.width + grid_x * 16) * 3 +
                    channel;
                for (int x = 0; x < 16; ++x) {
                    destination[local++] = lut[source[static_cast<std::size_t>(x) * 3]];
                }
            }
        }
    }
    cursor += local;
}

} // namespace ninfer::models::qwen3_5::frontend
