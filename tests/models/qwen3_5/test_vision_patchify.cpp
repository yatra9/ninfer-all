#include "models/qwen3_5/frontend/vision_patchify.h"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::uint16_t expected_bf16(std::uint8_t value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(static_cast<float>(value) / 127.5f - 1.0f);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::vector<std::uint8_t> make_frame(int width, int height, int seed) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                rgb[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
                    static_cast<std::uint8_t>((seed + y * 17 + x * 11 + channel * 67) & 0xff);
            }
        }
    }
    return rgb;
}

template <class Exception, class Function>
void expect_throws(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const Exception&) {}
}

void test_exact_layout_and_offset() {
    constexpr int width = 48;
    constexpr int height = 32;
    auto first = make_frame(width, height, 3);
    auto second = make_frame(width, height, 101);
    constexpr std::size_t patch_elements = 3 * 2 * 16 * 16;
    constexpr std::size_t prefix = 7;
    std::vector<std::uint16_t> output(prefix + patch_elements, 0x55aa);
    std::size_t cursor = prefix;

    fi::append_vision_patch_pair({width, height, first}, {width, height, second}, 1, 2,
                                 output, cursor);

    expect(cursor == output.size(), "writer advances by exactly one temporal patch");
    for (std::size_t index = 0; index < prefix; ++index) {
        expect(output[index] == 0x55aa, "writer preserves elements before the cursor");
    }
    std::size_t expected_index = prefix;
    const std::vector<std::uint8_t>* frames[] = {&first, &second};
    for (int channel = 0; channel < 3; ++channel) {
        for (const auto* frame : frames) {
            for (int y = 16; y < 32; ++y) {
                for (int x = 32; x < 48; ++x) {
                    const auto value = (*frame)[(static_cast<std::size_t>(y) * width + x) * 3 + channel];
                    expect(output[expected_index++] == expected_bf16(value),
                           "patch uses channel/temporal/y/x order and exact BF16 rounding");
                }
            }
        }
    }
}

void test_rejections() {
    auto frame = make_frame(32, 32, 0);
    auto different = make_frame(48, 32, 0);
    std::vector<std::uint16_t> output(3 * 2 * 16 * 16);
    std::size_t cursor = 0;
    expect_throws<std::invalid_argument>(
        [&] { fi::append_vision_patch_pair({32, 32, frame}, {48, 32, different}, 0, 0,
                                           output, cursor); },
        "mismatched temporal dimensions are rejected");
    expect_throws<std::invalid_argument>(
        [&] { fi::append_vision_patch_pair({32, 32, frame}, {32, 32, frame}, 2, 0,
                                           output, cursor); },
        "patches outside the frame are rejected");
    auto short_frame = frame;
    short_frame.pop_back();
    expect_throws<std::invalid_argument>(
        [&] { fi::append_vision_patch_pair({32, 32, short_frame}, {32, 32, frame}, 0, 0,
                                           output, cursor); },
        "incomplete RGB storage is rejected");
    output.pop_back();
    expect_throws<std::logic_error>(
        [&] { fi::append_vision_patch_pair({32, 32, frame}, {32, 32, frame}, 0, 0,
                                           output, cursor); },
        "undersized destination is rejected");
}

} // namespace

int main() {
    test_exact_layout_and_offset();
    test_rejections();
    if (failures != 0) {
        std::cerr << failures << " vision patchification checks failed\n";
        return 1;
    }
    std::cout << "vision patchification checks passed\n";
    return 0;
}
