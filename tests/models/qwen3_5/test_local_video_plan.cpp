#include "models/qwen3_5/frontend/local_video_plan.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using Timing = ninfer::media::local_video::FrameTiming;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<Timing> timings(std::size_t count) {
    std::vector<Timing> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(Timing{.source_index = static_cast<std::int64_t>(index * 3),
                                .source_pts = static_cast<std::int64_t>(100 + index * 1001),
                                .timestamp_seconds = 0.125 + static_cast<double>(index) / 29.97});
    }
    return result;
}

template <class Function>
void expect_invalid(Function&& function, std::string_view message) {
    try {
        function();
        expect(false, message);
    } catch (const std::invalid_argument&) {}
}

void test_representative_96ki_plan() {
    const auto frames = timings(256);
    const auto plan = fi::plan_local_video_prompt(1024, 768, frames, 98'304, 16'384);
    expect(plan.grid_width == 64 && plan.grid_height == 48, "1024x768 raw patch grid");
    expect(plan.tokens_per_temporal_group == 768, "1024x768 tokens per temporal group");
    expect(plan.timestamps.size() == 128 && plan.total_tokens == 98'304,
           "256 frames fill the 96Ki budget exactly");
    expect(plan.chunks.size() == 7, "96Ki input is divided into seven Vision chunks");
    expect(plan.chunks.front().temporal_count == 21 && plan.chunks.front().frame_count == 42 &&
               plan.chunks.front().token_count == 16'128,
           "full chunk stops at a temporal-group boundary");
    expect(plan.chunks.back().temporal_begin == 126 && plan.chunks.back().temporal_count == 2 &&
               plan.chunks.back().token_begin == 96'768 && plan.chunks.back().token_count == 1'536,
           "tail chunk covers the remaining groups without padding");
}

void test_variable_geometry_and_odd_tail() {
    const auto portrait_frames = timings(5);
    const auto portrait =
        fi::plan_local_video_prompt(768, 1024, portrait_frames, 98'304, 16'384);
    expect(portrait.tokens_per_temporal_group == 768 && portrait.total_tokens == 2'304,
           "portrait geometry derives tokens from its final dimensions");
    expect(portrait.chunks.size() == 1 && portrait.chunks[0].frame_count == 5 &&
               portrait.chunks[0].temporal_count == 3,
           "odd final group duplicates only its final frame");
    expect(std::abs(portrait.timestamps.back() - portrait_frames.back().timestamp_seconds) < 1e-12,
           "odd final timestamp duplicates the last selected frame");

    const auto small = fi::plan_local_video_prompt(640, 480, timings(4), 98'304, 16'384);
    expect(small.tokens_per_temporal_group == 300 && small.total_tokens == 600,
           "landscape geometry is not tied to 1024x768");
}

void test_rejections() {
    const auto frames = timings(256);
    expect_invalid([&] { (void)fi::plan_local_video_prompt(1000, 768, frames, 98'304, 16'384); },
                   "unaligned output dimensions are rejected");
    expect_invalid([&] { (void)fi::plan_local_video_prompt(1024, 768, frames, 98'303, 16'384); },
                   "aggregate token overflow is rejected");
    expect_invalid([&] { (void)fi::plan_local_video_prompt(4096, 4096, timings(2), 98'304, 16'384); },
                   "one group larger than the execution chunk is rejected");
    auto reversed = timings(2);
    reversed[1].timestamp_seconds = reversed[0].timestamp_seconds - 1.0;
    expect_invalid([&] { (void)fi::plan_local_video_prompt(640, 480, reversed, 98'304, 16'384); },
                   "non-monotonic timestamps are rejected");
}

} // namespace

int main() {
    test_representative_96ki_plan();
    test_variable_geometry_and_odd_tail();
    test_rejections();
    if (failures != 0) {
        std::cerr << failures << " local video plan checks failed\n";
        return 1;
    }
    std::cout << "local video plan checks passed\n";
    return 0;
}
