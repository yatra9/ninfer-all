#include "models/qwen3_5/frontend/local_video_prepare.h"
#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/vision_patchify.h"

#include <ninfer/types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
namespace lv = ninfer::media::local_video;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

std::vector<std::uint16_t> expected_payload(const fi::PreparedLocalVideoInput& prepared,
                                            const std::vector<lv::Frame>& frames) {
    const std::size_t groups = (frames.size() + 1) / 2;
    const std::size_t patches = groups * static_cast<std::size_t>(prepared.prompt.grid_height) *
                                prepared.prompt.grid_width;
    std::vector<std::uint16_t> output(patches * ninfer::models::qwen3_5::kPreparedVisionPatchFeatures);
    std::size_t cursor = 0;
    for (std::size_t temporal = 0; temporal < groups; ++temporal) {
        const auto& first = frames[temporal * 2];
        const auto& second = frames[std::min(temporal * 2 + 1, frames.size() - 1)];
        for (int block_y = 0; block_y < prepared.prompt.grid_height / 2; ++block_y) {
            for (int block_x = 0; block_x < prepared.prompt.grid_width / 2; ++block_x) {
                for (int merge_y = 0; merge_y < 2; ++merge_y) {
                    for (int merge_x = 0; merge_x < 2; ++merge_x) {
                        fi::append_vision_patch_pair(
                            {first.width, first.height, first.rgb},
                            {second.width, second.height, second.rgb}, block_y * 2 + merge_y,
                            block_x * 2 + merge_x, output, cursor);
                    }
                }
            }
        }
    }
    expect(cursor == output.size(), "reference payload fills its allocation");
    return output;
}

std::vector<lv::Frame> decode_reference(fi::PreparedLocalVideoInput& prepared) {
    auto reader = prepared.source->create_reader(prepared.reader_options);
    std::vector<lv::Frame> frames;
    while (true) {
        auto chunk = reader.read_chunk(2);
        for (auto& frame : chunk.frames) { frames.push_back(std::move(frame)); }
        if (chunk.eof) { break; }
    }
    return frames;
}

void run(const std::filesystem::path& path) {
    ninfer::OwnedLocalVideo input;
    input.path = path;
    input.deinterlace = ninfer::LocalVideoDeinterlace::Off;
    auto cancelled_source_cache = std::make_shared<fi::LocalVideoSourceCache>(1);
    std::atomic<unsigned> planning_checkpoints{0};
    bool planning_cancelled = false;
    try {
        (void)fi::prepare_local_video_input(
            input,
            ninfer::PreparationControl{
                .deadline = {},
                .cancellation = ninfer::CancellationView{[&planning_checkpoints] {
                    return planning_checkpoints.fetch_add(1, std::memory_order_relaxed) >= 2;
                }},
            },
            24, 12, cancelled_source_cache);
    } catch (const ninfer::RequestError& error) {
        planning_cancelled = error.kind() == ninfer::RequestErrorKind::Cancelled;
    }
    expect(planning_cancelled, "initial index construction observes request cancellation");
    expect(cancelled_source_cache->acquire(path)->source_stats().index_builds == 0,
           "cancelled initial index is not published");
    ninfer::OwnedLocalVideo oversized = input;
    oversized.scale = 128.0;
    bool pixels_are_budget = false;
    try {
        (void)fi::prepare_local_video_input(oversized, {}, 98'304, 16'384);
    } catch (const fi::ProcessorError& error) {
        pixels_are_budget = error.kind() == fi::ProcessorErrorKind::BudgetExceeded;
    }
    expect(pixels_are_budget, "local decoded-pixel limit is classified as a media budget");
    auto source_cache = std::make_shared<fi::LocalVideoSourceCache>(2);
    auto payload_account = std::make_shared<fi::MediaPreprocessCache>(0, 16 * 1024 * 1024, 1,
                                                                      16 * 1024 * 1024);
    auto prepared = fi::prepare_local_video_input(input, {}, 24, 12, source_cache,
                                                   payload_account);

    expect(prepared.prompt.width == 96 && prepared.prompt.height == 64,
           "prepared dimensions follow the fixture rather than a fixed resolution");
    expect(prepared.prompt.selected_frames.size() == 7, "all seven fixture frames are selected");
    expect(prepared.prompt.tokens_per_temporal_group == 6 && prepared.prompt.total_tokens == 24,
           "96x64 geometry produces six tokens per temporal group");
    expect(prepared.prompt.chunks.size() == 2, "four temporal groups are split into two chunks");
    expect(prepared.prompt.chunks[0].frame_count == 4 &&
               prepared.prompt.chunks[1].frame_begin == 4 &&
               prepared.prompt.chunks[1].frame_count == 3,
           "chunk boundaries retain the odd final source frame");

    auto frames = decode_reference(prepared);
    expect(frames.size() == 7, "reference reader decodes all selected frames");
    auto expected = expected_payload(prepared, frames);

    fi::LocalVideoPayloadReader reader(prepared);
    bool rejected_out_of_order = false;
    try {
        (void)reader.read_chunk(1);
    } catch (const std::invalid_argument&) {
        rejected_out_of_order = true;
    }
    expect(rejected_out_of_order, "payload reader rejects an out-of-order first chunk");

    std::vector<std::uint16_t> actual;
    for (std::size_t index = 0; index < prepared.prompt.chunks.size(); ++index) {
        auto payload = reader.read_chunk(index);
        expect(payload_account->stats().live_bytes == payload->patch_elements * sizeof(std::uint16_t),
               "local chunk payload is charged to the shared live-memory account");
        actual.insert(actual.end(), payload->span().begin(), payload->span().end());
    }
    expect(payload_account->stats().live_bytes == 0,
           "local chunk payload releases its live-memory reservation");
    auto tiny_account = std::make_shared<fi::MediaPreprocessCache>(0, 1, 1);
    auto memory_limited = fi::prepare_local_video_input(input, {}, 24, 12, source_cache,
                                                         tiny_account);
    bool live_limit_reported = false;
    try {
        fi::LocalVideoPayloadReader limited_reader(memory_limited);
        (void)limited_reader.read_chunk(0);
    } catch (const ninfer::RequestError& error) {
        live_limit_reported = error.kind() == ninfer::RequestErrorKind::MediaBudgetExceeded;
    }
    expect(live_limit_reported, "local chunk obeys the shared live-memory capacity");

    const std::size_t one_chunk_bytes =
        prepared.prompt.chunks.front().temporal_count *
        static_cast<std::size_t>(prepared.prompt.grid_height) * prepared.prompt.grid_width *
        ninfer::models::qwen3_5::kPreparedVisionPatchFeatures * sizeof(std::uint16_t);
    auto shared_account = std::make_shared<fi::MediaPreprocessCache>(0, one_chunk_bytes, 1,
                                                                     one_chunk_bytes);
    auto request_a = fi::prepare_local_video_input(input, {}, 24, 12, source_cache,
                                                    shared_account);
    auto request_b = fi::prepare_local_video_input(input, {}, 24, 12, source_cache,
                                                    shared_account);
    fi::LocalVideoPayloadReader reader_a(request_a);
    auto held = reader_a.read_chunk(0);
    auto competing = std::async(std::launch::async, [&request_b] {
        fi::LocalVideoPayloadReader reader_b(request_b);
        return reader_b.read_chunk(0);
    });
    expect(competing.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
           "a concurrent local chunk waits instead of exceeding shared live memory");
    held.reset();
    auto admitted = competing.get();
    expect(admitted && shared_account->stats().live_bytes == one_chunk_bytes,
           "waiting local chunk is admitted after the first releases its reservation");
    admitted.reset();
    expect(shared_account->stats().live_bytes == 0,
           "concurrent local chunk reservations fully return to the shared account");
    expect(actual == expected,
           "sequential chunk payloads exactly equal the independently decoded full payload");

    bool rejected_repeat = false;
    try {
        (void)reader.read_chunk(1);
    } catch (const std::invalid_argument&) {
        rejected_repeat = true;
    }
    expect(rejected_repeat, "payload reader rejects a repeated chunk");

    const auto stats = prepared.source->source_stats();
    expect(stats.index_builds == 1, "planning builds the source index once");
    expect(stats.index_reuses >= 2, "reference and payload readers reuse the prepared index");

    auto repeated = fi::prepare_local_video_input(input, {}, 24, 12, source_cache,
                                                   payload_account);
    expect(repeated.source == prepared.source,
           "a later request reuses the cached source for the unchanged path");
    expect(repeated.source->source_stats().index_builds == 1 &&
               repeated.source->source_stats().index_reuses > stats.index_reuses,
           "a later request reuses rather than rebuilds the completed index");
    expect(repeated.source_stats.index_builds == 1 && repeated.source_stats.index_reuses >= 3 &&
               repeated.index_stats.index_seconds > 0.0,
           "prepared diagnostics retain the request-visible index reuse and scan timing");

    std::atomic<unsigned> checkpoints{0};
    ninfer::PreparationControl cancelled{
        .deadline = {},
        .cancellation = ninfer::CancellationView{
            [&checkpoints] { return checkpoints.fetch_add(1, std::memory_order_relaxed) >= 2; }},
    };
    bool reported_cancelled = false;
    try {
        fi::LocalVideoPayloadReader interrupted(prepared, cancelled);
        (void)interrupted.read_chunk(0);
    } catch (const ninfer::RequestError& error) {
        reported_cancelled = error.kind() == ninfer::RequestErrorKind::Cancelled;
    }
    expect(reported_cancelled, "payload decoding reports cancellation from an in-flight checkpoint");

    fi::LocalVideoPayloadReader recovery(prepared);
    auto recovered = recovery.read_chunk(0);
    expect(!recovered->span().empty(),
           "a fresh request reads the shared source after an interrupted reader is destroyed");

    bool reported_deadline = false;
    try {
        fi::LocalVideoPayloadReader expired(
            prepared, ninfer::PreparationControl{
                          .deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1),
                          .cancellation = {},
                      });
        (void)expired.read_chunk(0);
    } catch (const ninfer::RequestError& error) {
        reported_deadline = error.kind() == ninfer::RequestErrorKind::QueueTimeout;
    }
    expect(reported_deadline, "payload decoding reports an expired request deadline");
}

} // namespace

// The product test links the real preparation-control implementation from ninfer_model_runtime.
// This definition is enabled only by the standalone WSLC source-level validation command.
#ifdef NINFER_LOCAL_VIDEO_PAYLOAD_STANDALONE
namespace ninfer::models::qwen3_5::frontend {
void check_preparation_control(const PreparationControl& control, std::string_view stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           std::string(stage) + " cancelled");
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           std::string(stage) + " exceeded request deadline");
    }
}
} // namespace ninfer::models::qwen3_5::frontend
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_local_video_payload VIDEO\n";
        return 2;
    }
    try {
        run(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "unexpected failure: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " local video payload checks failed\n";
        return 1;
    }
    std::cout << "local video payload checks passed\n";
    return 0;
}
