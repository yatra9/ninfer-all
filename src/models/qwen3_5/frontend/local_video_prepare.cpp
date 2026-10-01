#include "models/qwen3_5/frontend/local_video_prepare.h"

#include "models/qwen3_5/frontend/media_cache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::frontend {
namespace {

media::local_video::Deinterlace convert_deinterlace(LocalVideoDeinterlace value) {
    switch (value) {
    case LocalVideoDeinterlace::Auto:
        return media::local_video::Deinterlace::Auto;
    case LocalVideoDeinterlace::On:
        return media::local_video::Deinterlace::On;
    case LocalVideoDeinterlace::Off:
        return media::local_video::Deinterlace::Off;
    }
    throw std::invalid_argument("invalid local video deinterlace mode");
}

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

std::size_t checked_elements(std::size_t patches) {
    if (patches > std::numeric_limits<std::size_t>::max() / kPreparedVisionPatchFeatures) {
        throw std::overflow_error("local video patch payload exceeds size_t");
    }
    return patches * kPreparedVisionPatchFeatures;
}

void append_patch(const media::local_video::Frame& first,
                  const media::local_video::Frame& second, int grid_y, int grid_x,
                  std::span<std::uint16_t> output, std::size_t& cursor) {
    if (cursor > output.size() || output.size() - cursor < kPreparedVisionPatchFeatures) {
        throw std::logic_error("local video patch writer exceeded its allocation");
    }
    const std::array<const media::local_video::Frame*, 2> frames{&first, &second};
    const auto& lut = normalization_lut();
    std::uint16_t* destination = output.data() + cursor;
    std::size_t local = 0;
    for (int channel = 0; channel < 3; ++channel) {
        for (const media::local_video::Frame* frame : frames) {
            for (int y = 0; y < 16; ++y) {
                const std::uint8_t* source =
                    frame->rgb.data() +
                    (static_cast<std::size_t>(grid_y * 16 + y) * frame->width + grid_x * 16) * 3 +
                    channel;
                for (int x = 0; x < 16; ++x) {
                    destination[local++] = lut[source[static_cast<std::size_t>(x) * 3]];
                }
            }
        }
    }
    cursor += local;
}

} // namespace

struct LocalVideoPayloadReader::Impl {
    const PreparedLocalVideoInput* input = nullptr;
    media::local_video::VideoReader reader;
    PreparationControl control;
    std::size_t next_chunk = 0;

    Impl(const PreparedLocalVideoInput& prepared, PreparationControl request_control)
        : input(&prepared),
          reader([&] {
              media::local_video::Options options = prepared.reader_options;
              options.checkpoint = [request_control] {
                  check_preparation_control(request_control, "local video decoding");
              };
              return prepared.source->create_reader(std::move(options));
          }()),
          control(std::move(request_control)) {}
};

LocalVideoPayloadReader::LocalVideoPayloadReader(const PreparedLocalVideoInput& input,
                                                 PreparationControl control)
    : impl_(std::make_unique<Impl>(input, std::move(control))) {}

LocalVideoPayloadReader::~LocalVideoPayloadReader() = default;
LocalVideoPayloadReader::LocalVideoPayloadReader(LocalVideoPayloadReader&&) noexcept = default;
LocalVideoPayloadReader&
LocalVideoPayloadReader::operator=(LocalVideoPayloadReader&&) noexcept = default;

std::shared_ptr<qwen3_5::PreparedMediaPayload>
LocalVideoPayloadReader::read_chunk(std::size_t chunk_index) {
    if (!impl_ || chunk_index != impl_->next_chunk ||
        chunk_index >= impl_->input->prompt.chunks.size()) {
        throw std::invalid_argument("local video chunks must be read once in plan order");
    }
    check_preparation_control(impl_->control, "local video decoding");
    const LocalVideoChunkPlan& plan = impl_->input->prompt.chunks[chunk_index];
    media::local_video::Chunk decoded = impl_->reader.read_chunk(plan.frame_count);
    if (decoded.frames.size() != plan.frame_count) {
        throw std::runtime_error("local video ended before its prepared chunk boundary");
    }
    const std::size_t global_begin = plan.frame_begin;
    for (std::size_t index = 0; index < decoded.frames.size(); ++index) {
        const auto& expected = impl_->input->prompt.selected_frames[global_begin + index];
        const auto& actual = decoded.frames[index];
        if (actual.source_index != expected.source_index || actual.source_pts != expected.source_pts ||
            actual.timestamp_seconds != expected.timestamp_seconds ||
            actual.width != impl_->input->prompt.width ||
            actual.height != impl_->input->prompt.height ||
            actual.rgb.size() != static_cast<std::size_t>(actual.width) * actual.height * 3) {
            throw std::runtime_error("local video decode no longer matches its prepared plan");
        }
    }

    const std::size_t patches =
        plan.temporal_count * static_cast<std::size_t>(impl_->input->prompt.grid_height) *
        static_cast<std::size_t>(impl_->input->prompt.grid_width);
    auto payload = std::make_shared<qwen3_5::PreparedMediaPayload>();
    payload->patch_elements = checked_elements(patches);
    payload->patches = std::make_unique<std::uint16_t[]>(payload->patch_elements);
    std::size_t cursor = 0;
    for (std::size_t temporal = 0; temporal < plan.temporal_count; ++temporal) {
        const std::size_t first_index = temporal * 2;
        const std::size_t second_index = std::min(first_index + 1, decoded.frames.size() - 1);
        const auto& first = decoded.frames[first_index];
        const auto& second = decoded.frames[second_index];
        for (int block_y = 0; block_y < impl_->input->prompt.grid_height / 2; ++block_y) {
            check_preparation_control(impl_->control, "local video patchification");
            for (int block_x = 0; block_x < impl_->input->prompt.grid_width / 2; ++block_x) {
                for (int merge_y = 0; merge_y < 2; ++merge_y) {
                    for (int merge_x = 0; merge_x < 2; ++merge_x) {
                        append_patch(first, second, block_y * 2 + merge_y,
                                     block_x * 2 + merge_x, payload->mutable_span(), cursor);
                    }
                }
            }
        }
    }
    if (cursor != payload->patch_elements) {
        throw std::logic_error("local video patch writer left a short payload");
    }
    ++impl_->next_chunk;
    return payload;
}

PreparedLocalVideoInput
prepare_local_video_input(const OwnedLocalVideo& input, const PreparationControl& control,
                          std::uint64_t maximum_total_tokens,
                          std::uint64_t maximum_chunk_tokens) {
    check_preparation_control(control, "local video planning");
    auto source = std::make_shared<media::local_video::VideoSource>(input.path);
    media::local_video::Options options;
    options.start = input.start_frame;
    options.end = input.end_frame;
    options.skip = input.skip_frame;
    if (input.crop) {
        options.crop = media::local_video::Rect{input.crop->x, input.crop->y, input.crop->width,
                                                input.crop->height};
    }
    options.scale = input.scale;
    options.alignment = 32;
    options.deinterlace = convert_deinterlace(input.deinterlace);
    options.checkpoint = [&control] { check_preparation_control(control, "local video planning"); };

    media::local_video::VideoPlan video = source->plan(options);
    LocalVideoPromptPlan prompt =
        plan_local_video_prompt(video.width, video.height, video.selected_frames,
                                maximum_total_tokens, maximum_chunk_tokens);
    check_preparation_control(control, "local video planning");
    // The preparation control belongs to this call. Execution installs its own request control.
    options.checkpoint = {};
    return PreparedLocalVideoInput{
        .source = std::move(source),
        .reader_options = std::move(options),
        .prompt = std::move(prompt),
    };
}

} // namespace ninfer::models::qwen3_5::frontend
