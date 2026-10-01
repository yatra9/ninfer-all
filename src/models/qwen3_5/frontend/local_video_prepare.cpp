#include "models/qwen3_5/frontend/local_video_prepare.h"

#include "models/qwen3_5/frontend/media_cache.h"

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

} // namespace

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
