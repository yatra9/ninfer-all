#include "models/qwen3_5/frontend/local_video_prepare.h"

#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/vision_patchify.h"

#include <algorithm>
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

std::size_t checked_elements(std::size_t patches) {
    if (patches > std::numeric_limits<std::size_t>::max() / kPreparedVisionPatchFeatures) {
        throw std::overflow_error("local video patch payload exceeds size_t");
    }
    return patches * kPreparedVisionPatchFeatures;
}

} // namespace

struct LocalVideoSourceCache::Entry {
    std::filesystem::path path;
    std::uintmax_t size = 0;
    std::filesystem::file_time_type modified;
    std::uint64_t last_use = 0;
    std::shared_ptr<media::local_video::VideoSource> source;
};

LocalVideoSourceCache::LocalVideoSourceCache(std::size_t maximum_entries)
    : maximum_entries_(maximum_entries) {
    if (maximum_entries_ == 0) {
        throw std::invalid_argument("local video source cache must retain at least one entry");
    }
    entries_.reserve(maximum_entries_);
}

LocalVideoSourceCache::~LocalVideoSourceCache() = default;

std::shared_ptr<media::local_video::VideoSource>
LocalVideoSourceCache::acquire(const std::filesystem::path& path) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path);
    std::lock_guard lock(mutex_);
    ++clock_;
    for (Entry& entry : entries_) {
        if (entry.path == path && entry.size == size && entry.modified == modified) {
            entry.last_use = clock_;
            return entry.source;
        }
    }
    auto source = std::make_shared<media::local_video::VideoSource>(path);
    Entry replacement{path, size, modified, clock_, source};
    auto stale = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
        return entry.path == path;
    });
    if (stale != entries_.end()) {
        *stale = std::move(replacement);
    } else if (entries_.size() < maximum_entries_) {
        entries_.push_back(std::move(replacement));
    } else {
        auto oldest = std::min_element(entries_.begin(), entries_.end(),
                                       [](const Entry& a, const Entry& b) {
                                           return a.last_use < b.last_use;
                                       });
        *oldest = std::move(replacement);
    }
    return source;
}

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
                        append_vision_patch_pair(
                            RgbFrameView{first.width, first.height, first.rgb},
                            RgbFrameView{second.width, second.height, second.rgb},
                            block_y * 2 + merge_y, block_x * 2 + merge_x,
                            payload->mutable_span(), cursor);
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
                          std::uint64_t maximum_chunk_tokens,
                          const std::shared_ptr<LocalVideoSourceCache>& source_cache) {
    check_preparation_control(control, "local video planning");
    std::shared_ptr<media::local_video::VideoSource> source;
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

    media::local_video::VideoPlan video;
    LocalVideoPromptPlan prompt;
    try {
        source = source_cache ? source_cache->acquire(input.path)
                              : std::make_shared<media::local_video::VideoSource>(input.path);
        video = source->plan(options);
        prompt = plan_local_video_prompt(video.width, video.height, video.selected_frames,
                                         maximum_total_tokens, maximum_chunk_tokens);
    } catch (const LocalVideoPlanError& error) {
        throw ProcessorError(error.kind() == LocalVideoPlanErrorKind::BudgetExceeded
                                 ? ProcessorErrorKind::BudgetExceeded
                                 : ProcessorErrorKind::InvalidMedia,
                             error.what());
    } catch (const std::runtime_error& error) {
        throw ProcessorError(ProcessorErrorKind::InvalidMedia, error.what());
    }
    check_preparation_control(control, "local video planning");
    // The preparation control belongs to this call. Execution installs its own request control.
    options.checkpoint = {};
    const media::local_video::SourceStats source_stats = source->source_stats();
    return PreparedLocalVideoInput{
        .source = std::move(source),
        .reader_options = std::move(options),
        .index_stats = video.index_stats,
        .source_stats = source_stats,
        .prompt = std::move(prompt),
    };
}

} // namespace ninfer::models::qwen3_5::frontend
