#include "guarded_main.h"
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/vision_control.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

namespace qwen = ninfer::models::qwen3_5;

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::vector<std::uint16_t> encode(
    ninfer::DeviceContext& device, const qwen::execution::Parameters& parameters,
    const qwen::VisionItemControl& control, std::span<const std::uint16_t> patches) {
    const qwen::VisionConfig& config = parameters.model.config().vision.value();
    const auto plan = qwen::execution::VisionContext::plan_workspace(
        config, parameters.vision.value(), static_cast<std::uint32_t>(control.merged_count), 1);
    ninfer::DeviceArena arena(plan.capacity_bytes);
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    qwen::execution::VisionContext context(device, parameters);
    ninfer::Tensor output =
        qwen::execution::VisionContext::bind_output(backing, plan, control.merged_count);
    context.encode({patches, &control}, output, backing, plan);
    device.synchronize();
    std::vector<std::uint16_t> host(output.bytes() / sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(host.data(), output.data, output.bytes(), cudaMemcpyDeviceToHost));
    return host;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') return 77;
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
        return 77;
    }
    CUDA_CHECK(status);

    ninfer::DeviceContext device;
    ninfer::models::LoadOptions selected;
    selected.vision = true;
    auto model = qwen::load_model(artifact, selected, device);
    const qwen::execution::Parameters parameters(*model);
    if (!parameters.vision || !parameters.model.config().vision) {
        std::cerr << "artifact has no Vision tower\n";
        return 1;
    }
    const qwen::VisionConfig& config = *parameters.model.config().vision;
    constexpr std::int32_t groups = 4;
    const std::int32_t side = static_cast<std::int32_t>(config.spatial_merge_size * 2);
    const std::size_t patches_per_group = static_cast<std::size_t>(side) * side;

    qwen::PreparedPromptData prompt;
    prompt.token_ids.resize(groups * 4 + 1);
    prompt.token_types.resize(prompt.token_ids.size());
    qwen::VisionItem item;
    item.modality = qwen::PromptModality::Video;
    item.grid = {.temporal = groups, .height = side, .width = side};
    item.patch_count = groups * patches_per_group;
    for (std::int32_t group = 0; group < groups; ++group) {
        const std::size_t begin = static_cast<std::size_t>(group * 4 + 1);
        item.token_spans.push_back({.begin = begin, .count = 4});
        std::fill_n(prompt.token_types.begin() + begin, 4,
                    static_cast<std::uint8_t>(qwen::PromptModality::Video));
    }
    prompt.vision_items.push_back(item);
    prompt.prepare.media_items = 1;
    prompt.prepare.raw_patches = item.patch_count;
    prompt.prepare.vision_tokens = groups * 4;

    const qwen::VisionControlPlan control_plan = qwen::plan_vision_control(prompt, config);
    const qwen::VisionItemControl full = qwen::build_vision_control(prompt, control_plan, 0).items.at(0);
    const std::size_t patch_width = static_cast<std::size_t>(config.patch_width());
    std::vector<std::uint16_t> patches(full.patch_count * patch_width);
    for (std::size_t index = 0; index < patches.size(); ++index) {
        const std::size_t group = index / (patches_per_group * patch_width);
        const std::size_t within_group = index % (patches_per_group * patch_width);
        // Every group has the same normalized-image-like distribution but a distinct hash seed,
        // so repeated or reordered temporal groups remain observable without a distribution shift.
        const std::uint32_t mixed =
            static_cast<std::uint32_t>(within_group * 2'654'435'761ULL) ^
            static_cast<std::uint32_t>((group + 1) * 2'246'822'519ULL);
        const float value =
            static_cast<float>(static_cast<std::int32_t>(mixed % 4097U) - 2048) / 1024.0F;
        patches[index] = float_to_bf16(value);
    }

    const std::vector<std::uint16_t> whole = encode(device, parameters, full, patches);
    std::vector<std::uint16_t> chunked;
    for (const std::pair<std::int32_t, std::int32_t> range :
         {std::pair{0, 1}, std::pair{1, 2}, std::pair{3, 1}}) {
        const qwen::VisionItemControl slice =
            qwen::slice_vision_control(full, range.first, range.second);
        const std::size_t patch_begin = slice.patch_begin - full.patch_begin;
        const auto output = encode(device, parameters, slice,
                                   std::span(patches).subspan(patch_begin * patch_width,
                                                              slice.patch_count * patch_width));
        chunked.insert(chunked.end(), output.begin(), output.end());
    }
    if (whole.size() != chunked.size()) {
        std::cerr << "full and chunked Vision output sizes differ: " << whole.size() << " vs "
                  << chunked.size() << '\n';
        return 1;
    }
    for (std::size_t index = 0; index < whole.size(); ++index) {
        if (!std::isfinite(bf16_to_float(whole[index])) ||
            !std::isfinite(bf16_to_float(chunked[index]))) {
            std::cerr << "full or chunked Vision output is non-finite at BF16 element " << index
                      << '\n';
            return 1;
        }
    }
    if (whole != chunked) {
        const auto mismatch = std::mismatch(whole.begin(), whole.end(), chunked.begin());
        float maximum_absolute = 0.0F;
        float maximum_relative = 0.0F;
        double squared_error = 0.0;
        double left_squared = 0.0;
        double right_squared = 0.0;
        double dot = 0.0;
        std::size_t differing = 0;
        for (std::size_t index = 0; index < whole.size(); ++index) {
            const float left = bf16_to_float(whole[index]);
            const float right = bf16_to_float(chunked[index]);
            const float absolute = std::abs(left - right);
            const float scale = std::max({std::abs(left), std::abs(right), 1.0e-6F});
            maximum_absolute = std::max(maximum_absolute, absolute);
            maximum_relative = std::max(maximum_relative, absolute / scale);
            squared_error += static_cast<double>(absolute) * absolute;
            left_squared += static_cast<double>(left) * left;
            right_squared += static_cast<double>(right) * right;
            dot += static_cast<double>(left) * right;
            differing += whole[index] != chunked[index];
        }
        const double rmse = std::sqrt(squared_error / static_cast<double>(whole.size()));
        const double cosine = dot / std::sqrt(left_squared * right_squared);
        const std::size_t token_width =
            whole.size() / static_cast<std::size_t>(full.merged_count);
        double minimum_token_cosine = 1.0;
        double maximum_token_rmse = 0.0;
        double maximum_token_nrmse = 0.0;
        for (std::size_t token = 0; token < static_cast<std::size_t>(full.merged_count); ++token) {
            double token_error = 0.0;
            double token_left_squared = 0.0;
            double token_right_squared = 0.0;
            double token_dot = 0.0;
            for (std::size_t column = 0; column < token_width; ++column) {
                const std::size_t index = token * token_width + column;
                const double left = bf16_to_float(whole[index]);
                const double right = bf16_to_float(chunked[index]);
                const double difference = left - right;
                token_error += difference * difference;
                token_left_squared += left * left;
                token_right_squared += right * right;
                token_dot += left * right;
            }
            maximum_token_rmse =
                std::max(maximum_token_rmse, std::sqrt(token_error / token_width));
            maximum_token_nrmse =
                std::max(maximum_token_nrmse, std::sqrt(token_error / token_left_squared));
            minimum_token_cosine =
                std::min(minimum_token_cosine,
                         token_dot / std::sqrt(token_left_squared * token_right_squared));
        }
        std::cerr << "chunked Vision embedding differs at BF16 element "
                  << std::distance(whole.begin(), mismatch.first) << " of " << whole.size()
                  << "; values=" << bf16_to_float(*mismatch.first) << ','
                  << bf16_to_float(chunked[std::distance(whole.begin(), mismatch.first)])
                  << "; differing=" << differing << "; max_abs=" << maximum_absolute
                  << "; max_rel=" << maximum_relative << "; rmse=" << rmse
                  << "; cosine=" << cosine << "; max_token_rmse=" << maximum_token_rmse
                  << "; max_token_nrmse=" << maximum_token_nrmse
                  << "; min_token_cosine=" << minimum_token_cosine << '\n';
        if (!std::isfinite(cosine) || !std::isfinite(minimum_token_cosine) ||
            !std::isfinite(rmse) || !std::isfinite(maximum_token_rmse) ||
            !std::isfinite(maximum_token_nrmse) || cosine < 0.999 ||
            minimum_token_cosine < 0.999 || rmse > 0.03 || maximum_token_nrmse > 0.05) {
            return 1;
        }
    }
    std::cout << "full and chunked Vision embeddings are equivalent: " << whole.size()
              << " BF16 elements\n";
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
