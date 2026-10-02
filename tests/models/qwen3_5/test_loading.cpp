#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/model_fixture.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using namespace ninfer::test::artifact_fixture;

using ninfer::test::qwen_fixture::ModelFixture;

void logical_data_and_instances() {
    ModelFixture fixture;
    fixture.file.root["components"]["vision"] = {{"config", Json::object()}, {"target", "text"}};
    fixture.file.root["components"]["text"]["config"]["rope_parameters"]["partial_rotary_factor"] =
        0.4999999999;
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    auto plan = qwen::plan_load(reader);
    require(plan.config().text.rope_parameters->partial_rotary_factor == 0.5F &&
                plan.config().text.rope_parameters->rotary_dim == 4,
            "rotary width was derived before normalizing the PositiveF32 config value");
    require(plan.config().text.architecture == Architecture::Qwen3_5 &&
                plan.config().text.hidden_size == 128,
            "binding chose a checkpoint-specific geometry");
    require(plan.resources().public_token_count == 265 &&
                plan.resources().tokenizer->encode("<|video_pad|>") == std::vector<int>{264},
            "tokenizer_config added tokens were not included in the public domain");
    const auto& attention = std::get<qwen::AttentionWeights>(plan.weights().text.layers[0].mixer);
    const auto& query     = plan.parameter(attention.query);
    const auto& gate      = plan.parameter(attention.gate);
    require(query.binding.parts[0].object == gate.binding.parts[0].object &&
                query.binding.parts[0].begin == 0 && gate.binding.parts[0].begin == 24 * 128,
            "Q/gate logical row correspondence changed");
    require(plan.uses(attention.query)[0].activation_input_divisor == 2.0F &&
                plan.uses(attention.gate)[0].activation_input_divisor == 3.0F,
            "shared-parent Uses contaminated one another");
    require(plan.parameter(plan.weights().text.token_embedding).binding.parts[0].object ==
                plan.parameter(plan.weights().text.output_head).binding.parts[0].object,
            "explicit shared embedding/head did not bind the same parent");
    const auto capacity = plan.materialization().device_capacity(0);
    require(!plan.weights().vision && plan.resources().preprocessor_config_json.empty(),
            "unused Vision was required");
    rejects([&] { (void)qwen::plan_load(reader, {.vision = true}); },
            "incomplete selected Vision was accepted");
    rejects([&] { (void)qwen::plan_load(reader, {.speculative = SpeculativeBackend::Mtp}); },
            "absent MTP was accepted");

    fixture.file.root["metadata"]["name"] = "another-training-run-and-recipe";
    fixture.file.root["components"]["text"]["config"]["num_hidden_layers"] = 2;
    fixture.file.root["components"]["text"]["config"]["layer_types"]       = {"full_attention",
                                                                              "full_attention"};
    const auto original_bindings = fixture.file.root["bindings"];
    for (const auto& [name, binding] : original_bindings.items()) {
        if (name.starts_with("text/layers/0/")) {
            auto second = name;
            second.replace(12, 1, "1");
            fixture.file.root["bindings"][second] = binding;
        }
    }
    const auto original_uses = fixture.file.root["uses"];
    for (auto use : original_uses) {
        auto parameter = use["parameter"].get<std::string>();
        if (parameter.starts_with("text/layers/0/")) {
            auto input = use["input"].get<std::string>();
            parameter.replace(12, 1, "1");
            input.replace(12, 1, "1");
            use["parameter"] = parameter;
            use["input"]     = input;
            fixture.file.root["uses"].push_back(use);
        }
    }
    fixture.file.write();
    artifact::Reader changed(fixture.file.entry);
    auto expanded = qwen::plan_load(changed);
    require(expanded.weights().text.layers.size() == 2 &&
                expanded.config().text.compact_layer_indices == std::vector<std::uint32_t>{0, 1},
            "config did not drive layer geometry and compact indices");
    require(expanded.materialization().device_capacity(0) == capacity,
            "cross-layer sharing duplicated parent storage");

    // --devices: each stage owns whole layers. The two layers of this fixture share every parent
    // object, so putting them on different stages cannot be satisfied and must be refused rather
    // than left living on one of the two devices.
    rejects<artifact::ArtifactError>(
        [&] { (void)qwen::plan_load(changed, {.ranks = 2}); },
        "layers that share parent storage were split across stages");
    rejects<std::invalid_argument>(
        [&] { (void)qwen::plan_load(changed, {.ranks = 5}); },
        "a split with more devices than layers was accepted");
    rejects<std::invalid_argument>(
        [&] { (void)qwen::plan_load(changed, {.ranks = 2, .stage_layers = {1}}); },
        "--stage-layers with the wrong number of counts was accepted");
    rejects<std::invalid_argument>(
        [&] { (void)qwen::plan_load(changed, {.ranks = 2, .stage_layers = {2, 1}}); },
        "--stage-layers that overshoot the model were accepted");
    rejects<std::invalid_argument>(
        [&] { (void)qwen::plan_load(changed, {.stage_layers = {2}}); },
        "--stage-layers without a multi-device split was accepted");
}

void native_uses() {
    const std::array<std::uint64_t, 2> shape{128, 64};
    const auto geometry = weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, shape);
    std::vector<std::byte> bytes(geometry.bytes);
    const WeightParent parent{geometry, bytes.data(), 8.0F};
    const WeightView gate{{64, 64}, {{&parent, 0, 4096}}};
    const WeightView up{{64, 64}, {{&parent, 4096, 8192}}};
    const auto joined = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA4, 2.0F},
                                                          {up, ops::LinearPolicy::AllowA8, 2.0F});
    require(joined.policy == ops::LinearPolicy::AllowA8 && joined.weight.payload == bytes.data() &&
                joined.weight.n == 128 && joined.weight.weight_scale_divisor == 8.0F &&
                joined.weight.input_scale_divisor == 2.0F,
            "shared quantization did not intersect Uses or preserve the complete parent");
    const WeightView full{{128, 64}, {{&parent, 0, 8192}}};
    for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        const auto no_aux = ops::prepare_linear_weight({full, policy});
        require(no_aux.policy == policy && no_aux.weight.payload == bytes.data(),
                "non-A4 NVFP4 use required an unused activation divisor");
        const auto mixed = ops::prepare_linear_swiglu_weight(
            {gate, ops::LinearPolicy::AllowA4, 2.0F}, {up, policy});
        require(mixed.policy == policy, "combined use required an unused child auxiliary");
        rejects<std::invalid_argument>(
            [&] { (void)ops::prepare_linear_weight({full, policy, 0.0F}); },
            "present activation divisor was accepted without value validation");
    }
    rejects<std::invalid_argument>(
        [&] { (void)ops::prepare_linear_weight({full, ops::LinearPolicy::AllowA4}); },
        "A4 NVFP4 use accepted a missing activation divisor");
    const auto a16 = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::A16Only, 2.0F},
                                                       {up, ops::LinearPolicy::AllowA4, 3.0F});
    require(a16.policy == ops::LinearPolicy::A16Only && a16.weight.input_scale_divisor == 2.0F,
            "A16-only use intersection rejected independent unused activation divisors");
    const auto second = ops::prepare_linear_weight({full, ops::LinearPolicy::A16Only, 3.0F});
    const auto first  = ops::prepare_linear_weight({full, ops::LinearPolicy::AllowA4, 2.0F});
    require(first.weight.input_scale_divisor == 2.0F && second.weight.input_scale_divisor == 3.0F &&
                first.policy == ops::LinearPolicy::AllowA4 &&
                second.policy == ops::LinearPolicy::A16Only && parent.weight_scale_divisor == 8.0F,
            "preparing one use modified another use or its shared parent");
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA4, 2.0F},
                                                    {up, ops::LinearPolicy::AllowA4, 3.0F});
        },
        "one native activation silently used differing divisors");
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({up, ops::LinearPolicy::A16Only, 2.0F},
                                                    {gate, ops::LinearPolicy::A16Only, 2.0F});
        },
        "reordered parent regions silently changed fused gate/up row order");
    const WeightView short_gate{{64, 64}, {{&parent, 0, 4032}}};
    const WeightView long_up{{64, 64}, {{&parent, 4032, 8192}}};
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({short_gate, ops::LinearPolicy::A16Only, 2.0F},
                                                    {long_up, ops::LinearPolicy::A16Only, 2.0F});
        },
        "combined coverage hid an incorrect gate/up boundary");
}

void invalid_model_data() {
    ModelFixture fixture;
    const Json original = fixture.file.root;
    const auto bad      = [&](auto mutate) {
        fixture.file.root = original;
        mutate(fixture.file.root);
        fixture.file.write();
        rejects(
            [&] {
                artifact::Reader reader(fixture.file.entry);
                (void)qwen::plan_load(reader);
            },
            "invalid model data was accepted");
    };
    bad([](Json& root) { root["components"]["text"]["config"]["layer_types"] = Json::array(); });
    bad([](Json& root) { root["components"]["text"]["config"]["num_key_value_heads"] = 3; });
    bad([](Json& root) {
        root["components"]["text"]["config"]["rope_parameters"]["mrope_section"] = {0, 2, 0};
    });
    bad([](Json& root) {
        root["bindings"]["text/layers/0/attention/query"]["parts"][0]["range"] = {0, 8 * 128};
    });
    bad([](Json& root) { root["uses"][1].erase("activation_policy"); });
    bad([](Json& root) { root["components"]["text"]["config"]["vocab_size"] = 264; });
}

} // namespace

int main() {
    try {
        logical_data_and_instances();
        native_uses();
        invalid_model_data();
        std::cout << "model config, binding, resources and Use checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
