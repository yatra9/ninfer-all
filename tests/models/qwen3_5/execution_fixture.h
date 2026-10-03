#pragma once
#include "models/qwen3_5/model_fixture.h"

namespace ninfer::test::qwen_fixture {
inline void execution_fixture(ModelFixture& fixture) {
    // Keep the byte tokenizer; use one attention and one GDN layer with supported dense geometry.
    std::vector<std::pair<std::string, std::string>> resources;
    for (const auto& object : fixture.file.root["objects"]) {
        if (object["kind"] != "resource") { continue; }
        const auto offset = object["offset"].get<std::size_t>();
        const auto bytes = object["bytes"].get<std::size_t>();
        resources.emplace_back(object["id"].get<std::string>(),
            std::string(reinterpret_cast<const char*>(fixture.file.payload.data() + offset), bytes));
    }
    fixture.file.payload.clear();
    fixture.file.root["objects"] = Json::array();
    fixture.file.root["uses"] = Json::array();
    fixture.file.root["bindings"] = Json::object();
    auto& config = fixture.file.root["components"]["text"]["config"];
    config["hidden_size"] = 5120;
    config["num_hidden_layers"] = 2;
    config["layer_types"] = {"full_attention", "linear_attention"};
    config["linear_num_key_heads"] = 16;
    config["linear_key_head_dim"] = 128;
    config["linear_num_value_heads"] = 48;
    config["linear_value_head_dim"] = 128;
    config["linear_conv_kernel_dim"] = 4;
    config["num_attention_heads"] = 24;
    config["num_key_value_heads"] = 4;
    config["head_dim"] = 256;
    config["intermediate_size"] = 17408;
    config["rope_parameters"]["mrope_section"] = {22, 21, 21};
    fixture.parameter("text/token_embedding", {272, 5120});
    fixture.file.root["bindings"]["text/output_head"] = {{"object", "text/token_embedding"}};
    fixture.use("text/output_head", "text/final_hidden");
    fixture.parameter("text/final_norm", {5120});
    const std::string p = "text/layers/0/";
    fixture.parameter(p + "input_norm", {5120});
    fixture.parameter(p + "post_attention_norm", {5120});
    fixture.tensor("attention-storage", {14336, 5120});
    const std::array roles = {"query", "key", "gate", "value"};
    const std::array rows = {0, 6144, 7168, 13312, 14336};
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto name = p + "attention/" + roles[i];
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "attention-storage"},
                                     {"range", {rows[i] * 5120, rows[i + 1] * 5120}}}})}};
        fixture.use(name, p + "mixer_input");
    }
    fixture.parameter(p + "attention/query_norm", {256});
    fixture.parameter(p + "attention/key_norm", {256});
    fixture.parameter(p + "attention/output", {5120, 6144}, p + "attention/gated_output");
    fixture.tensor("gate-up-storage", {34816, 5120}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    for (int i = 0; i < 2; ++i) {
        const auto name = p + (i == 0 ? "mlp/gate" : "mlp/up");
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gate-up-storage"},
                                     {"range", {i * 17408 * 5120, (i + 1) * 17408 * 5120}}}})}};
        fixture.use(name, p + "ffn_input");
    }
    fixture.tensor("down-storage", {5120, 17408}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    fixture.file.root["bindings"][p + "mlp/down"] = {{"object", "down-storage"}};
    fixture.use(p + "mlp/down", p + "mlp/product");
    const std::string g = "text/layers/1/";
    fixture.parameter(g + "input_norm", {5120});
    fixture.parameter(g + "post_attention_norm", {5120});
    fixture.tensor("gdn-input-storage", {16384, 5120}, QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale);
    const std::array gdn_roles = {"query", "key", "value", "z"};
    const std::array gdn_rows = {0, 2048, 4096, 10240, 16384};
    for (std::size_t i = 0; i < gdn_roles.size(); ++i) {
        const auto name = g + "gdn/" + gdn_roles[i];
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gdn-input-storage"},
                                     {"range", {gdn_rows[i] * 5120, gdn_rows[i + 1] * 5120}}}})}};
        fixture.use(name, g + "mixer_input");
    }
    fixture.tensor("gdn-control-storage", {96, 5120});
    for (int i = 0; i < 2; ++i) {
        const auto name = g + (i == 0 ? "gdn/a_projection" : "gdn/b_projection");
        fixture.file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", "gdn-control-storage"},
                                     {"range", {i * 48 * 5120, (i + 1) * 48 * 5120}}}})}};
        fixture.use(name, g + "mixer_input");
    }
    for (const auto* name : {"a_log", "dt_bias"}) {
        fixture.tensor(g + "gdn/" + name, {48}, QType::FP32);
        fixture.file.root["bindings"][g + "gdn/" + name] = {{"object", g + "gdn/" + name}};
    }
    fixture.parameter(g + "gdn/convolution", {4, 10240});
    fixture.parameter(g + "gdn/norm", {128});
    fixture.parameter(g + "gdn/output", {5120, 6144}, g + "gdn/gated_output");
    for (const auto* name : {"gate", "up"}) {
        fixture.file.root["bindings"][g + "mlp/" + name] = fixture.file.root["bindings"][p + "mlp/" + name];
        fixture.use(g + "mlp/" + name, g + "ffn_input");
    }
    fixture.file.root["bindings"][g + "mlp/down"] = fixture.file.root["bindings"][p + "mlp/down"];
    fixture.use(g + "mlp/down", g + "mlp/product");
    for (const auto& [role, bytes] : resources) {
        if (role == "tokenizer_config.json") {
            auto tokenizer = Json::parse(bytes);
            tokenizer["add_bos_token"] = false;
            tokenizer["add_prefix_space"] = false;
            tokenizer["pad_token"] = "<|endoftext|>";
            for (auto& token : tokenizer["added_tokens_decoder"]) {
                if (token["content"] == "<think>" || token["content"] == "</think>") {
                    token["special"] = false;
                }
            }
            fixture.resource(role, tokenizer.dump());
        } else if (role == "tokenizer.json") {
            auto tokenizer = Json::parse(bytes);
            for (auto& token : tokenizer["added_tokens"]) {
                if (token["content"] == "<think>" || token["content"] == "</think>") {
                    token["special"] = false;
                }
            }
            fixture.resource(role, tokenizer.dump());
        } else {
            fixture.resource(role, bytes);
        }
    }
    fixture.file.root["files"][0]["payload_bytes"] = fixture.file.payload.size();
    fixture.file.write();
}
} // namespace ninfer::test::qwen_fixture
