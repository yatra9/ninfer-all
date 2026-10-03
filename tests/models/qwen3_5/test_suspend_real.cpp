#include "ninfer/engine.h"

#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
using namespace ninfer;
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
}

// Explicit artifact, image and KV capacity; no implicit artifact discovery or downloads.
int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cout << "usage: test ARTIFACT PNG KV_CAPACITY [cache|control|host|d2h|h2d|unmap|release|access]\n";
        return 77;
    }
    try {
        EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = static_cast<std::uint32_t>(std::stoul(argv[3]));
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.kv_cache = KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
        options.gdn_state_fp16 = true;
        const bool control = argc == 5 && std::string(argv[4]) == "control";
        options.enable_model_suspend = !control;
        // The host fault targets lazy snapshot allocation during suspend.
        // Pinned snapshots are allocated before this probe is armed.
        if (argc == 5 && std::string(argv[4]) == "host") {
            options.suspend_snapshot_memory = SuspendSnapshotMemory::Pageable;
        }
        options.enable_vision = true;
        options.vision_residency = VisionResidency::Overlay;
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = ProposalHead::Optimized;
        options.context_cache.host_kv_capacity_bytes = 8ULL << 30;
        options.context_cache.host_state_slots = 8;
        Engine engine(options);
        const auto memory = engine.memory_summary();
        const auto load = engine.load_summary();
        std::cout << "logical weights=" << memory.weights.capacity_bytes
                  << " persistent=" << memory.sequence.capacity_bytes
                  << " workspace=" << memory.workspace.capacity_bytes
                  << " pinned_vision_weights=" << load.pinned_weight_bytes
                  << " overlay_window=" << load.overlay_window_bytes << std::endl;
        const bool cache = control || (argc == 5 && std::string(argv[4]) == "cache");
        RequestOptions request;
        request.execution.requested_output_tokens = 32;
        request.execution.sampling.temperature = 0;
        // Force actual prefill/Vision execution each time, rather than a cached image frontier.
        request.execution.allow_prefix_reuse = cache;
        const auto text = engine.tokenize_text("Name three primary colors. Answer briefly.");
        const auto baseline = engine.generate(engine.prepare_tokens(text, false), request);
        require(!baseline.generated_token_ids.empty(), "empty text baseline");
        if (argc == 5 && !cache) {
            const std::string mode = argv[4];
            require(mode == "host" || mode == "d2h" || mode == "h2d" || mode == "unmap" ||
                    mode == "release" || mode == "access", "unknown failure mode");
            const auto snapshot_bytes = std::to_string(memory.sequence.capacity_bytes);
            const auto workspace_bytes = std::to_string(engine.residency().workspace_device_bytes);
#ifdef _WIN32
            require(_putenv_s("NINFER_SNAPSHOT_BYTES", snapshot_bytes.c_str()) == 0,
                    "cannot configure backing probe");
            require(_putenv_s("NINFER_WORKSPACE_BYTES", workspace_bytes.c_str()) == 0,
                    "cannot configure workspace probe");
#else
            require(setenv("NINFER_SNAPSHOT_BYTES", snapshot_bytes.c_str(), 1) == 0,
                    "cannot configure backing probe");
            require(setenv("NINFER_WORKSPACE_BYTES", workspace_bytes.c_str(), 1) == 0,
                    "cannot configure workspace probe");
#endif
            const auto before = engine.residency();
            const bool restoring = mode == "h2d" || mode == "access";
            if (restoring) { (void)engine.suspend(); }
            bool failed = false;
            try {
                if (restoring) { (void)engine.resume(); }
                else { (void)engine.suspend(); }
            } catch (const ModelResidencyError& error) {
                require(error.kind() == ModelResidencyErrorKind::Failure, "wrong failure kind");
                failed = true;
            }
            require(failed, "injected failure was not observed");
            const auto status = engine.residency();
            if (mode == "host") {
                require(status.state == ModelResidencyState::Ready &&
                        status.retained_device_bytes == before.retained_device_bytes &&
                        status.persistent_snapshot_bytes == 0, "host failure changed backing");
                const auto result = engine.generate(engine.prepare_tokens(text, false), request);
                require(result.generated_token_ids == baseline.generated_token_ids,
                        "host allocation failure corrupted generation");
                (void)engine.suspend();
                (void)engine.resume();
            } else {
                require(status.state == ModelResidencyState::Error && !engine.is_available() &&
                        !status.last_error.empty(), "copy failure was exposed as available");
                require(mode == "d2h" || status.persistent_snapshot_bytes > 0,
                        "restore failure dropped diagnostic snapshot");
                require(mode != "d2h" || status.retained_device_bytes == before.retained_device_bytes,
                        "snapshot failure detached backing");
                bool rejected = false;
                try { (void)engine.generate(engine.prepare_tokens(text, false), request); }
                catch (const RequestError&) { rejected = true; }
                require(rejected, "ERROR accepted inference");
            }
            std::cout << "failure=" << mode << " safely handled: " << status.last_error << std::endl;
            return 0;
        }
        std::ifstream file(argv[2], std::ios::binary);
        require(file.good(), "image file cannot be opened");
        OwnedMedia image;
        image.kind = MediaKind::Image;
        image.media_type = "image/png";
        image.bytes.assign(std::istreambuf_iterator<char>(file), {});
        PromptInput input;
        input.options.reasoning_effort = ReasoningEffort::None;
        input.context_cache.retention = cache ? CacheRetentionHint::Default : CacheRetentionHint::Disposable;
        ChatMessage message;
        MessagePart text_part;
        text_part.text = "Name the two main colors in this image. Answer briefly.";
        message.parts.push_back(std::move(text_part));
        MessagePart image_part;
        image_part.kind = MessagePartKind::Media;
        image_part.media = std::move(image);
        message.parts.push_back(std::move(image_part));
        input.messages.push_back(std::move(message));
        const auto cold_vision = engine.generate(engine.prepare(input), request);
        require(cold_vision.timings.overlay_windows > 0, "Vision baseline did not open an overlay window");
        const auto vision = cache ? engine.generate(engine.prepare(input), request) : cold_vision;
        if (cache) {
            require(vision.reused_prompt_tokens > 0, "cache control did not reuse a frontier");
            std::cout << "cold_equals_cached=" << (cold_vision.generated_token_ids == vision.generated_token_ids)
                      << " cached_tokens=" << vision.reused_prompt_tokens << std::endl;
            const auto repeated = engine.generate(engine.prepare(input), request);
            require(repeated.generated_token_ids == vision.generated_token_ids,
                    "cache control changed without suspend");
        }
        const auto exclusive = cold_vision.timings.overlay_exclusive_windows;
        if (control) {
            require(!engine.residency().enabled, "suspend was enabled in the disabled control");
            return 0;
        }
        const auto ready = engine.residency();
        for (int iteration = 0; iteration < 3; ++iteration) {
            const auto suspended = engine.suspend();
            require(suspended.retained_device_bytes == 0, "backing was retained");
            if (const char* borrow = std::getenv("NINFER_BORROW_COMMAND")) {
                require(std::system(borrow) == 0, "separate GPU memory borrower failed");
            }
            const auto resumed = engine.resume();
            require(resumed.retained_device_bytes == ready.retained_device_bytes, "backing size changed");
            const auto result = engine.generate(engine.prepare_tokens(text, false), request);
            require(result.generated_token_ids == baseline.generated_token_ids, "text token vector changed");
            const auto actual = engine.generate(engine.prepare(input), request);
            if (cache) {
                require(actual.reused_prompt_tokens > 0, "restored Vision frontier was lost");
            } else {
                require(actual.reused_prompt_tokens == 0 && actual.timings.overlay_windows > 0,
                        "Vision execution was bypassed by cache");
                require(actual.timings.overlay_exclusive_windows == exclusive, "Vision loan tier changed");
            }
            require(actual.generated_token_ids == vision.generated_token_ids, "Vision token vector changed");
            std::cout << "cycle=" << iteration + 1 << " text_tokens=" << result.generated_token_ids.size()
                      << " vision_tokens=" << actual.generated_token_ids.size()
                      << " overlay=" << actual.timings.overlay_windows << " exclusive=" << exclusive
                      << " resume_seconds=" << resumed.last_resume_seconds << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
