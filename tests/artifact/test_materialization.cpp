#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::test {
void materialization_cuda_errors(DeviceContext& device);
}

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void suspend_restore(DeviceContext& device) {
    if (!EvictableWeightPool::supported(device)) { return; }
    for (const bool overlay : {false, true}) {
        Fixture fixture;
        fixture.write(true);
        auto reader = std::make_shared<Reader>(fixture.entry);
        Binder binder(*reader);
        (void)binder.parameter("matrix", {2, 130});
        const auto divisor = reader->find("divisors");
        binder.require_device(divisor);
        const auto resource = binder.resource("text", "tokenizer.json");
        if (overlay) { binder.evict_device(divisor, 1); }
        auto plan = std::move(binder).finish(overlay ? EvictableWeightPool::kChunkBytes : 256);
        std::unique_ptr<EvictableWeightPool> pool;
        if (overlay) {
            pool = std::make_unique<EvictableWeightPool>(device, EvictableWeightPool::Config{
                .arena_bytes = static_cast<std::size_t>(plan.device_capacity()),
                .evictable_tail_bytes = static_cast<std::size_t>(plan.evictable_tail_bytes)});
        }
        auto backing = materialize(*reader, MaterializationPlan(plan), device, nullptr, std::move(pool), true);
        const auto quantized = reader->find("q5");
        const auto* address = backing.device_parent(quantized).data;
        const auto host = backing.host_bytes(resource);
        std::vector<std::byte> expected(528), restored(528);
        CUDA_CHECK(cudaMemcpy(expected.data(), address, expected.size(), cudaMemcpyDeviceToHost));
        backing.retain_restore_source(reader, std::move(plan));
        DeviceBuffer output(8);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t execution = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaMemcpyAsync(output.p, address, 8, cudaMemcpyDeviceToDevice, device.stream));
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
        for (int cycle = 0; cycle < 10; ++cycle) {
            device.synchronize();
            backing.detach_backing();
            require(backing.physical_bytes() == 0, "weight backing stayed allocated");
            const auto upload = backing.restore_backing(device);
            require(upload.h2d_bytes == 536 && backing.device_parent(quantized).data == address,
                    "weight restore changed bindings or omitted bytes");
            require(backing.host_bytes(resource).data() == host.data(), "Host resources were recreated");
            CUDA_CHECK(cudaMemcpy(restored.data(), address, restored.size(), cudaMemcpyDeviceToHost));
            require(restored == expected, "multipart weight restore changed bytes");
            CUDA_CHECK(cudaGraphLaunch(execution, device.stream));
            device.synchronize();
            std::array<std::byte, 8> result{};
            output.copy_to_host(result.data(), result.size());
            require(std::equal(result.begin(), result.end(), expected.begin()), "existing graph failed after weight restore");
        }
        CUDA_CHECK(cudaGraphExecDestroy(execution));
        CUDA_CHECK(cudaGraphDestroy(graph));
        backing.detach_backing();
        std::filesystem::last_write_time(fixture.entry,
            std::filesystem::last_write_time(fixture.entry) + std::chrono::seconds(2));
        rejects([&] { (void)backing.restore_backing(device); }, "changed restore source was accepted");
        require(backing.physical_bytes() == 0, "changed source attached backing before validation");
    }
}

void transcode_suspend_restore(DeviceContext& device) {
    if (!EvictableWeightPool::supported(device)) { return; }
    for (const auto target : {QType::Q4_G64_FP16, QType::Q6_G64_FP16}) {
        Fixture fixture;
        const std::array<std::uint64_t, 2> shape{2, 128};
        const auto geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
        fixture.payload.assign(geometry.bytes, std::byte{});
        for (std::size_t i = 0; i < geometry.code_bytes; ++i) { fixture.payload[i] = std::byte(i % 17); }
        for (std::size_t i = 0; i < geometry.scale_bytes; i += 2) {
            put_word(fixture.payload, geometry.scale_offset + i, 0x3c00, 2);
        }
        fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
            {"objects", Json::array({{{"id", "matrix"}, {"kind", "tensor"}, {"shape", {2, 128}},
                {"format", "q8_g32_fp16"}, {"layout", "row_split_k128_v1"}, {"offset", 0}, {"bytes", geometry.bytes}}})},
            {"bindings", {{"matrix", {{"object", "matrix"}}}}}, {"uses", Json::array()},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", geometry.bytes}}})}};
        fixture.write();
        auto reader = std::make_shared<Reader>(fixture.entry);
        Binder binder(*reader);
        const auto object = reader->find("matrix");
        binder.require_device(object);
        binder.transcode_device(object, target);
        auto plan = std::move(binder).finish();
        DeviceArena destination(plan.device_capacity());
        const std::array<DeviceSpan, 1> too_small{DeviceSpan{destination.base(), 1}};
        rejects([&] { (void)upload_device_materialization(*reader, plan, too_small, device); },
                "undersized existing upload destination was accepted");
        auto backing = materialize(*reader, MaterializationPlan(plan), device, nullptr, nullptr, true);
        const auto& parent = backing.device_parent(object);
        std::vector<std::byte> expected(parent.geometry.bytes), actual(expected.size());
        CUDA_CHECK(cudaMemcpy(expected.data(), parent.data, expected.size(), cudaMemcpyDeviceToHost));
        backing.retain_restore_source(reader, std::move(plan));
        device.synchronize();
        backing.detach_backing();
        const auto stats = backing.restore_backing(device);
        CUDA_CHECK(cudaMemcpy(actual.data(), parent.data, actual.size(), cudaMemcpyDeviceToHost));
        require(actual == expected && stats.h2d_bytes == expected.size() && stats.read_bytes == geometry.bytes,
                "transcoded weight restore differs from startup representation");
    }
}

void materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    ObjectHandle divisors;
    ObjectHandle resource;
    const std::byte* original_host = nullptr;
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        row       = binder.parameter("row", {1, 130});
        quantized = reader.find("q5");
        divisors  = reader.find("divisors");
        (void)binder.parameter("values", {2}, Residency::Host, QType::FP32);
        binder.require_device(divisors);
        resource      = binder.resource("text", "tokenizer.json");
        original_host = binder.host_object(divisors).data();
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& stats = backing->stats();
    require(stats.device_object_count == 2 && stats.h2d_bytes == 536 &&
                stats.device_capacity_bytes == 776 && stats.retained_host_bytes == 13,
            "resident parents were duplicated or file gaps were allocated");
    require(backing->host_bytes(divisors).data() == original_host,
            "retained Host bytes moved after creating borrowed resource views");
    require(backing->host_bytes(resource).size() == 5,
            "Host resource was lost after Reader destruction");
    const auto view   = bind_view(row, *backing);
    const auto native = native_weight(view);
    require(native.n == 1 && native.k == 130 &&
                native.payload == backing->device_parent(quantized).data,
            "row view no longer refers to its owning parent");
    std::vector<std::byte> downloaded(528);
    CUDA_CHECK(cudaMemcpy(downloaded.data(), backing->device_parent(quantized).data,
                          downloaded.size(), cudaMemcpyDeviceToHost));
    require(std::equal(downloaded.begin(), downloaded.end(), fixture.payload.begin() + 256),
            "cross-file upload changed encoded parent bytes");
    std::array<std::byte, 8> values{};
    CUDA_CHECK(cudaMemcpy(values.data(), backing->device_parent(divisors).data, values.size(),
                          cudaMemcpyDeviceToHost));
    require(std::equal(values.begin(), values.end(), backing->host_bytes(divisors).begin()),
            "Host/device demand did not retain identical bytes");
    std::array<std::byte, 1> code{};
    CUDA_CHECK(cudaMemcpy(code.data(), native.qdata, 1, cudaMemcpyDeviceToHost));
    require(code[0] == std::byte{0x52}, "native row pointer addressed a different row");
}

// Overlay Vision residency plans: an evict-ranked object lands in a chunk-aligned arena tail that
// an eviction pool can borrow whole, and a pinned object lands in the page-locked block instead of
// device memory.
void overlay_placement(DeviceContext& device) {
    constexpr std::uint64_t kChunk = EvictableWeightPool::kChunkBytes;
    Fixture fixture;
    fixture.write(true);
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto matrix   = binder.parameter("matrix", {2, 130});
        const auto divisors = reader.find("divisors");
        binder.require_device(divisors);
        binder.evict_device(divisors, 700);
        const auto plan = std::move(binder).finish(kChunk);
        require(plan.device_objects.size() == 2 && plan.device_objects.front().offset == 0 &&
                    plan.evictable_tail_offset == kChunk && plan.evictable_tail_bytes == 8 &&
                    plan.device_objects.back().offset == kChunk &&
                    plan.device_capacity(0) == kChunk + 8,
                "evict-ranked object was not planned into a chunk-aligned arena tail");
        if (!EvictableWeightPool::supported(device)) {
            std::cout << "note: VMM unsupported, overlay transaction not exercised\n";
        } else {
            auto pool = std::make_unique<EvictableWeightPool>(
                device, EvictableWeightPool::Config{
                            .arena_bytes = static_cast<std::size_t>(plan.device_capacity(0)),
                            .evictable_tail_bytes =
                                static_cast<std::size_t>(plan.evictable_tail_bytes),
                        });
            auto backing = materialize(reader, MaterializationPlan(plan), device, nullptr,
                                       std::move(pool));
            EvictableWeightPool* const live = backing.weight_pool();
            require(live != nullptr, "pool-backed materialization dropped its pool");
            live->capture_window_mirror(kChunk, device.transfer_stream);
            const std::byte* const resident  = backing.device_parent(reader.find("q5")).data;
            const std::byte* const evictable = backing.device_parent(divisors).data;
            std::array<std::byte, 8> before{};
            CUDA_CHECK(cudaMemcpy(before.data(), evictable, before.size(),
                                  cudaMemcpyDeviceToHost));
            {
                auto transaction = live->evict(kChunk, device.stream);
                require(transaction.leased().bytes == kChunk,
                        "overlay window borrowed an unexpected extent");
                CUDA_CHECK(cudaMemsetAsync(transaction.leased().data, 0x5A, kChunk, device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
            }
            require(!live->poisoned(), "closing the overlay window poisoned the pool");
            require(backing.device_parent(divisors).data == evictable &&
                        backing.device_parent(reader.find("q5")).data == resident,
                    "weight addresses moved across an overlay window");
            std::array<std::byte, 8> after{};
            CUDA_CHECK(cudaMemcpy(after.data(), evictable, after.size(), cudaMemcpyDeviceToHost));
            require(after == before, "evict-ranked bytes were not restored from the mirror");
            std::array<std::byte, 4> head{};
            CUDA_CHECK(cudaMemcpy(head.data(), resident, head.size(), cudaMemcpyDeviceToHost));
            require(head[0] == std::byte{0x31}, "a resident object was inside the borrowed chunk");
            (void)matrix;
        }
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto row = binder.parameter("row", {1, 130}, Residency::Pinned);
        const auto plan = std::move(binder).finish();
        require(plan.device_capacity(0) == 0 && plan.pinned_objects.size() == 1 &&
                    plan.pinned_capacity_bytes == 528,
                "pinned residency did not plan the page-locked block");
        auto backing = materialize(reader, MaterializationPlan(plan), device);
        require(backing.stats().pinned_bytes == 528 && backing.pinned_block().size() == 528 &&
                    !backing.has_device(reader.find("q5")),
                "pinned object received device backing or was not accounted");
        const auto view = bind_view(row, backing);
        require(view.parts.front().parent->data == backing.pinned_block().data(),
                "pinned view does not address the pinned block");
        require(std::equal(backing.pinned_block().begin(), backing.pinned_block().end(),
                           fixture.payload.begin() + 256),
                "pinned block content differs from the artifact payload");
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        binder.require_pinned(reader.find("q5"));
        rejects([&] { (void)std::move(binder).finish(); },
                "an object with both device and pinned placement was accepted");
    }
}

void failure_and_host_only(DeviceContext& device) {
    Fixture fixture;
    fixture.write();
    {
        std::fstream part(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        part.seekp(16);
        part.put(0);
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), device); },
                "invalid required continuation was uploaded");
    }
    fixture.write();
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto resource = binder.resource("text", "tokenizer.json");
    auto backing        = materialize(reader, std::move(binder).finish(), device);
    require(backing.stats().device_capacity_bytes == 0 && backing.stats().h2d_bytes == 0 &&
                backing.host_bytes(resource).size() == 5,
            "Host-only demand allocated device weights");
    materialization(device);
}

void file_roundtrip(DeviceContext& device, const std::filesystem::path& path, bool writer_fixture) {
    std::optional<MaterializedArtifact> storage;
    std::map<std::size_t, std::vector<std::byte>> expected;
    std::vector<ParameterReference> parameters;
    {
        Reader reader(path);
        require(reader.file_bytes() < 32ULL * 1024 * 1024,
                "this complete-byte fixture check is limited to small artifacts");
        Binder binder(reader);
        for (const auto& [name, binding] : reader.directory().bindings) {
            auto shape = binding.whole_object
                             ? reader.directory().tensor(binding.parts.front().object).shape
                             : Shape{binding.elements};
            parameters.push_back(binder.parameter(name, std::move(shape)));
        }
        for (const auto& [key, use] : reader.directory().uses) {
            for (const auto& [name, auxiliary] : use.auxiliaries) {
                (void)binder.values(auxiliary);
            }
        }
        for (const auto& [name, component] : reader.directory().components) {
            for (const auto& [role, object] : component.resources) {
                (void)binder.resource(name, role);
            }
        }
        auto plan = std::move(binder).finish();
        for (const auto& placement : plan.device_objects) {
            auto bytes = reader.read_object(placement.object);
            if (writer_fixture && reader.directory().tensor(placement.object).id == "matrix") {
                require(bytes.size() == 130 * 130 * 2, "Python writer changed tensor dimensions");
                for (std::size_t i = 0; i < bytes.size(); ++i) {
                    require(bytes[i] == std::byte((i * 37 + 11) % 251),
                            "C++ reading differs from known Python writer input bytes");
                }
            }
            expected.emplace(placement.object.index, std::move(bytes));
        }
        storage.emplace(materialize(reader, std::move(plan), device));
    }
    for (const auto& [index, bytes] : expected) {
        std::vector<std::byte> actual(bytes.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), storage->device_parent({index}).data, actual.size(),
                              cudaMemcpyDeviceToHost));
        require(actual == bytes, "small artifact upload differs from source bytes");
    }
    for (const auto& parameter : parameters) {
        const auto view        = bind_view(parameter, *storage);
        std::uint64_t elements = 0;
        for (const auto& part : view.parts) {
            elements += part.end - part.begin;
            require(part.parent->data != nullptr, "resolved view lost its parent");
        }
        require(elements == weight_element_count(view.shape),
                "resolved view lost logical coverage");
    }
    std::cout << path.filename().string() << ": all bound parent bytes and logical views passed\n";
}

void staging_reuse(DeviceContext& device) {
    // More than one full staging ring, with distinct pages and a partial final block. Both
    // readers first encounter the same continuation; startup and restore must preserve bytes.
    constexpr std::size_t bytes = 5ULL * 64 * 1024 * 1024 + 1024;
    Fixture fixture;
    fixture.payload.clear();
    fixture.root    = {{"components", {{"text", {{"config", Json::object()}}}}},
                       {"objects", Json::array({{{"id", "large"},
                                                 {"kind", "tensor"},
                                                 {"shape", {bytes / 2}},
                                                 {"format", "bf16"},
                                                 {"layout", "contiguous_le_v1"},
                                                 {"offset", 4096},
                                                 {"bytes", bytes}}})},
                       {"bindings", {{"large", {{"object", "large"}}}}},
                       {"uses", Json::array()},
                       {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 4096}},
                                              {{"path", "weights-large.bin"}, {"payload_bytes", bytes}}})}};
    const auto text = fixture.root.dump();
    std::array<std::byte, 4096> header{};
    const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
    put_word(header, 8, text.size(), 8);
    std::memcpy(header.data() + 32, text.data(), text.size());
    std::ofstream entry(fixture.entry, std::ios::binary | std::ios::trunc);
    entry.exceptions(std::ios::badbit | std::ios::failbit);
    entry.write(reinterpret_cast<const char*>(header.data()), header.size());
    const std::array<std::byte, 4096> unused_entry_payload{};
    entry.write(reinterpret_cast<const char*>(unused_entry_payload.data()), unused_entry_payload.size());
    entry.close();
    const auto part_path = fixture.directory / "weights-large.bin";
    std::ofstream file(part_path, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    header.fill(std::byte{});
    std::copy(kPartMagic.begin(), kPartMagic.end(), header.begin());
    put_word(header, 8, 1, 8);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    std::array<std::byte, 4096> page{};
    for (std::size_t offset = 0; offset < bytes; offset += page.size()) {
        for (std::size_t i = 0; i < page.size(); ++i) {
            page[i] = std::byte(((offset + i) * 17 + (offset / 4096) * 13) % 251);
        }
        file.write(reinterpret_cast<const char*>(page.data()),
                   std::min(page.size(), bytes - offset));
    }
    file.close();
    auto reader = std::make_shared<Reader>(fixture.entry);
    Binder binder(*reader);
    (void)binder.parameter("large", {bytes / 2});
    auto plan = std::move(binder).finish();
    const bool suspendable = EvictableWeightPool::supported(device);
    auto backing = materialize(*reader, MaterializationPlan(plan), device, nullptr, nullptr, suspendable);
    const auto* base = backing.device_parent(reader->find("large")).data;
    std::vector<std::byte> chunk(1024 * 1024);
    const auto verify = [&] {
        for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
            const auto count = std::min(chunk.size(), bytes - offset);
            CUDA_CHECK(cudaMemcpy(chunk.data(), base + offset, count, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < count; ++i) {
                const auto position = offset + i;
                require(chunk[i] == std::byte((position * 17 + (position / 4096) * 13) % 251),
                        "staging slot reuse overwrote an in-flight or later block");
            }
        }
    };
    verify();
    if (suspendable) {
        backing.retain_restore_source(reader, plan);
        for (int cycle = 0; cycle < 3; ++cycle) {
            backing.detach_backing();
            const auto stats = backing.restore_backing(device);
            require(stats.h2d_bytes == bytes && stats.peak_staging_bytes == 4ULL * 64 * 1024 * 1024,
                    "parallel restore omitted payload or increased staging capacity");
            require(backing.device_parent(reader->find("large")).data == base,
                    "parallel restore changed the weight address");
            verify();
        }
    }
    // Bypass the immutable-source guard to exercise an I/O failure after read-ahead and H2D
    // have started. The reader must join pending I/O and drain uploads before freeing slots.
    std::filesystem::resize_file(part_path, 2ULL * 64 * 1024 * 1024 + 4096);
    const std::array<DeviceSpan, 1> destinations{DeviceSpan{const_cast<std::byte*>(base), plan.device_capacity()}};
    rejects([&] { (void)upload_device_materialization(*reader, plan, destinations, device); },
            "truncated parallel read was accepted");
    device.synchronize();
}

// Two ranks on one card: the expert-offload split's planning, its separate per-rank arenas and its
// per-rank upload stream, exercised without a second GPU. A repeated device id is supported for
// exactly this reason -- it frees no memory, but every placement and transfer path is the real one.
void pipeline_rank_placement() {
    const std::array<int, 2> ids{0, 0};
    DeviceContext split{std::span<const int>(ids)};
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto matrix   = reader.find("q5");
    const auto divisors = reader.find("divisors");
    std::vector<std::byte> expected = reader.read_object(divisors);

    Binder binder(reader);
    (void)binder.parameter("matrix", {2, 130});
    binder.require_device(divisors);
    binder.device_rank(divisors, 1);
    const auto plan = std::move(binder).finish();
    require(plan.device_rank_count() == 2 && plan.device_capacity(0) == 528 &&
                plan.device_capacity(1) == 8,
            "per-rank device capacities were not planned independently");
    for (const auto& placement : plan.device_objects) {
        require((placement.object.index == divisors.index) == (placement.rank == 1) &&
                    placement.offset == 0,
                "each rank's objects must start at the base of that rank's own arena");
    }

    auto backing = materialize(reader, MaterializationPlan(plan), split);
    require(backing.stats().device_capacity_bytes == 528 &&
                backing.stats().offloaded_device_capacity_bytes == 8 &&
                backing.stats().h2d_bytes == 536,
            "split materialization did not report one arena per rank");
    require(backing.device_parent(matrix).data != backing.device_parent(divisors).data,
            "both ranks were served from one allocation");
    std::vector<std::byte> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), backing.device_parent(divisors).data, actual.size(),
                          cudaMemcpyDeviceToHost));
    require(actual == expected, "the offloaded rank received the wrong bytes");

    {
        Binder rejected(reader);
        rejects([&] { rejected.device_rank(divisors, 1); },
                "a pipeline rank was accepted without a device placement");
    }
    {
        Binder rejected(reader);
        rejected.require_device(divisors);
        rejected.device_rank(divisors, 1);
        rejects([&] { rejected.device_rank(divisors, 0); },
                "one object was accepted on two different devices");
    }
    {
        // An offloaded rank holds expert blocks and nothing a Vision window could borrow, so the
        // evictable tail and a non-primary rank must not be planned together.
        Binder rejected(reader);
        rejected.require_device(divisors);
        rejected.evict_device(divisors, 1);
        rejected.device_rank(divisors, 1);
        rejects([&] { (void)std::move(rejected).finish(); },
                "an evictable placement was accepted away from the primary device");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        DeviceContext device;
        if (argc == 3 && (std::string_view(argv[1]) == "--artifact" ||
                          std::string_view(argv[1]) == "--writer-fixture")) {
            file_roundtrip(device, argv[2], std::string_view(argv[1]) == "--writer-fixture");
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument("expected [--artifact|--writer-fixture PATH]");
        }
        materialization(device);
        suspend_restore(device);
        transcode_suspend_restore(device);
        failure_and_host_only(device);
        overlay_placement(device);
        pipeline_rank_placement();
#if defined(NINFER_TEST_LINK_WRAP)
        ninfer::test::materialization_cuda_errors(device);
#else
        // CUDA fault injection needs GNU ld --wrap; MSVC has no link-time interposition, so those
        // cases are not built here. The remaining checks still run against the real runtime.
        std::cout << "artifact materialization CUDA fault injection skipped (no --wrap)\n";
#endif
        staging_reuse(device);
        std::cout << "artifact materialization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
