#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
bool delay_transcode = false;
int delayed_copies = 0;
constexpr auto copy_delay = std::chrono::milliseconds(20);
}
extern "C" cudaError_t __real_cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind);
extern "C" cudaError_t __wrap_cudaMemcpy(void* destination, const void* source,
                                        std::size_t bytes, cudaMemcpyKind kind) {
    if (delay_transcode && kind == cudaMemcpyHostToDevice) {
        ++delayed_copies;
        std::this_thread::sleep_for(copy_delay);
    }
    return __real_cudaMemcpy(destination, source, bytes, kind);
}
namespace {
using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;
void check(DeviceContext& device, bool direct, bool transcode) {
    Fixture fixture;
    const std::array<std::uint64_t, 2> shape{2, 128};
    const auto geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
    const unsigned count = direct && transcode ? 2 : 1;
    const auto stride = (geometry.bytes + 255) / 256 * 256;
    fixture.payload.assign((count - 1) * stride + geometry.bytes, std::byte{});
    Json objects = Json::array();
    for (unsigned index = 0; index < count; ++index) {
        const auto offset = index * stride;
        for (std::size_t i = 0; i < geometry.code_bytes; ++i) {
            fixture.payload[offset + i] = std::byte(i % 17);
        }
        for (std::size_t i = 0; i < geometry.scale_bytes; i += 2) {
            put_word(fixture.payload, offset + geometry.scale_offset + i, 0x3c00, 2);
        }
        objects.push_back({{"id", "matrix" + std::to_string(index)}, {"kind", "tensor"},
            {"shape", {2, 128}}, {"format", "q8_g32_fp16"}, {"layout", "row_split_k128_v1"},
            {"offset", offset}, {"bytes", geometry.bytes}});
    }
    fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
        {"objects", objects}, {"bindings", Json::object()}, {"uses", Json::array()},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", fixture.payload.size()}}})}};
    fixture.write();
    auto reader = std::make_shared<Reader>(fixture.entry);
    Binder binder(*reader);
    std::uint64_t expected_bytes = 0;
    std::vector<ObjectHandle> handles;
    for (unsigned index = 0; index < count; ++index) {
        const auto handle = reader->find("matrix" + std::to_string(index));
        handles.push_back(handle);
        binder.require_device(handle);
        if (transcode && index == count - 1) {
            binder.transcode_device(handle, QType::Q4_G64_FP16);
            expected_bytes += weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowSplit, shape).bytes;
        } else { expected_bytes += geometry.bytes; }
    }
    auto plan = std::move(binder).finish();
    auto backing = materialize(*reader, MaterializationPlan(plan), device, nullptr, nullptr, true);
    std::vector<std::vector<std::byte>> before;
    for (const auto handle : handles) {
        const auto& parent = backing.device_parent(handle);
        before.emplace_back(parent.geometry.bytes);
        CUDA_CHECK(cudaMemcpy(before.back().data(), parent.data, before.back().size(), cudaMemcpyDeviceToHost));
    }
    backing.retain_restore_source(reader, std::move(plan));
    backing.detach_backing();
    delayed_copies = 0;
    delay_transcode = true;
    const auto stats = backing.restore_backing(device);
    delay_transcode = false;
    require(stats.upload_seconds > 0 && stats.h2d_bytes == expected_bytes && stats.read_bytes > 0,
            "restore timing or byte counts missing");
    if (transcode) {
        require(delayed_copies == 1 && stats.upload_seconds >= std::chrono::duration<double>(copy_delay).count(),
                "restore timing omitted transcode upload");
    } else { require(delayed_copies == 0, "ordinary pipeline used transcode route"); }
    for (unsigned index = 0; index < count; ++index) {
        std::vector<std::byte> after(before[index].size());
        CUDA_CHECK(cudaMemcpy(after.data(), backing.device_parent(handles[index]).data, after.size(), cudaMemcpyDeviceToHost));
        require(after == before[index], "timed restore changed weight bytes");
    }
}
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        DeviceContext device(0);
        if (!EvictableWeightPool::supported(device)) { return 77; }
        check(device, true, false); check(device, false, true); check(device, true, true);
        std::cout << "PASS ordinary, transcode-only and mixed restore timings\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
