#include "models/qwen3_5/program/program_impl.h"
#include "core/evictable_kv_pool.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {
void residency_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}
bool stream_idle(cudaStream_t stream) {
    if (!stream) { return true; }
    const auto result = cudaStreamQuery(stream);
    if (result == cudaErrorNotReady) { return false; }
    residency_cuda(result, "query residency stream");
    return true;
}

void prepare_snapshot_pages(std::byte* storage, std::size_t bytes) {
    // Touch one byte per 4 KiB instead of clearing the snapshot. This commits all pages on
    // supported hosts before the synchronous pageable D2H enters its serial copy path.
    constexpr std::size_t stride = 4096;
    const auto pages = bytes / stride + (bytes % stride != 0);
    const auto touch = [storage](std::size_t begin, std::size_t end) {
        volatile std::byte* destination = storage;
        for (auto page = begin; page < end; ++page) { destination[page * stride] = std::byte{}; }
    };
    if (bytes < 64ULL * 1024 * 1024) {
        touch(0, pages);
    } else {
        const auto count = std::min(4U, std::max(1U, std::thread::hardware_concurrency()));
        std::vector<std::jthread> workers;
        workers.reserve(count);
        for (unsigned i = 0; i < count; ++i) {
            try {
                workers.emplace_back(touch, pages * i / count, pages * (i + 1) / count);
            } catch (const std::system_error&) {
                // Page preparation is an optimization: if workers are unavailable, finish
                // the unassigned pages on the caller while earlier workers finish theirs.
                touch(pages * i / count, pages);
                break;
            }
        }
        // jthread destruction joins even if a later thread could not be created.
    }
    // new[] need not be page aligned: include the page containing the final byte.
    if (bytes) { static_cast<volatile std::byte*>(storage)[bytes - 1] = std::byte{}; }
}
}

bool detail::ProgramImpl::residency_idle() const {
    if (!residency_storage_intact || has_context_transaction() || pending_transaction_ ||
        seal_window_claimed_.load() || (vision_broker && vision_broker->window_open()) ||
        (kv_arena && (kv_arena->lease_open() || kv_arena->poisoned()))) { return false; }
    const auto* weights = parameters.model.weight_pool();
    if (weights && (weights->transaction_open() || weights->poisoned())) { return false; }
    for (const auto& request : requests) {
        if (request.lifecycle != Lifecycle::Empty || request.pending.kind != PendingKind::None) {
            return false;
        }
    }
    // Completed background writes can remain unpublished while the worker sleeps idle.
    // Publish only completions already available; never wait for pending transfers.
    if (hybrid_) { hybrid_->poll_for_residency(); }
    if (hybrid_ && (hybrid_->restore_open() || hybrid_->restore_pending() ||
                    hybrid_->transfers_pending())) { return false; }
    for (std::size_t rank = 0; rank < compute_streams.size(); ++rank) {
        if (!stream_idle(compute_streams[rank]) || !stream_idle(transfer_streams[rank])) { return false; }
    }
    return stream_idle(device.vision_stream) &&
           (!hybrid_ || stream_idle(hybrid_->residency_restore_stream()));
}

void detail::ProgramImpl::snapshot_persistent() {
    if (!parameters.model.options().enable_model_suspend || !residency_idle()) {
        throw std::logic_error("persistent snapshot requires suspend-enabled idle Program");
    }
    // Keep the startup-sized pinned buffer; pageable mode prepares a temporary buffer per cycle.
    std::unique_ptr<std::byte[]> snapshot;
    if (!residency_pinned_snapshot) {
        snapshot.reset(new std::byte[persistent.capacity()]);
        prepare_snapshot_pages(snapshot.get(), persistent.capacity());
    }
    void* destination = residency_pinned_snapshot ? residency_pinned_snapshot->data() : snapshot.get();
    for (std::size_t rank = 0; rank < compute_streams.size(); ++rank) {
        residency_cuda(cudaStreamSynchronize(compute_streams[rank]), "drain compute stream");
        residency_cuda(cudaStreamSynchronize(transfer_streams[rank]), "drain transfer stream");
    }
    if (device.vision_stream) {
        residency_cuda(cudaStreamSynchronize(device.vision_stream), "drain Vision stream");
    }
    if (hybrid_ && hybrid_->residency_restore_stream()) {
        residency_cuda(cudaStreamSynchronize(hybrid_->residency_restore_stream()), "drain context restore stream");
    }
    residency_cuda(cudaMemcpy(destination, persistent.base(), persistent.capacity(),
                              cudaMemcpyDeviceToHost), "snapshot whole persistent arena");
    residency_snapshot = std::move(snapshot);
    residency_snapshot_valid = true;
}

void detail::ProgramImpl::detach_storage() {
    if (!residency_snapshot_valid) { throw std::logic_error("persistent snapshot is missing"); }
    // Set before the first physical change, including partial detach failures.
    residency_storage_intact = false;
    residency_host_cache_quiescent = false;
    if (kv_arena) { kv_arena->detach_backing(); }
    else { persistent.detach_backing(); }
    workspace_storage.detach_backing();
    residency_host_cache_quiescent = true;
}

Program::StorageRestoreTiming detail::ProgramImpl::restore_storage() {
    if (!residency_snapshot_valid || residency_storage_intact) {
        throw std::logic_error("Program storage is not suspended");
    }
    using Clock = std::chrono::steady_clock;
    residency_host_cache_quiescent = false;
    const auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    Program::StorageRestoreTiming timing;
    const auto persistent_map_start = Clock::now();
    if (kv_arena) { kv_arena->attach_backing(); }
    else { persistent.attach_backing(); }
    timing.map_seconds = elapsed(persistent_map_start);
    const auto h2d_start = Clock::now();
    residency_cuda(cudaMemcpy(persistent.base(), residency_snapshot_data(), persistent.capacity(),
                              cudaMemcpyHostToDevice), "restore whole persistent arena");
    // Pageable H2D can return once staged, before DMA completes on the default stream.
    // Engine streams are non-blocking: finish and check this copy before publishing READY.
    residency_cuda(cudaStreamSynchronize(nullptr), "complete persistent arena restore");
    timing.h2d_seconds = elapsed(h2d_start);
    const auto workspace_map_start = Clock::now();
    workspace_storage.attach_backing();
    timing.map_seconds += elapsed(workspace_map_start);
    // Scratch and Vision bridge are written before use; no arena clear or graph reconstruction.
    residency_storage_intact = true;
    return timing;
}

bool Program::residency_idle() const { return impl_->residency_idle(); }
std::size_t Program::persistent_capacity() const noexcept { return impl_->persistent.capacity(); }
std::size_t Program::persistent_device_bytes() const noexcept {
    return impl_->kv_arena ? impl_->kv_arena->physical_bytes() : impl_->persistent.physical_bytes();
}
std::size_t Program::workspace_device_bytes() const noexcept { return impl_->workspace_storage.physical_bytes(); }
std::size_t Program::snapshot_bytes() const noexcept {
    return impl_->residency_snapshot_valid ? impl_->persistent.capacity() : 0;
}
std::size_t Program::snapshot_capacity_bytes() const noexcept {
    return (impl_->residency_pinned_snapshot || impl_->residency_snapshot) ? impl_->persistent.capacity() : 0;
}
bool Program::snapshot_pinned() const noexcept { return impl_->residency_pinned_snapshot.has_value(); }
void Program::snapshot_persistent() { impl_->snapshot_persistent(); }
void Program::detach_storage() { impl_->detach_storage(); }
Program::StorageRestoreTiming Program::restore_storage() { return impl_->restore_storage(); }
void Program::release_snapshot() noexcept {
    impl_->residency_snapshot_valid = false;
    impl_->residency_snapshot.reset();
}
void Program::residency_error() noexcept {
    impl_->residency_storage_intact = false;
    impl_->residency_host_cache_quiescent = false;
}
} // namespace ninfer::models::qwen3_5
