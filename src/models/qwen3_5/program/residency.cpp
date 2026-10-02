#include "models/qwen3_5/program/program_impl.h"
#include "core/evictable_kv_pool.h"

#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <string>

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
    // No value initialization and no pinned allocation: touching every byte happens in D2H.
    auto snapshot = std::unique_ptr<std::byte[]>(new std::byte[persistent.capacity()]);
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
    residency_cuda(cudaMemcpy(snapshot.get(), persistent.base(), persistent.capacity(),
                              cudaMemcpyDeviceToHost), "snapshot whole persistent arena");
    residency_snapshot = std::move(snapshot);
}

void detail::ProgramImpl::detach_storage() {
    if (!residency_snapshot) { throw std::logic_error("persistent snapshot is missing"); }
    // Set before the first physical change, including partial detach failures.
    residency_storage_intact = false;
    if (kv_arena) { kv_arena->detach_backing(); }
    else { persistent.detach_backing(); }
    workspace_storage.detach_backing();
}

Program::StorageRestoreTiming detail::ProgramImpl::restore_storage() {
    if (!residency_snapshot || residency_storage_intact) {
        throw std::logic_error("Program storage is not suspended");
    }
    using Clock = std::chrono::steady_clock;
    const auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    Program::StorageRestoreTiming timing;
    const auto persistent_map_start = Clock::now();
    if (kv_arena) { kv_arena->attach_backing(); }
    else { persistent.attach_backing(); }
    timing.map_seconds = elapsed(persistent_map_start);
    const auto h2d_start = Clock::now();
    residency_cuda(cudaMemcpy(persistent.base(), residency_snapshot.get(), persistent.capacity(),
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
    return impl_->residency_snapshot ? impl_->persistent.capacity() : 0;
}
void Program::snapshot_persistent() { impl_->snapshot_persistent(); }
void Program::detach_storage() { impl_->detach_storage(); }
Program::StorageRestoreTiming Program::restore_storage() { return impl_->restore_storage(); }
void Program::release_snapshot() noexcept { impl_->residency_snapshot.reset(); }
void Program::residency_error() noexcept { impl_->residency_storage_intact = false; }
} // namespace ninfer::models::qwen3_5
