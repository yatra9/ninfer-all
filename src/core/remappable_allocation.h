#pragma once

#include <cstddef>
#include <memory>

namespace ninfer {
// The reservation survives detach. Callers must exclude and drain all users first.
class RemappableDeviceAllocation {
public:
    explicit RemappableDeviceAllocation(std::size_t bytes);
    ~RemappableDeviceAllocation();
    RemappableDeviceAllocation(const RemappableDeviceAllocation&) = delete;
    RemappableDeviceAllocation& operator=(const RemappableDeviceAllocation&) = delete;
    [[nodiscard]] void* data() const noexcept;
    [[nodiscard]] std::size_t physical_bytes() const noexcept;
    [[nodiscard]] bool attached() const noexcept;
    void attach();
    void detach();
    enum class AttachFault { None, Create, Map, Access };
    // One-shot fault at the next attach, for partial-allocation lifetime tests.
    void inject_attach_fault(AttachFault fault) noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer
