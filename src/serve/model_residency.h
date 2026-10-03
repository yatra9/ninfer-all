#pragma once

#include "ninfer/types.h"
#include "serve/request.h"
#include <nlohmann/json.hpp>

namespace ninfer::serve {
inline const char* residency_state_name(ModelResidencyState state) {
    switch (state) {
    case ModelResidencyState::Ready: return "ready";
    case ModelResidencyState::Suspending: return "suspending";
    case ModelResidencyState::Suspended: return "suspended";
    case ModelResidencyState::Resuming: return "resuming";
    case ModelResidencyState::Error: return "error";
    }
    throw std::logic_error("unknown model residency state");
}
inline ApiError residency_error_to_api_error(const ModelResidencyError& exception) {
    ApiError error;
    error.message = exception.what();
    switch (exception.kind()) {
    case ModelResidencyErrorKind::Busy:
        error.status = 409; error.type = "invalid_state"; error.code = "model_busy"; break;
    case ModelResidencyErrorKind::Unsupported:
        error.status = 400; error.code = "model_suspend_disabled"; break;
    case ModelResidencyErrorKind::Failure:
        error.status = 500; error.type = "internal_error"; error.code = "model_residency_error"; break;
    }
    return error;
}
inline nlohmann::json model_residency_report(const std::string& model, const ModelResidencyStatus& s) {
    return {{"object", "model.residency"}, {"model", model}, {"enabled", s.enabled},
            {"state", residency_state_name(s.state)}, {"weight_restore_source", "artifact"},
            {"weight_device_bytes", s.weight_device_bytes}, {"persistent_device_bytes", s.persistent_device_bytes},
            {"workspace_device_bytes", s.workspace_device_bytes}, {"device_bytes", s.retained_device_bytes},
            {"retained_device_bytes", s.retained_device_bytes}, {"released_device_bytes", s.released_device_bytes},
            {"mapped_device_bytes", s.mapped_device_bytes}, {"restored_device_bytes", s.mapped_device_bytes},
            {"persistent_snapshot_bytes", s.persistent_snapshot_bytes},
            {"persistent_snapshot_capacity_bytes", s.persistent_snapshot_capacity_bytes},
            {"persistent_snapshot_pinned", s.persistent_snapshot_pinned},
            {"weight_artifact_read_bytes", s.weight_artifact_read_bytes}, {"weight_h2d_bytes", s.weight_h2d_bytes},
            {"last_suspend_seconds", s.last_suspend_seconds}, {"suspend_total_seconds", s.last_suspend_seconds},
            {"last_resume_seconds", s.last_resume_seconds}, {"resume_total_seconds", s.last_resume_seconds},
            {"restore_seconds", s.last_resume_seconds}, {"weight_restore_seconds", s.weight_restore_seconds},
            {"persistent_snapshot_d2h_seconds", s.persistent_snapshot_d2h_seconds},
            {"persistent_snapshot_h2d_seconds", s.persistent_snapshot_h2d_seconds},
            {"vmm_unmap_release_seconds", s.vmm_unmap_release_seconds}, {"vmm_map_seconds", s.vmm_map_seconds},
            {"last_error", s.last_error}};
}
} // namespace ninfer::serve
