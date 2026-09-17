#include "flash_train/ops_engine.hpp"

#include "flash_train/control.h"

#if FTRAIN_PLATFORM_HYGON_HIP
#    include <hip/hip_runtime_api.h>
#endif

#include "flash_train/error.hpp"

namespace ftrain {

FTrainDeviceId getCurrentDeviceId() {
#if FTRAIN_PLATFORM_HYGON_HIP
    int device_id           = 0;
    const hipError_t status = hipGetDevice(&device_id);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDevice failed: %s", hipGetErrorString(status));
    }
    return static_cast<FTrainDeviceId>(device_id);
#else
#    error "getCurrentDeviceId is not implemented for the selected platform"
#endif
}

OpsEngineBase::OpsEngineBase(const Pattern& supported_pattern) : supported_pattern_(supported_pattern) {}

OpsEngineBase::~OpsEngineBase() = default;

std::optional<RoleMapping> matchRoles(const Pattern& user_pattern, const Pattern& supported_pattern) {
    if (user_pattern.getKey() != supported_pattern.getKey()) { return std::nullopt; }

    if (const std::optional<MatchResult> fast = Matcher::matchBySignature(user_pattern, supported_pattern)) {
        return RoleMapping::fromMatchResult(*fast);
    }
    if (const std::optional<MatchResult> exact = Matcher::match(user_pattern, supported_pattern)) {
        return RoleMapping::fromMatchResult(*exact);
    }
    return std::nullopt;
}

}  // namespace ftrain
