#include "flash_train/engine/base.hpp"

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

}  // namespace ftrain
