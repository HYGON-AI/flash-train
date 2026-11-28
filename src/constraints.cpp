// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "flash_train/control.h"

#if FTRAIN_PLATFORM_HYGON_HIP
#    include <hip/hip_runtime_api.h>
#endif

#include "flash_train/error.hpp"
#include "flash_train/constraints.hpp"

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

Constraints::Constraints(FTrainDeviceId device_id) : device_id_(device_id) {
#if FTRAIN_PLATFORM_HYGON_HIP
    hipDeviceProp_t properties{};
    const hipError_t status = hipGetDeviceProperties(&properties, device_id);
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "hipGetDeviceProperties failed: %s", hipGetErrorString(status));
    }
    arch_ = properties.gcnArchName;
    const auto separator = arch_.find(':');
    if (separator != std::string::npos) arch_.resize(separator);
#else
#    error "Constraints arch query is not implemented for the selected platform"
#endif
}

}  // namespace ftrain
