// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "flash_train/control.h"

#if FTRAIN_PLATFORM_HYGON_HIP
#    include <roctx.h>
#endif

#include "flash_train/trace.hpp"

namespace ftrain {

ScopedTraceRange::ScopedTraceRange(const char* message) noexcept {
#if FTRAIN_PLATFORM_HYGON_HIP
    active_ = roctxRangePushA(message) >= 0;
#else
#    error "ScopedTraceRange construction is not implemented for the selected platform"
#endif
}

ScopedTraceRange::~ScopedTraceRange() noexcept {
#if FTRAIN_PLATFORM_HYGON_HIP
    if (active_) { (void)roctxRangePop(); }
#else
#    error "ScopedTraceRange destruction is not implemented for the selected platform"
#endif
}

void markTraceEvent(const char* message) noexcept {
#if FTRAIN_PLATFORM_HYGON_HIP
    roctxMarkA(message);
#else
#    error "markTraceEvent is not implemented for the selected platform"
#endif
}

}  // namespace ftrain
