// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_PRIMITIVE_HYGON_ARCH_HPP_
#define FTRAIN_PRIMITIVE_HYGON_ARCH_HPP_

#include "flash_train/constraints.hpp"

namespace ftrain {

inline bool isGfx936(const Constraints& constraints) noexcept { return constraints.getArch() == "gfx936"; }

inline bool isGfx938(const Constraints& constraints) noexcept { return constraints.getArch() == "gfx938"; }

}  // namespace ftrain

#endif
