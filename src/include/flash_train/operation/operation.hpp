// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_OPERATION_OPERATION_HPP_
#define FTRAIN_OPERATION_OPERATION_HPP_

#include <variant>

#include "flash_train/operation/gemm.hpp"

namespace ftrain {

using OperationValue =
    std::variant<GemmAttributes, GroupedABCDGemmAttributes, GroupedBCDGemmAttributes, GroupedABGemmAttributes>;

}  // namespace ftrain

#endif
