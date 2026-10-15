// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/h_aggregate_fwd/fp32_h_aggregate_fwd.hpp"
#include "flash_train/primitive/hygon/h_aggregate_fwd/mixed_fp32_h_aggregate_fwd.hpp"
#include "flash_train/family/h_aggregate_fwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<HAggregateFwdProblem>>> HAggregateFwdFamily::makeRecords() {
    return {std::make_shared<MixedFp32HAggregateFwd>(), std::make_shared<Fp32HAggregateFwd>()};
}

}  // namespace ftrain
