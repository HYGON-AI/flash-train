// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.hpp"
#include "flash_train/family/csa_compressor_fwd/family.hpp"

namespace ftrain {

// Fixed algorithms share the same BF16 storage contract.
std::vector<std::shared_ptr<const Primitive<CSACompressorFwdProblem>>> CSACompressorFwdFamily::makeRecords() {
    return {std::make_shared<Bf16CSACompressorFwd>(),
            std::make_shared<Bf16CSACompressorFwdCachedCapacity8>(),
            std::make_shared<Bf16CSACompressorFwdCachedCapacity16>(),
            std::make_shared<Bf16CSACompressorFwdCachedCapacity32>(),
            std::make_shared<Bf16CSACompressorFwdCachedCapacity64>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity128>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity160>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity256>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity256Dim16>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity256Dim8>(),
            std::make_shared<Bf16CSACompressorFwdTiledCapacity256Dim4>(),
            std::make_shared<Bf16CSACompressorFwdCached>(),
            std::make_shared<Bf16CSACompressorFwdTiled>(),
            std::make_shared<Bf16CSACompressorFwdR128>(),
            std::make_shared<Bf16CSACompressorFwdR128WindowMajor>()};
}

}  // namespace ftrain
