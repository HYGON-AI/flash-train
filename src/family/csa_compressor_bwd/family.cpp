// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hpp"
#include "flash_train/family/csa_compressor_bwd/family.hpp"

namespace ftrain {

// A single implementation: the kernel's element types are fixed.
std::vector<std::shared_ptr<const Primitive<CSACompressorBwdProblem>>> CSACompressorBwdFamily::makeRecords() {
    return {std::make_shared<Bf16CSACompressorBwd>(),
            std::make_shared<Bf16CSACompressorBwdCachedCapacity8>(),
            std::make_shared<Bf16CSACompressorBwdCachedCapacity16>(),
            std::make_shared<Bf16CSACompressorBwdCachedCapacity32>(),
            std::make_shared<Bf16CSACompressorBwdCachedCapacity64>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity128>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity160>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity256>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity256Dim16>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity256Dim8>(),
            std::make_shared<Bf16CSACompressorBwdTiledCapacity256Dim4>(),
            std::make_shared<Bf16CSACompressorBwdCached>(),
            std::make_shared<Bf16CSACompressorBwdTiled>(),
            std::make_shared<Bf16CSACompressorBwdR128>(),
            std::make_shared<Bf16CSACompressorBwdR128Wide>(),
            std::make_shared<Bf16CSACompressorBwdShortRows1>(),
            std::make_shared<Bf16CSACompressorBwdShortRows4>()};
}

}  // namespace ftrain
