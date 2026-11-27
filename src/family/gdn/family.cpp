// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "flash_train/family/gdn/family.hpp"
#if FTRAIN_HAS_GFX936
#include "flash_train/primitive/hygon/gdn/fused_forward.hpp"
#include "flash_train/primitive/hygon/gdn/four_forward.hpp"
#include "flash_train/primitive/hygon/gdn/chunk_backward.hpp"
#endif
#if FTRAIN_HAS_GFX938
#include "flash_train/primitive/hygon/gdn/fused_forward_gfx938.hpp"
#include "flash_train/primitive/hygon/gdn/chunk_backward_gfx938.hpp"
#endif
#if FTRAIN_HAS_GFX936
#include "flash_train/primitive/hygon/gdn2/chunk_forward.hpp"
#include "flash_train/primitive/hygon/gdn2/chunk_backward.hpp"
#endif
#if FTRAIN_HAS_GFX938
#include "flash_train/primitive/hygon/gdn2/chunk_forward_gfx938.hpp"
#include "flash_train/primitive/hygon/gdn2/chunk_backward_gfx938.hpp"
#endif

namespace ftrain {

std::vector<std::shared_ptr<const Primitive<GdnForwardProblem>>> GdnForwardFamily::makeRecords() {
    std::vector<std::shared_ptr<const Primitive<GdnForwardProblem>>> records;
#if FTRAIN_HAS_GFX936
    records.push_back(std::make_shared<GdnFourForward>());
    records.push_back(std::make_shared<GdnFusedForward>(FTRAIN_NUMERIC_TYPE_BF16));
    records.push_back(std::make_shared<GdnFusedForward>(FTRAIN_NUMERIC_TYPE_FP32));
#endif
#if FTRAIN_HAS_GFX938
    records.push_back(std::make_shared<GdnFusedForwardGfx938>(FTRAIN_NUMERIC_TYPE_BF16));
    records.push_back(std::make_shared<GdnFusedForwardGfx938>(FTRAIN_NUMERIC_TYPE_FP32));
#endif
    return records;
}

std::vector<std::shared_ptr<const Primitive<GdnBackwardProblem>>> GdnBackwardFamily::makeRecords() {
    std::vector<std::shared_ptr<const Primitive<GdnBackwardProblem>>> records;
#if FTRAIN_HAS_GFX936
    records.push_back(std::make_shared<GdnChunkBackward>());
#endif
#if FTRAIN_HAS_GFX938
    records.push_back(std::make_shared<GdnChunkBackwardGfx938>());
#endif
    return records;
}

std::vector<std::shared_ptr<const Primitive<Gdn2ForwardProblem>>> Gdn2ForwardFamily::makeRecords() {
    std::vector<std::shared_ptr<const Primitive<Gdn2ForwardProblem>>> records;
#if FTRAIN_HAS_GFX936
    records.push_back(std::make_shared<Gdn2ChunkForward>());
#endif
#if FTRAIN_HAS_GFX938
    records.push_back(std::make_shared<Gdn2ChunkForwardGfx938>());
#endif
    return records;
}

std::vector<std::shared_ptr<const Primitive<Gdn2BackwardProblem>>> Gdn2BackwardFamily::makeRecords() {
    std::vector<std::shared_ptr<const Primitive<Gdn2BackwardProblem>>> records;
#if FTRAIN_HAS_GFX936
    records.push_back(std::make_shared<Gdn2ChunkBackward>());
#endif
#if FTRAIN_HAS_GFX938
    records.push_back(std::make_shared<Gdn2ChunkBackwardGfx938>());
#endif
    return records;
}

}  // namespace ftrain
