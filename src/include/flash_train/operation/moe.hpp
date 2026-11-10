#ifndef FTRAIN_OPERATION_MOE_HPP_
#define FTRAIN_OPERATION_MOE_HPP_

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "flash_train/flash_train.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

class MoeSession;

struct MoeTensorMetadata final {
    const void* identity = nullptr;
    uint64_t version = 0;
};

template<OperationKind Kind>
struct MoeAttributes final {
    std::shared_ptr<MoeSession> session;
    std::array<MoeTensorMetadata, Kind == OperationKind::kAGGroupGemm ? 4 : 5> tensors{};
};

using AGGroupGemmAttributes = MoeAttributes<OperationKind::kAGGroupGemm>;
using MoEReduceRSAttributes = MoeAttributes<OperationKind::kMoEReduceRS>;

template<OperationKind Kind, typename RoleId>
class MoeOperationTraits {
  public:
    using Id = RoleId;
    using Type = MoeAttributes<Kind>;

    static constexpr PatternOperationId createPatternOperationId(Id id) noexcept {
        return PatternOperationId{id.opaque};
    }

    template<typename... InputIds>
    static PatternOperationNode makeNode(FTrainTensorId output, InputIds... inputs) {
        return PatternOperationNode{Kind, {OperandTraits<OperandKind::kTensor>::createPatternOperandId(inputs)...},
                                     {OperandTraits<OperandKind::kTensor>::createPatternOperandId(output)}};
    }
};

template<>
class OperationTraits<OperationKind::kAGGroupGemm>
    : public MoeOperationTraits<OperationKind::kAGGroupGemm, FTrainAGGroupGemmOpId> {
  public:
    static PatternOperationNode createPatternOperationNode(FTrainTensorId input, FTrainTensorId weight,
                                                           FTrainTensorId routing, FTrainTensorId output) {
        return makeNode(output, input, weight, routing);
    }
};

template<>
class OperationTraits<OperationKind::kMoEReduceRS>
    : public MoeOperationTraits<OperationKind::kMoEReduceRS, FTrainMoEReduceRSOpId> {
  public:
    static PatternOperationNode createPatternOperationNode(FTrainTensorId input, FTrainTensorId weight,
                                                           FTrainTensorId routing, FTrainTensorId expert_weight,
                                                           FTrainTensorId output) {
        return makeNode(output, input, weight, routing, expert_weight);
    }
};

}

#endif
