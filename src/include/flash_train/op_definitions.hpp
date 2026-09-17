#ifndef FTRAIN_OP_DEFINITIONS_HPP_
#define FTRAIN_OP_DEFINITIONS_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/op_schema.hpp"
#include "flash_train/pattern.hpp"

namespace ftrain {

template<>
class Operand<OperandKind::kTensor> {
  public:
    using Id = FTrainTensorId;

    // Adds a tensor operand to pattern and returns its public ID.
    static Id addToPattern(Pattern& pattern) { return Id{pattern.addOperand(OperandKind::kTensor).getIndex()}; }
};

template<>
class Operand<OperandKind::kTensorList> {
  public:
    using Id = FTrainTensorListId;

    // Adds a tensor-list operand to pattern and returns its public ID.
    static Id addToPattern(Pattern& pattern) { return Id{pattern.addOperand(OperandKind::kTensorList).getIndex()}; }
};

template<>
class Operand<OperandKind::kGroupedTensor> {
  public:
    using Id = FTrainGroupedTensorId;

    // Adds a grouped-tensor operand to pattern and returns its public ID.
    static Id addToPattern(Pattern& pattern) { return Id{pattern.addOperand(OperandKind::kGroupedTensor).getIndex()}; }
};

template<>
class Operation<OperationKind::kGemm> {
  public:
    using Id       = FTrainGemmOpId;
    using TensorId = Operand<OperandKind::kTensor>::Id;

    // Adds a Gemm operation to pattern and returns its public operation ID.
    static Id addToPattern(Pattern& pattern, TensorId a, TensorId b, TensorId c, TensorId d, TensorId alpha, TensorId beta) {
        const OperationId operation_id =
            pattern.addOperation(OperationKind::kGemm,
                                 std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}},
                                 std::vector<OperandId>{OperandId{d.opaque}});
        return Id{operation_id.getIndex()};
    }
};

template<>
class Operation<OperationKind::kGroupedABCDGemm> {
  public:
    using Id        = FTrainGroupedABCDGemmOpId;
    using GroupedId = Operand<OperandKind::kGroupedTensor>::Id;
    using TensorId  = Operand<OperandKind::kTensor>::Id;

    // Adds a GroupedABCDGemm operation to pattern and returns its public
    // operation ID.
    static Id addToPattern(Pattern& pattern, GroupedId a, GroupedId b, GroupedId c, GroupedId d, TensorId alpha,
                     TensorId beta) {
        const OperationId operation_id =
            pattern.addOperation(OperationKind::kGroupedABCDGemm,
                                 std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}},
                                 std::vector<OperandId>{OperandId{d.opaque}});
        return Id{operation_id.getIndex()};
    }
};

template<>
class Operation<OperationKind::kGroupedBCDGemm> {
  public:
    using Id        = FTrainGroupedBCDGemmOpId;
    using ListId    = Operand<OperandKind::kTensorList>::Id;
    using GroupedId = Operand<OperandKind::kGroupedTensor>::Id;
    using TensorId  = Operand<OperandKind::kTensor>::Id;

    // Adds a GroupedBCDGemm operation to pattern and returns its public
    // operation ID.
    static Id addToPattern(Pattern& pattern, ListId a, GroupedId b, GroupedId c, GroupedId d, TensorId alpha, TensorId beta) {
        const OperationId operation_id =
            pattern.addOperation(OperationKind::kGroupedBCDGemm,
                                 std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}},
                                 std::vector<OperandId>{OperandId{d.opaque}});
        return Id{operation_id.getIndex()};
    }
};

template<>
class Operation<OperationKind::kGroupedABGemm> {
  public:
    using Id        = FTrainGroupedABGemmOpId;
    using GroupedId = Operand<OperandKind::kGroupedTensor>::Id;
    using ListId    = Operand<OperandKind::kTensorList>::Id;
    using TensorId  = Operand<OperandKind::kTensor>::Id;

    // Adds a GroupedABGemm operation to pattern and returns its public
    // operation ID.
    static Id addToPattern(Pattern& pattern, GroupedId a, GroupedId b, ListId c, ListId d, TensorId alpha, TensorId beta) {
        const OperationId operation_id =
            pattern.addOperation(OperationKind::kGroupedABGemm,
                                 std::vector<OperandId>{OperandId{a.opaque}, OperandId{b.opaque}, OperandId{c.opaque},
                                                        OperandId{alpha.opaque}, OperandId{beta.opaque}},
                                 std::vector<OperandId>{OperandId{d.opaque}});
        return Id{operation_id.getIndex()};
    }
};

}  // namespace ftrain

#endif
