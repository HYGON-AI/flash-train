#ifndef FTRAIN_OPERATION_GEMM_HPP_
#define FTRAIN_OPERATION_GEMM_HPP_

#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/base.hpp"

namespace ftrain {

// Per-operation parameters. The attribute types below store
// already-validated values and perform no validation.
class GemmAttributes final {
  public:
    explicit GemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABCDGemmAttributes final {
  public:
    explicit GroupedABCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedBCDGemmAttributes final {
  public:
    explicit GroupedBCDGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

class GroupedABGemmAttributes final {
  public:
    explicit GroupedABGemmAttributes(FTrainNumericType compute_type) noexcept : compute_type_(compute_type) {}

    FTrainNumericType getComputeType() const noexcept { return compute_type_; }

  private:
    FTrainNumericType compute_type_;
};

// The typed role IDs of one Gemm operation: inputs a, b, c, alpha, and
// beta, and output d.
struct GemmTopology {
    FTrainTensorId a;
    FTrainTensorId b;
    FTrainTensorId c;
    FTrainTensorId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
};

// The typed role IDs of one GroupedABCDGemm operation: inputs a, b, c,
// alpha, and beta, and output d.
struct GroupedABCDGemmTopology {
    FTrainGroupedTensorId a;
    FTrainGroupedTensorId b;
    FTrainGroupedTensorId c;
    FTrainGroupedTensorId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
};

// The typed role IDs of one GroupedBCDGemm operation: inputs a, b, c,
// alpha, and beta, and output d.
struct GroupedBCDGemmTopology {
    FTrainTensorListId a;
    FTrainGroupedTensorId b;
    FTrainGroupedTensorId c;
    FTrainGroupedTensorId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
};

// The typed role IDs of one GroupedABGemm operation: inputs a, b, c,
// alpha, and beta, and output d.
struct GroupedABGemmTopology {
    FTrainGroupedTensorId a;
    FTrainGroupedTensorId b;
    FTrainTensorListId c;
    FTrainTensorListId d;
    FTrainTensorId alpha;
    FTrainTensorId beta;
};

template<>
class Operation<OperationKind::kGemm> {
  public:
    using Id       = FTrainGemmOpId;
    using Topology = GemmTopology;
    using Type     = GemmAttributes;

    // Returns the semantic input ports, in order a, b, c, alpha, beta.
    static std::vector<PatternOperandId> getInputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kTensor>::getPatternOperandId(topology.a),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.b),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.c),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.alpha),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.beta)};
    }

    // Returns the semantic output ports, in order d.
    static std::vector<PatternOperandId> getOutputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kTensor>::getPatternOperandId(topology.d)};
    }
};

template<>
class Operation<OperationKind::kGroupedABCDGemm> {
  public:
    using Id       = FTrainGroupedABCDGemmOpId;
    using Topology = GroupedABCDGemmTopology;
    using Type     = GroupedABCDGemmAttributes;

    // Returns the semantic input ports, in order a, b, c, alpha, beta.
    static std::vector<PatternOperandId> getInputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.a),
                Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.b),
                Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.c),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.alpha),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.beta)};
    }

    // Returns the semantic output ports, in order d.
    static std::vector<PatternOperandId> getOutputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.d)};
    }
};

template<>
class Operation<OperationKind::kGroupedBCDGemm> {
  public:
    using Id       = FTrainGroupedBCDGemmOpId;
    using Topology = GroupedBCDGemmTopology;
    using Type     = GroupedBCDGemmAttributes;

    // Returns the semantic input ports, in order a, b, c, alpha, beta.
    static std::vector<PatternOperandId> getInputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kTensorList>::getPatternOperandId(topology.a),
                Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.b),
                Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.c),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.alpha),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.beta)};
    }

    // Returns the semantic output ports, in order d.
    static std::vector<PatternOperandId> getOutputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.d)};
    }
};

template<>
class Operation<OperationKind::kGroupedABGemm> {
  public:
    using Id       = FTrainGroupedABGemmOpId;
    using Topology = GroupedABGemmTopology;
    using Type     = GroupedABGemmAttributes;

    // Returns the semantic input ports, in order a, b, c, alpha, beta.
    static std::vector<PatternOperandId> getInputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.a),
                Operand<OperandKind::kGroupedTensor>::getPatternOperandId(topology.b),
                Operand<OperandKind::kTensorList>::getPatternOperandId(topology.c),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.alpha),
                Operand<OperandKind::kTensor>::getPatternOperandId(topology.beta)};
    }

    // Returns the semantic output ports, in order d.
    static std::vector<PatternOperandId> getOutputPatternIds(const Topology& topology) {
        return {Operand<OperandKind::kTensorList>::getPatternOperandId(topology.d)};
    }
};

}  // namespace ftrain

#endif
