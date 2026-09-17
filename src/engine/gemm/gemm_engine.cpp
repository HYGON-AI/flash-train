#include "flash_train/finder/gemm/finder.hpp"
#include "flash_train/problem/gemm/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/engine/gemm/gemm_engine.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive/gemm/fp32_gemm.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

constexpr PatternOperandId kAOperandId{0};
constexpr PatternOperandId kBOperandId{1};
constexpr PatternOperandId kCOperandId{2};
constexpr PatternOperandId kAlphaOperandId{3};
constexpr PatternOperandId kBetaOperandId{4};
constexpr PatternOperandId kDOperandId{5};
constexpr PatternOperationId kGemmOpId{0};

const StorageView& getStorageView(const Args& args, PatternOperandId operand_id) {
    return std::get<Tensor>(*args.getOperand(operand_id)).getStorageView();
}

// Everything the Gemm family contributes to its engine; OpsEngine consumes
// exactly these static functions (the OpsEngine comment carries the
// contract).
struct GemmFamily {
    using Problem = GemmProblem;

    static PatternBuilder makeSupportedPattern() {
        PatternBuilder pattern;
        const FTrainTensorId a     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId b     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
        const FTrainTensorId d     = pattern.addOperand<OperandKind::kTensor>();
        static_cast<void>(pattern.addOperation<OperationKind::kGemm>(a, b, c, d, alpha, beta));
        return pattern;
    }

    static std::vector<std::shared_ptr<const Primitive<GemmProblem>>> makeRecords() {
        return {std::make_shared<Fp32Gemm>()};
    }

    // No validateProblem override: the GemmProblem constructor performs
    // the family's cross-field validation (see its comment).
    static GemmProblem makeProblem(const Args& args) {
        return GemmProblem{getStorageView(args, kAOperandId),
                           getStorageView(args, kBOperandId),
                           getStorageView(args, kCOperandId),
                           getStorageView(args, kDOperandId),
                           getStorageView(args, kAlphaOperandId),
                           getStorageView(args, kBetaOperandId),
                           std::get<GemmAttributes>(*args.getOpArgument(kGemmOpId))};
    }
};

// The Gemm engine: the family policy above plus its selection policies.
using GemmOpsEngine = OpsEngine<GemmFamily, RegistrationOrderFinder<GemmProblem>>;

}  // namespace

std::shared_ptr<OpsEngineBase> makeGemmOpsEngine() { return std::make_shared<GemmOpsEngine>(); }

}  // namespace ftrain
