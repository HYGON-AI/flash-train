#include "flash_train/ops/gemm/finder.hpp"
#include "flash_train/ops/gemm/problem.hpp"

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

    static GemmProblem makeProblem(const Args& args) {
        return GemmProblem{getStorageView(args, kAOperandId),
                           getStorageView(args, kBOperandId),
                           getStorageView(args, kCOperandId),
                           getStorageView(args, kDOperandId),
                           getStorageView(args, kAlphaOperandId),
                           getStorageView(args, kBetaOperandId),
                           std::get<GemmAttributes>(*args.getOpArgument(kGemmOpId))};
    }

    static void validateProblem(const GemmProblem& problem, const SelectionContext&) {
        const StorageView& a = problem.a;
        const StorageView& b = problem.b;
        const StorageView& c = problem.c;
        const StorageView& d = problem.d;
        if (a.getDims().size() != 2 || b.getDims().size() != 2 || c.getDims().size() != 2 || d.getDims().size() != 2) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires rank-two a, b, c, and d Tensors");
        }
        if (!problem.alpha.getDims().empty() || !problem.beta.getDims().empty()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires scalar alpha and beta Tensors");
        }
        if (a.getDims()[1] != b.getDims()[0]) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm a column count %lld does not match b row count %lld",
                            static_cast<long long>(a.getDims()[1]), static_cast<long long>(b.getDims()[0]));
        }
        const GemmShape shape = problem.getShape();
        const std::vector<std::int64_t> expected_output_dims{static_cast<std::int64_t>(shape.m),
                                                             static_cast<std::int64_t>(shape.n)};
        if (c.getDims() != expected_output_dims || d.getDims() != expected_output_dims) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm c and d shapes must both be m-by-n");
        }

        std::uint64_t matrix_bytes = 0;
        if (gemmMatrixBytesOverflow(shape.m, shape.k, matrix_bytes) ||
            gemmMatrixBytesOverflow(shape.k, shape.n, matrix_bytes) ||
            gemmMatrixBytesOverflow(shape.m, shape.n, matrix_bytes)) {
            throw Exception(FTRAIN_STATUS_OVERFLOW, "Gemm matrix byte size overflows uint64_t");
        }
    }
};

// The Gemm engine: the family policy above plus its selection policies.
using GemmOpsEngine = OpsEngine<GemmFamily, RegistrationOrderFinder<GemmProblem>>;

}  // namespace

std::shared_ptr<OpsEngineBase> makeGemmOpsEngine() { return std::make_shared<GemmOpsEngine>(); }

}  // namespace ftrain
