#include "flash_train/ops/gemm/finder.hpp"
#include "flash_train/ops/gemm/problem.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/engine/gemm/gemm_engine.hpp"
#include "flash_train/operation/operation.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive/primitive.hpp"
#include "flash_train/binding.hpp"
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
    return std::get<Tensor>(args.getOperand(operand_id)).getStorage().getStorageView();
}

enum class GemmSelectionTokenTag : std::uint64_t {
    kFormatVersion = 1,
    kA,
    kB,
    kC,
    kAlpha,
    kBeta,
    kD,
    kDimensions,
    kStrides,
    kComputeType,
    kMemoryRelations,
};

void appendSignedVector(GemmSelectionTokenTag tag, const std::vector<std::int64_t>& operands,
                        std::vector<std::uint64_t>& tokens) {
    tokens.push_back(static_cast<std::uint64_t>(tag));
    tokens.push_back(static_cast<std::uint64_t>(operands.size()));
    for (const std::int64_t operand : operands) { tokens.push_back(static_cast<std::uint64_t>(operand)); }
}

void appendStorageViewTokens(GemmSelectionTokenTag tag, const StorageView& view, std::size_t required_alignment,
                             std::vector<std::uint64_t>& tokens) {
    const bool has_memory = view.getMemory() != nullptr;
    const std::uintptr_t address_class =
        has_memory ? reinterpret_cast<std::uintptr_t>(view.getMemory()) % required_alignment : 0;

    tokens.push_back(static_cast<std::uint64_t>(tag));
    tokens.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    tokens.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    tokens.push_back(static_cast<std::uint64_t>(view.isHostMemory()));
    tokens.push_back(static_cast<std::uint64_t>(has_memory));
    tokens.push_back(static_cast<std::uint64_t>(address_class));
    appendSignedVector(GemmSelectionTokenTag::kDimensions, view.getDims(), tokens);
    appendSignedVector(GemmSelectionTokenTag::kStrides, view.getStrides(), tokens);
}

PatternBuilder makeSupportedPattern() {
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

class GemmOpsEngine final : public OpsEngine<GemmProblem> {
  public:
    GemmOpsEngine()
        : OpsEngine(makeSupportedPattern().buildPattern(), PrimitiveList{std::make_shared<Fp32Gemm>()},
                    std::vector<std::shared_ptr<const Finder>>{std::make_shared<RegistrationOrderFinder>()}) {}

  private:
    void validateProblem(const GemmProblem& problem, const SelectionContext&) const override {
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

    GemmProblem makeProblem(const Args& args) const override {
        return GemmProblem{getStorageView(args, kAOperandId),
                           getStorageView(args, kBOperandId),
                           getStorageView(args, kCOperandId),
                           getStorageView(args, kDOperandId),
                           getStorageView(args, kAlphaOperandId),
                           getStorageView(args, kBetaOperandId),
                           std::get<GemmAttributes>(args.getOpArgument(kGemmOpId).getAttributes())};
    }

    std::vector<std::uint64_t> makeSelectionTokens(const GemmProblem& problem, const SelectionContext&) const override {
        std::vector<std::uint64_t> tokens;
        tokens.reserve(96);
        tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kFormatVersion));
        tokens.push_back(1);
        appendStorageViewTokens(GemmSelectionTokenTag::kA, problem.a, alignof(float), tokens);
        appendStorageViewTokens(GemmSelectionTokenTag::kB, problem.b, alignof(float), tokens);
        appendStorageViewTokens(GemmSelectionTokenTag::kC, problem.c, alignof(float), tokens);
        appendStorageViewTokens(GemmSelectionTokenTag::kAlpha, problem.alpha, alignof(float), tokens);
        appendStorageViewTokens(GemmSelectionTokenTag::kBeta, problem.beta, alignof(float), tokens);
        appendStorageViewTokens(GemmSelectionTokenTag::kD, problem.d, alignof(float), tokens);
        tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kComputeType));
        tokens.push_back(static_cast<std::uint64_t>(problem.attributes.getComputeType()));

        const MemoryRelations relations = getMemoryRelations(problem);
        tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kMemoryRelations));
        tokens.push_back(static_cast<std::uint64_t>(relations.has_range_overflow));
        tokens.push_back(static_cast<std::uint64_t>(relations.output_input_overlap));
        return tokens;
    }
};

}  // namespace

std::shared_ptr<OpsEngineBase> makeGemmOpsEngine() { return std::make_shared<GemmOpsEngine>(); }

}  // namespace ftrain
