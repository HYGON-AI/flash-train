#include "flash_train/gemm.hpp"
#include "flash_train/op_definitions.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/op_schema.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/runtime.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

constexpr OperandId kAOperandId{0};
constexpr OperandId kBOperandId{1};
constexpr OperandId kCOperandId{2};
constexpr OperandId kAlphaOperandId{3};
constexpr OperandId kBetaOperandId{4};
constexpr OperandId kDOperandId{5};
constexpr OperationId kGemmOpId{0};
constexpr std::uint64_t kRequiredWorkspaceBytes = 0;

const StorageView& getStorageView(const Args& args, OperandId operand_id) {
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

struct MemoryRelations {
    bool has_range_overflow;
    bool output_input_overlap;
};

bool hasGemmRanks(const GemmProblem& problem) {
    return problem.a.getDims().size() == 2 && problem.b.getDims().size() == 2 && problem.c.getDims().size() == 2 &&
           problem.d.getDims().size() == 2 && problem.alpha.getDims().empty() && problem.beta.getDims().empty();
}

bool matrixBytesOverflow(std::uint64_t rows, std::uint64_t columns, std::uint64_t& bytes) noexcept {
    if (rows != 0 && columns > std::numeric_limits<std::uint64_t>::max() / rows) { return true; }
    const std::uint64_t elements = rows * columns;
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) { return true; }
    bytes = elements * sizeof(float);
    return false;
}

bool hasRangeOverflow(const void* memory, std::uint64_t bytes) noexcept {
    if (memory == nullptr || bytes == 0) { return false; }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(memory);
    return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

bool rangesOverlap(const void* first_memory, std::uint64_t first_bytes, const void* second_memory,
                   std::uint64_t second_bytes) noexcept {
    if (first_memory == nullptr || second_memory == nullptr || first_bytes == 0 || second_bytes == 0) { return false; }
    if (hasRangeOverflow(first_memory, first_bytes) || hasRangeOverflow(second_memory, second_bytes)) { return true; }

    const std::uintptr_t first_begin  = reinterpret_cast<std::uintptr_t>(first_memory);
    const std::uintptr_t second_begin = reinterpret_cast<std::uintptr_t>(second_memory);
    const std::uintptr_t first_end    = first_begin + static_cast<std::uintptr_t>(first_bytes);
    const std::uintptr_t second_end   = second_begin + static_cast<std::uintptr_t>(second_bytes);
    return first_begin < second_end && second_begin < first_end;
}

MemoryRelations getMemoryRelations(const GemmProblem& problem) {
    if (!hasGemmRanks(problem)) { return MemoryRelations{false, false}; }

    const GemmShape shape = problem.getShape();
    std::uint64_t a_bytes = 0;
    std::uint64_t b_bytes = 0;
    std::uint64_t c_bytes = 0;
    std::uint64_t d_bytes = 0;
    if (matrixBytesOverflow(shape.m, shape.k, a_bytes) || matrixBytesOverflow(shape.k, shape.n, b_bytes) ||
        matrixBytesOverflow(shape.m, shape.n, c_bytes) || matrixBytesOverflow(shape.m, shape.n, d_bytes)) {
        return MemoryRelations{true, false};
    }

    const void* a_memory     = problem.a.getMemory();
    const void* b_memory     = problem.b.getMemory();
    const void* c_memory     = problem.c.getMemory();
    const void* alpha_memory = problem.alpha.getMemory();
    const void* beta_memory  = problem.beta.getMemory();
    void* d_memory           = problem.d.getMemory();

    const bool has_range_overflow = hasRangeOverflow(a_memory, a_bytes) || hasRangeOverflow(b_memory, b_bytes) ||
                                    hasRangeOverflow(c_memory, c_bytes) ||
                                    hasRangeOverflow(alpha_memory, sizeof(float)) ||
                                    hasRangeOverflow(beta_memory, sizeof(float)) || hasRangeOverflow(d_memory, d_bytes);
    const bool output_input_overlap = rangesOverlap(d_memory, d_bytes, a_memory, a_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, b_memory, b_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, c_memory, c_bytes) ||
                                      rangesOverlap(d_memory, d_bytes, alpha_memory, sizeof(float)) ||
                                      rangesOverlap(d_memory, d_bytes, beta_memory, sizeof(float));
    return MemoryRelations{has_range_overflow, output_input_overlap};
}

bool isAlignedForFloat(const void* memory) noexcept {
    return memory == nullptr || reinterpret_cast<std::uintptr_t>(memory) % alignof(float) == 0;
}

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

Pattern makeSupportedPattern() {
    Pattern pattern;
    const FTrainTensorId a     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId b     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId c     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId alpha = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId beta  = Operand<OperandKind::kTensor>::addToPattern(pattern);
    const FTrainTensorId d     = Operand<OperandKind::kTensor>::addToPattern(pattern);
    static_cast<void>(Operation<OperationKind::kGemm>::addToPattern(pattern, a, b, c, d, alpha, beta));
    return pattern;
}

class Fp32Gemm final : public Primitive<GemmProblem> {
  public:
    const char* getName() const noexcept override { return "Fp32Gemm"; }

    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<Fp32Gemm>(*this); }

    Result isApplicable(const GemmProblem& problem, const SelectionContext&) const override {
        // --- data type / layout / memory ---
        const StorageView* const views[6] = {&problem.a, &problem.b,     &problem.c,
                                             &problem.d, &problem.alpha, &problem.beta};
        for (const StorageView* const view : views) {
            if (view->getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
                return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not fp32");
            }
            if (view->getIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
                return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
            }
            if (view->isHostMemory()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is host memory"); }
            if (!isAlignedForFloat(view->getMemory())) {
                return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not float-aligned");
            }
        }

        // --- shape semantics / attributes ---
        if (!hasGemmRanks(problem)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "matrix operands are not rank 2"); }
        if (problem.attributes.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type is not fp32");
        }

        // --- degenerate shape fast path ---
        const GemmShape shape = problem.getShape();
        if (shape.m * shape.n == 0) { return Result{}; }

        // --- address rules ---
        const bool has_required_addresses =
            problem.c.getMemory() != nullptr && problem.alpha.getMemory() != nullptr &&
            problem.beta.getMemory() != nullptr && problem.d.getMemory() != nullptr &&
            (shape.k == 0 || (problem.a.getMemory() != nullptr && problem.b.getMemory() != nullptr));
        if (!has_required_addresses) { return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null"); }

        // --- memory relations ---
        const MemoryRelations relations = getMemoryRelations(problem);
        if (relations.has_range_overflow) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address range overflows");
        }
        if (relations.output_input_overlap) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "the output overlaps an input");
        }
        return Result{};
    }

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return kRequiredWorkspaceBytes; }

    void configure(const GemmProblem& problem) override {
        a_     = static_cast<const float*>(problem.a.getMemory());
        b_     = static_cast<const float*>(problem.b.getMemory());
        c_     = static_cast<const float*>(problem.c.getMemory());
        alpha_ = static_cast<const float*>(problem.alpha.getMemory());
        beta_  = static_cast<const float*>(problem.beta.getMemory());
        d_     = static_cast<float*>(problem.d.getMemory());
        m_     = problem.getShape().m;
        n_     = problem.getShape().n;
        k_     = problem.getShape().k;
    }

  private:
    void executeImpl(void*, std::uint64_t, FTrainStream stream) override {
        launchFp32Gemm(a_, b_, c_, alpha_, beta_, d_, m_, n_, k_, stream);
    }

    const float* a_     = nullptr;
    const float* b_     = nullptr;
    const float* c_     = nullptr;
    const float* alpha_ = nullptr;
    const float* beta_  = nullptr;
    float* d_           = nullptr;
    std::uint64_t m_    = 0;
    std::uint64_t n_    = 0;
    std::uint64_t k_    = 0;
};

class RegistrationOrderFinder final : public Finder {
  public:
    bool isEnabled(const Args&, const SelectionContext&) const override { return true; }

    PrimitiveList findCandidates(const PrimitiveList& records, const Args&, const SelectionContext&) const override {
        return records;
    }

    void sortCandidates(const Args&, const SelectionContext&, PrimitiveList&) const override {}
};

class GemmOpsEngine final : public OpsEngine<GemmProblem> {
  public:
    GemmOpsEngine()
        : OpsEngine(makeSupportedPattern(), PrimitiveList{std::make_shared<Fp32Gemm>()},
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
        if (matrixBytesOverflow(shape.m, shape.k, matrix_bytes) ||
            matrixBytesOverflow(shape.k, shape.n, matrix_bytes) ||
            matrixBytesOverflow(shape.m, shape.n, matrix_bytes)) {
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
