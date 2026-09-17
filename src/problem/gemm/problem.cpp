#include "flash_train/problem/gemm/problem.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "flash_train/error.hpp"

namespace ftrain {
namespace {

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

// Computes the byte size of a rows-by-columns float block. Returns false
// and leaves bytes untouched when the size overflows std::uint64_t.
bool gemmMatrixBytesOverflow(std::uint64_t rows, std::uint64_t columns, std::uint64_t& bytes) noexcept {
    if (rows != 0 && columns > std::numeric_limits<std::uint64_t>::max() / rows) { return true; }
    const std::uint64_t elements = rows * columns;
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) { return true; }
    bytes = elements * sizeof(float);
    return false;
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

GemmSelectionTokenTag portTokenTag(GemmPort port) noexcept {
    switch (port) {
        case GemmPort::kA: return GemmSelectionTokenTag::kA;
        case GemmPort::kB: return GemmSelectionTokenTag::kB;
        case GemmPort::kC: return GemmSelectionTokenTag::kC;
        case GemmPort::kD: return GemmSelectionTokenTag::kD;
        case GemmPort::kAlpha: return GemmSelectionTokenTag::kAlpha;
        case GemmPort::kBeta: return GemmSelectionTokenTag::kBeta;
    }
    return GemmSelectionTokenTag::kA;
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

}  // namespace

GemmProblem::GemmProblem(StorageView a, StorageView b, StorageView c, StorageView d, StorageView alpha,
                         StorageView beta, GemmAttributes attributes)
    : ports_{std::move(a), std::move(b), std::move(c), std::move(d), std::move(alpha), std::move(beta)},
      attributes_(attributes) {
    const StorageView& a_view     = view(GemmPort::kA);
    const StorageView& b_view     = view(GemmPort::kB);
    const StorageView& c_view     = view(GemmPort::kC);
    const StorageView& d_view     = view(GemmPort::kD);
    const StorageView& alpha_view = view(GemmPort::kAlpha);
    const StorageView& beta_view  = view(GemmPort::kBeta);

    if (a_view.getDims().size() != 2 || b_view.getDims().size() != 2 || c_view.getDims().size() != 2 ||
        d_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires rank-two a, b, c, and d Tensors");
    }
    if (!alpha_view.getDims().empty() || !beta_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires scalar alpha and beta Tensors");
    }

    const GemmShape shape = getShape();
    if (a_view.getDims()[1] != b_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm a column count %lld does not match b row count %lld",
                        static_cast<long long>(a_view.getDims()[1]), static_cast<long long>(b_view.getDims()[0]));
    }
    const std::vector<std::int64_t> expected_output_dims{static_cast<std::int64_t>(shape.m),
                                                         static_cast<std::int64_t>(shape.n)};
    if (c_view.getDims() != expected_output_dims || d_view.getDims() != expected_output_dims) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm c and d shapes must both be m-by-n");
    }

    std::uint64_t matrix_bytes = 0;
    if (gemmMatrixBytesOverflow(shape.m, shape.k, matrix_bytes) ||
        gemmMatrixBytesOverflow(shape.k, shape.n, matrix_bytes) ||
        gemmMatrixBytesOverflow(shape.m, shape.n, matrix_bytes)) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "Gemm matrix byte size overflows uint64_t");
    }
}

GemmShape GemmProblem::getShape() const noexcept {
    const StorageView& a_view = view(GemmPort::kA);
    const StorageView& b_view = view(GemmPort::kB);
    return GemmShape{static_cast<std::uint64_t>(a_view.getDims()[0]), static_cast<std::uint64_t>(b_view.getDims()[1]),
                     static_cast<std::uint64_t>(a_view.getDims()[1])};
}

bool GemmProblem::hasEmptyOutput() const noexcept { return getShape().m * getShape().n == 0; }

FTrainNumericType GemmProblem::getComputeType() const noexcept { return attributes_.getComputeType(); }

FTrainNumericType GemmProblem::getNumericType(GemmPort port) const noexcept { return view(port).getNumericType(); }

FTrainIndexType GemmProblem::getIndexType(GemmPort port) const noexcept { return view(port).getIndexType(); }

const std::vector<std::int64_t>& GemmProblem::getStrides(GemmPort port) const noexcept {
    return view(port).getStrides();
}

bool GemmProblem::isHostMemory(GemmPort port) const noexcept { return view(port).isHostMemory(); }

void* GemmProblem::getAddress(GemmPort port) const noexcept { return view(port).getMemory(); }

MemoryRelations GemmProblem::getMemoryRelations() const {
    const GemmShape shape = getShape();
    std::uint64_t a_bytes = 0;
    std::uint64_t b_bytes = 0;
    std::uint64_t c_bytes = 0;
    std::uint64_t d_bytes = 0;
    if (gemmMatrixBytesOverflow(shape.m, shape.k, a_bytes) || gemmMatrixBytesOverflow(shape.k, shape.n, b_bytes) ||
        gemmMatrixBytesOverflow(shape.m, shape.n, c_bytes) || gemmMatrixBytesOverflow(shape.m, shape.n, d_bytes)) {
        return MemoryRelations{true, false};
    }

    const void* a_memory     = getAddress(GemmPort::kA);
    const void* b_memory     = getAddress(GemmPort::kB);
    const void* c_memory     = getAddress(GemmPort::kC);
    const void* alpha_memory = getAddress(GemmPort::kAlpha);
    const void* beta_memory  = getAddress(GemmPort::kBeta);
    void* d_memory           = getAddress(GemmPort::kD);

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

std::vector<std::uint64_t> GemmProblem::getSelectionTokens() const {
    std::vector<std::uint64_t> tokens;
    tokens.reserve(96);
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kFormatVersion));
    tokens.push_back(1);
    for (const GemmPort port : kAllGemmPorts) {
        appendStorageViewTokens(portTokenTag(port), view(port), alignof(float), tokens);
    }
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kComputeType));
    tokens.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));

    const MemoryRelations relations = getMemoryRelations();
    tokens.push_back(static_cast<std::uint64_t>(GemmSelectionTokenTag::kMemoryRelations));
    tokens.push_back(static_cast<std::uint64_t>(relations.has_range_overflow));
    tokens.push_back(static_cast<std::uint64_t>(relations.output_input_overlap));
    return tokens;
}

}  // namespace ftrain
