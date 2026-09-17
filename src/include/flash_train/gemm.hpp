#ifndef FTRAIN_GEMM_HPP_
#define FTRAIN_GEMM_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/op_schema.hpp"
#include "flash_train/storage_view.hpp"

namespace ftrain {

class OpsEngineBase;

// GEMM extents: m rows, n columns, k reduction depth.
struct GemmShape {
    std::uint64_t m;
    std::uint64_t n;
    std::uint64_t k;
};

// One Gemm call's parameters, describing d = alpha * (a @ b) + beta * c.
// Assembled by the engine from Args; the views are non-owning and stay valid
// for the call.
struct GemmProblem {
    StorageView a;
    StorageView b;
    StorageView c;
    StorageView d;
    StorageView alpha;
    StorageView beta;
    GemmAttributes attributes;

    // Returns {m, n, k}, where m = a's rows, k = a's columns (= b's rows),
    // and n = b's columns.
    GemmShape getShape() const noexcept {
        return GemmShape{static_cast<std::uint64_t>(a.getDims().empty() ? 0 : a.getDims()[0]),
                         static_cast<std::uint64_t>(b.getDims().empty()      ? 0
                                                    : b.getDims().size() > 1 ? b.getDims()[1]
                                                                             : 0),
                         static_cast<std::uint64_t>(a.getDims().size() > 1 ? a.getDims()[1] : 0)};
    }
};

// Creates the built-in Gemm engine: six Tensor operands (a, b, c, alpha,
// beta, d) and one Gemm operation. Allocation failure throws std::bad_alloc.
std::shared_ptr<OpsEngineBase> makeGemmOpsEngine();

// Enqueues d = alpha * (a @ b) + beta * c on stream: a is m-by-k, b is
// k-by-n, c and d are m-by-n, all row-major and contiguous; alpha and beta
// each address one float. When m * n is nonzero, every pointer must be
// non-null and float-aligned, and d must not overlap any input. The caller
// keeps the memory valid until the stream's work completes. m * n == 0
// returns without launching. A launch failure throws Exception with
// FTRAIN_STATUS_INTERNAL_ERROR. The stream is not synchronized.
void launchFp32Gemm(const float* a, const float* b, const float* c, const float* alpha, const float* beta, float* d,
                    std::uint64_t m, std::uint64_t n, std::uint64_t k, FTrainStream stream);

}  // namespace ftrain

#endif
