#include "flash_train/problem/gemm/problem.hpp"

#include <cstdint>
#include <limits>
#include <vector>

#include "flash_train/error.hpp"

namespace ftrain {

GemmProblem::GemmProblem(Tensor a, Tensor b, Tensor c, Tensor d, Tensor alpha, Tensor beta, GemmAttributes attributes)
    : a_(std::move(a)), b_(std::move(b)), c_(std::move(c)), d_(std::move(d)), alpha_(std::move(alpha)),
      beta_(std::move(beta)), attributes_(attributes) {
    const StorageView& a_view     = a_.getStorageView();
    const StorageView& b_view     = b_.getStorageView();
    const StorageView& c_view     = c_.getStorageView();
    const StorageView& d_view     = d_.getStorageView();
    const StorageView& alpha_view = alpha_.getStorageView();
    const StorageView& beta_view  = beta_.getStorageView();

    if (a_view.getDims().size() != 2 || b_view.getDims().size() != 2 || c_view.getDims().size() != 2 ||
        d_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires rank-two a, b, c, and d Tensors");
    }
    if (!alpha_view.getDims().empty() || !beta_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires scalar alpha and beta Tensors");
    }
    if (a_view.isHostMemory() || b_view.isHostMemory() || c_view.isHostMemory() || d_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires a, b, c, and d in device memory");
    }
    if (!alpha_view.isHostMemory() || !beta_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Gemm requires alpha and beta in host memory");
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

    // Every matrix's byte size must fit std::uint64_t, so each element
    // count must stay within the float-element budget.
    const std::uint64_t max_matrix_elements = std::numeric_limits<std::uint64_t>::max() / sizeof(float);
    if ((shape.k != 0 && shape.m > max_matrix_elements / shape.k) ||
        (shape.n != 0 && shape.k > max_matrix_elements / shape.n) ||
        (shape.n != 0 && shape.m > max_matrix_elements / shape.n)) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "Gemm matrix byte size overflows uint64_t");
    }
}

std::vector<std::uint64_t> GemmProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(96);
    std::uint64_t field = 0;

    const StorageView* const views[6] = {&a_.getStorageView(), &b_.getStorageView(),     &c_.getStorageView(),
                                         &d_.getStorageView(), &alpha_.getStorageView(), &beta_.getStorageView()};
    for (const StorageView* const view : views) {
        const bool has_memory = view->getMemory() != nullptr;

        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(view->getNumericType()));
        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(view->getIndexType()));
        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(has_memory));
        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(
            has_memory ? reinterpret_cast<std::uintptr_t>(view->getMemory()) % alignof(float) : 0));

        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(view->getDims().size()));
        for (const std::int64_t dim : view->getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
        key.push_back(++field);
        key.push_back(static_cast<std::uint64_t>(view->getStrides().size()));
        for (const std::int64_t stride : view->getStrides()) { key.push_back(static_cast<std::uint64_t>(stride)); }
    }

    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(++field);
    key.push_back(static_cast<std::uint64_t>(isInPlace()));
    return key;
}

}  // namespace ftrain
