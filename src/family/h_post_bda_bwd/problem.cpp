// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/h_post_bda_bwd/problem.hpp"

namespace ftrain {
namespace {

std::uint64_t elementBytes(FTrainNumericType dtype) noexcept {
    switch (dtype) {
        case FTRAIN_NUMERIC_TYPE_FP64:
        case FTRAIN_NUMERIC_TYPE_INT64:
        case FTRAIN_NUMERIC_TYPE_UINT64: return 8;
        case FTRAIN_NUMERIC_TYPE_FP32:
        case FTRAIN_NUMERIC_TYPE_INT32:
        case FTRAIN_NUMERIC_TYPE_UINT32: return 4;
        case FTRAIN_NUMERIC_TYPE_FP16:
        case FTRAIN_NUMERIC_TYPE_BF16:
        case FTRAIN_NUMERIC_TYPE_INT16:
        case FTRAIN_NUMERIC_TYPE_UINT16: return 2;
        default: return 1;
    }
}

constexpr const char* kOpName = "HPostBdaBwd";

}  // namespace

HPostBdaBwdProblem::HPostBdaBwdProblem(Tensor grad_output, Tensor residual, Tensor x, Tensor bias, Tensor h_res,
                                       Tensor h_post, Tensor grad_residual, Tensor grad_x, Tensor grad_h_res,
                                       Tensor grad_h_post, Tensor grad_bias, HPostBdaBwdAttributes attributes)
    : grad_output_(std::move(grad_output)), residual_(std::move(residual)), x_(std::move(x)), bias_(std::move(bias)),
      h_res_(std::move(h_res)), h_post_(std::move(h_post)), grad_residual_(std::move(grad_residual)),
      grad_x_(std::move(grad_x)), grad_h_res_(std::move(grad_h_res)), grad_h_post_(std::move(grad_h_post)),
      grad_bias_(std::move(grad_bias)), attributes_(attributes) {
    const StorageView& grad_output_view   = grad_output_.getStorageView();
    const StorageView& residual_view      = residual_.getStorageView();
    const StorageView& x_view             = x_.getStorageView();
    const StorageView& bias_view          = bias_.getStorageView();
    const StorageView& h_res_view         = h_res_.getStorageView();
    const StorageView& h_post_view        = h_post_.getStorageView();
    const StorageView& grad_residual_view = grad_residual_.getStorageView();
    const StorageView& grad_x_view        = grad_x_.getStorageView();
    const StorageView& grad_h_res_view    = grad_h_res_.getStorageView();
    const StorageView& grad_h_post_view   = grad_h_post_.getStorageView();
    const StorageView& grad_bias_view     = grad_bias_.getStorageView();

    for (const Tensor* port : {&grad_output_, &residual_, &x_, &bias_, &h_res_, &h_post_, &grad_residual_, &grad_x_,
                               &grad_h_res_, &grad_h_post_, &grad_bias_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    if (grad_output_view.isHostMemory() || residual_view.isHostMemory() || x_view.isHostMemory() ||
        bias_view.isHostMemory() || h_res_view.isHostMemory() || h_post_view.isHostMemory() ||
        grad_residual_view.isHostMemory() || grad_x_view.isHostMemory() || grad_h_res_view.isHostMemory() ||
        grad_h_post_view.isHostMemory() || grad_bias_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires data Tensors in device memory", kOpName);
    }
    if (grad_output_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_output");
    }
    if (residual_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "residual");
    }
    if (x_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "x");
    }
    if (bias_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "bias");
    }
    if (h_res_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "h_res");
    }
    if (h_post_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "h_post");
    }
    if (grad_residual_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_residual");
    }
    if (grad_x_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_x");
    }
    if (grad_h_res_view.getDims().size() != 4) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_h_res");
    }
    if (grad_h_post_view.getDims().size() != 3) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_h_post");
    }
    if (grad_bias_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_bias");
    }

    if (grad_output_view.getDims() != residual_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_output");
    }
    if (grad_residual_view.getDims() != residual_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_residual");
    }
    if (grad_x_view.getDims() != x_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_x");
    }
    if (grad_h_res_view.getDims() != h_res_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_h_res");
    }
    if (grad_h_post_view.getDims() != h_post_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_h_post");
    }
    if (h_res_view.getDims()[2] != h_res_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_res");
    }
    if (residual_view.getDims()[2] != h_res_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "residual");
    }
    if (h_post_view.getDims()[0] != residual_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_post");
    }
    if (h_post_view.getDims()[1] != residual_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_post");
    }
    if (h_post_view.getDims()[2] != residual_view.getDims()[2]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "h_post");
    }
    if (x_view.getDims()[0] != residual_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "x");
    }
    if (x_view.getDims()[1] != residual_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "x");
    }
    if (x_view.getDims()[2] != residual_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "x");
    }
    if (bias_view.getDims()[0] != residual_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "bias");
    }
    if (grad_bias_view.getDims()[0] != residual_view.getDims()[3]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "grad_bias");
    }
}

std::vector<std::uint64_t> HPostBdaBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(224);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port : {&grad_output_, &residual_, &x_, &bias_, &h_res_, &h_post_, &grad_residual_, &grad_x_,
                               &grad_h_res_, &grad_h_post_, &grad_bias_}) {
        const auto& view = port->getStorageView();
        key.push_back(static_cast<std::uint64_t>(view.getNumericType()));
        key.push_back(static_cast<std::uint64_t>(view.getIndexType()));
        key.push_back(view.getMemory() != nullptr);
        key.push_back(reinterpret_cast<std::uintptr_t>(view.getMemory()) % elementBytes(view.getNumericType()));
        key.push_back(view.getDims().size());
        for (const auto dim : view.getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
        key.push_back(view.getStrides().size());
        for (const auto stride : view.getStrides()) { key.push_back(static_cast<std::uint64_t>(stride)); }
    }
    return key;
}

}  // namespace ftrain
