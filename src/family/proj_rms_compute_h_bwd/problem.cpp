// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

#include "flash_train/error.hpp"
#include "flash_train/family/proj_rms_compute_h_bwd/problem.hpp"

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

constexpr const char* kOpName      = "ProjRmsComputeHBwd";
constexpr std::int64_t kAlphaCount = 3;

}  // namespace

ProjRmsComputeHBwdProblem::ProjRmsComputeHBwdProblem(Tensor grad_h_pre, Tensor grad_h_post, Tensor grad_h_res,
                                                     Tensor grad_r, Tensor h_pre, Tensor h_post, Tensor proj, Tensor r,
                                                     Tensor x, Tensor weight, Tensor alpha_pre, Tensor alpha_post,
                                                     Tensor alpha_res, Tensor grad_x, Tensor grad_weight,
                                                     Tensor grad_alpha, Tensor grad_bias, Tensor n, Tensor eps,
                                                     Tensor compute_h_eps, ProjRmsComputeHBwdAttributes attributes)
    : grad_h_pre_(std::move(grad_h_pre)), grad_h_post_(std::move(grad_h_post)), grad_h_res_(std::move(grad_h_res)),
      grad_r_(std::move(grad_r)), h_pre_(std::move(h_pre)), h_post_(std::move(h_post)), proj_(std::move(proj)),
      r_(std::move(r)), x_(std::move(x)), weight_(std::move(weight)), alpha_pre_(std::move(alpha_pre)),
      alpha_post_(std::move(alpha_post)), alpha_res_(std::move(alpha_res)), grad_x_(std::move(grad_x)),
      grad_weight_(std::move(grad_weight)), grad_alpha_(std::move(grad_alpha)), grad_bias_(std::move(grad_bias)),
      n_(std::move(n)), eps_(std::move(eps)), compute_h_eps_(std::move(compute_h_eps)), attributes_(attributes) {
    const StorageView& n_view             = n_.getStorageView();
    const StorageView& eps_view           = eps_.getStorageView();
    const StorageView& compute_h_eps_view = compute_h_eps_.getStorageView();
    const StorageView& x_view             = x_.getStorageView();
    const StorageView& weight_view        = weight_.getStorageView();
    const StorageView& grad_alpha_view    = grad_alpha_.getStorageView();
    const StorageView& grad_bias_view     = grad_bias_.getStorageView();
    const StorageView& proj_view          = proj_.getStorageView();
    const StorageView& grad_x_view        = grad_x_.getStorageView();
    const StorageView& grad_weight_view   = grad_weight_.getStorageView();
    const StorageView& r_view             = r_.getStorageView();
    const StorageView& grad_r_view        = grad_r_.getStorageView();
    const StorageView& grad_h_res_view    = grad_h_res_.getStorageView();

    if (!n_view.getDims().empty() || !eps_view.getDims().empty() || !compute_h_eps_view.getDims().empty()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires rank-zero scalar Tensors", kOpName);
    }
    if (!n_view.isHostMemory() || !eps_view.isHostMemory() || !compute_h_eps_view.isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s requires scalar Tensors in host memory", kOpName);
    }

    for (const Tensor* port :
         {&grad_h_pre_,  &grad_h_post_, &grad_h_res_, &grad_r_,    &h_pre_,      &h_post_,       &proj_,
          &r_,           &x_,           &weight_,     &alpha_pre_, &alpha_post_, &alpha_res_,    &grad_x_,
          &grad_weight_, &grad_alpha_,  &grad_bias_,  &n_,         &eps_,        &compute_h_eps_}) {
        const auto& view    = port->getStorageView();
        std::uint64_t bytes = elementBytes(view.getNumericType());
        for (const std::int64_t dim : view.getDims()) {
            if (dim != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim)) {
                throw Exception(FTRAIN_STATUS_OVERFLOW, "%s: tensor byte size overflows uint64_t", kOpName);
            }
            bytes *= static_cast<std::uint64_t>(dim);
        }
    }

    // n participates in problem validation or implementation selection.
    if (n_view.getNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: n has the wrong scalar type", kOpName);
    }
    std::memcpy(&n_value_, n_view.getMemory(), sizeof(n_value_));
    // eps participates in problem validation or implementation selection.
    if (eps_view.getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps has the wrong scalar type", kOpName);
    }
    std::memcpy(&eps_value_, eps_view.getMemory(), sizeof(eps_value_));
    // compute_h_eps participates in problem validation or implementation selection.
    if (compute_h_eps_view.getNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: compute_h_eps has the wrong scalar type", kOpName);
    }
    std::memcpy(&compute_h_eps_value_, compute_h_eps_view.getMemory(), sizeof(compute_h_eps_value_));

    for (const Tensor* const port :
         {&grad_h_pre_, &grad_h_post_, &grad_h_res_, &grad_r_, &h_pre_, &h_post_, &proj_, &r_, &x_, &weight_,
          &alpha_pre_, &alpha_post_, &alpha_res_, &grad_x_, &grad_weight_, &grad_alpha_, &grad_bias_}) {
        if (port->getStorageView().isHostMemory()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s must be in device memory", kOpName, "a port");
        }
    }

    for (const Tensor* const port : {&grad_h_pre_, &grad_h_post_, &grad_h_res_, &grad_r_, &h_pre_, &h_post_, &proj_,
                                     &r_, &grad_x_, &grad_weight_}) {
        if (port->getStorageView().getDims().size() != 2) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "a matrix port");
        }
    }
    if (x_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "x");
    }
    if (weight_view.getDims().size() != 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "weight");
    }
    for (const Tensor* const port : {&alpha_pre_, &alpha_post_, &alpha_res_}) {
        if (port->getStorageView().getDims().size() != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "an alpha scalar");
        }
        if (port->getStorageView().getDims().size() != 1 || port->getStorageView().getDims()[0] != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                            "an alpha scalar");
        }
    }
    if (grad_alpha_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_alpha");
    }
    if (grad_alpha_view.getDims()[0] != kAlphaCount) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "grad_alpha");
    }
    if (grad_bias_view.getDims().size() != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an invalid rank", kOpName, "grad_bias");
    }

    if (weight_view.getDims()[1] != x_view.getDims()[1]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "weight");
    }
    if (proj_view.getDims()[0] != x_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "proj");
    }
    if (proj_view.getDims()[1] != weight_view.getDims()[0]) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "proj");
    }
    if (grad_x_view.getDims() != x_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_x");
    }
    if (grad_weight_view.getDims() != weight_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_weight");
    }
    if (r_view.getDims().size() != 2 || r_view.getDims()[0] != dim(x_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "r");
    }
    if (r_view.getDims().size() != 2 || r_view.getDims()[1] != 1) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "r");
    }
    if (grad_r_view.getDims() != r_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent shape", kOpName, "grad_r");
    }
    if (grad_bias_view.getDims()[0] != dim(weight_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "grad_bias");
    }
    const std::int64_t streams = getNumStreams();
    if (getOutputs() <= 2 || streams <= 0 || static_cast<std::uint64_t>(streams) > (getOutputs() - 1) / 2) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: require 0 < 2*n < N", kOpName);
    }
    for (const Tensor* const port : {&h_pre_, &h_post_, &grad_h_pre_, &grad_h_post_}) {
        if (port->getStorageView().getDims().size() != 2 || port->getStorageView().getDims()[0] != dim(x_, 0)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                            "a stream port");
        }
        if (port->getStorageView().getDims().size() != 2 || port->getStorageView().getDims()[1] != streams) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName,
                            "a stream port");
        }
    }
    // The residual stream's own gradient is the only (N - 2n)-wide port: the
    // forward's h_res is not an input of the backward.
    if (grad_h_res_view.getDims().size() != 2 || grad_h_res_view.getDims()[0] != dim(x_, 0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "grad_h_res");
    }
    if (grad_h_res_view.getDims().size() != 2 ||
        grad_h_res_view.getDims()[1] != static_cast<std::int64_t>(getOutputs()) - 2 * streams) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: %s has an inconsistent dimension", kOpName, "grad_h_res");
    }
    if (getReduction() == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: the RMS reduction depth must be positive", kOpName);
    }
    if (!(getEps() > 0.0F) || !std::isfinite(getEps())) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: eps must be finite and positive", kOpName);
    }
    if (getComputeHEps() < 0.0F || !std::isfinite(getComputeHEps())) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s: compute_h_eps must be finite and non-negative", kOpName);
    }
}

std::vector<std::uint64_t> ProjRmsComputeHBwdProblem::getProblemKey() const {
    std::vector<std::uint64_t> key;
    key.reserve(352);
    key.push_back(static_cast<std::uint64_t>(attributes_.getComputeType()));
    key.push_back(static_cast<std::uint64_t>(getNumStreams()));
    // Fixed scalar prefix; each port carries lengths for variable-size fields.
    // Encode alignment/presence, never allocation addresses (same contract as Gemm).
    for (const Tensor* port :
         {&grad_h_pre_,  &grad_h_post_, &grad_h_res_, &grad_r_,    &h_pre_,      &h_post_,       &proj_,
          &r_,           &x_,           &weight_,     &alpha_pre_, &alpha_post_, &alpha_res_,    &grad_x_,
          &grad_weight_, &grad_alpha_,  &grad_bias_,  &n_,         &eps_,        &compute_h_eps_}) {
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
