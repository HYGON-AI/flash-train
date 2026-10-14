// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/family/moe_reduce_rs/problem.hpp>

#include <flash_train/error.hpp>

#include <cstdint>
#include <utility>

namespace ftrain {
namespace {

void appendTensorKey(std::vector<std::uint64_t>& key, const Tensor& tensor) {
    const auto& view = tensor.getStorageView();
    key.push_back(static_cast<std::uint64_t>(view.getNumericType()));
    key.push_back(static_cast<std::uint64_t>(view.getIndexType()));
    key.push_back(static_cast<std::uint64_t>(view.getDims().size()));
    for (const auto dim : view.getDims()) { key.push_back(static_cast<std::uint64_t>(dim)); }
    for (const auto stride : view.getStrides()) { key.push_back(static_cast<std::uint64_t>(stride)); }
}

void validateRank(std::uint64_t session_id, const Tensor& tensor, std::size_t rank, const char* name) {
    if (session_id == 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS session id is zero"); }
    if (tensor.getStorageView().getDims().size() != rank) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS %s must have rank %zu", name, rank);
    }
    if (tensor.getStorageView().getMemory() == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS %s must not be null", name);
    }
    if (tensor.getStorageView().isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS %s must be device memory", name);
    }
}

bool isRoutingType(const Tensor& tensor) {
    const auto type = tensor.getStorageView().getNumericType();
    return type == FTRAIN_NUMERIC_TYPE_INT32 || type == FTRAIN_NUMERIC_TYPE_INT64;
}

void validateRelationships(const Tensor& input, const Tensor& weight, const Tensor& chosen_experts,
                           const Tensor& expert_weight, const Tensor& output, FTrainMoERSGemmVariant variant) {
    const auto& input_view = input.getStorageView();
    const auto& weight_view = weight.getStorageView();
    const auto& chosen_view = chosen_experts.getStorageView();
    const auto& scale_view = expert_weight.getStorageView();
    const auto& output_view = output.getStorageView();
    const auto input_k = input_view.getDims().at(1);
    const auto output_n = output_view.getDims().at(1);
    const auto weight_first = weight_view.getDims().at(1);
    const auto weight_second = weight_view.getDims().at(2);
    if (variant == FTRAIN_MOERS_GEMM_NK_DIRECT) {
        if (input_k != weight_second || output_n != weight_first) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "MoEReduceRS NK weight must be [E, N, K] and match input/output");
        }
    } else if (variant == FTRAIN_MOERS_GEMM_KN_DIRECT) {
        if (input_k != weight_first || output_n != weight_second) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "MoEReduceRS KN weight must be [E, K, N] and match input/output");
        }
    } else {
        const bool matches_nk = input_k == weight_second && output_n == weight_first;
        const bool matches_kn = input_k == weight_first && output_n == weight_second;
        if (!matches_nk && !matches_kn) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "MoEReduceRS layout-auto weight must match [E, N, K] or [E, K, N]");
        }
    }
    if (chosen_view.getDims().at(0) * chosen_view.getDims().at(1) != input_view.getDims().at(0)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS routing rows do not match input rows");
    }
    if (scale_view.getDims() != chosen_view.getDims()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS scale shape does not match routing shape");
    }
    if (!isRoutingType(chosen_experts)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS chosen_experts must use INT32 or INT64");
    }
}

}  // namespace

MoEReduceRSProblem::MoEReduceRSProblem(Tensor input_value, Tensor weight_value, Tensor chosen_experts_value,
                                       Tensor expert_weight_value, Tensor output_value,
                                       MoEReduceRSAttributes attributes)
    : input_(std::move(input_value)), weight_(std::move(weight_value)), chosen_experts_(std::move(chosen_experts_value)),
      expert_weight_(std::move(expert_weight_value)), output_(std::move(output_value)), attributes_(attributes) {
    validateRank(attributes_.getSessionId(), input_, 2, "input");
    validateRank(attributes_.getSessionId(), weight_, 3, "weight");
    validateRank(attributes_.getSessionId(), chosen_experts_, 2, "chosen_experts");
    validateRank(attributes_.getSessionId(), expert_weight_, 2, "expert_weight");
    validateRank(attributes_.getSessionId(), output_, 2, "output");
    if (attributes_.getRoute() <= FTRAIN_MOERS_ROUTE_INVALID || attributes_.getRoute() >= FTRAIN_MOERS_ROUTE_COUNT ||
        attributes_.getGemmVariant() <= FTRAIN_MOERS_GEMM_INVALID ||
        attributes_.getGemmVariant() >= FTRAIN_MOERS_GEMM_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "MoEReduceRS route or gemm variant is invalid");
    }
    validateRelationships(input_, weight_, chosen_experts_, expert_weight_, output_, attributes_.getGemmVariant());
}

std::vector<std::uint64_t> MoEReduceRSProblem::getProblemKey() const {
    std::vector<std::uint64_t> key{attributes_.getSessionId(), static_cast<std::uint64_t>(OperationKind::kMoEReduceRS),
                                   static_cast<std::uint64_t>(attributes_.getRoute()),
                                   static_cast<std::uint64_t>(attributes_.getGemmVariant())};
    appendTensorKey(key, input_);
    appendTensorKey(key, weight_);
    appendTensorKey(key, chosen_experts_);
    appendTensorKey(key, expert_weight_);
    appendTensorKey(key, output_);
    return key;
}

}  // namespace ftrain
