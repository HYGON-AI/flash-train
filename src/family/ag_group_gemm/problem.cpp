// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <flash_train/family/ag_group_gemm/problem.hpp>

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
    if (session_id == 0) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm session id is zero"); }
    if (tensor.getStorageView().getDims().size() != rank) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm %s must have rank %zu", name, rank);
    }
    if (tensor.getStorageView().getMemory() == nullptr) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm %s must not be null", name);
    }
    if (tensor.getStorageView().isHostMemory()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm %s must be device memory", name);
    }
}

bool isRoutingType(const Tensor& tensor) {
    const auto type = tensor.getStorageView().getNumericType();
    return type == FTRAIN_NUMERIC_TYPE_INT32 || type == FTRAIN_NUMERIC_TYPE_INT64;
}

void validateRelationships(const Tensor& input, const Tensor& weight, const Tensor& topk_ids,
                           const Tensor& output) {
    const auto& input_view = input.getStorageView();
    const auto& weight_view = weight.getStorageView();
    const auto& topk_view = topk_ids.getStorageView();
    const auto& output_view = output.getStorageView();
    if (input_view.getDims().at(1) != weight_view.getDims().at(1)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm input K does not match weight K");
    }
    if (output_view.getDims().at(1) != weight_view.getDims().at(2)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm output N does not match weight N");
    }
    if (topk_view.getDims().at(1) == 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm topk is empty");
    }
    if (!isRoutingType(topk_ids)) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm topk_ids must use INT32 or INT64");
    }
    // input/topk_ids are rank-local (or ragged rank-local) views while output is
    // the globally gathered grouped-Gemm result.  Its row count therefore
    // depends on the session's full_m rather than the local input rows; the
    // primitive/session owns that runtime contract.
}

}  // namespace

AGGroupGemmProblem::AGGroupGemmProblem(Tensor input_value, Tensor weight_value, Tensor topk_ids_value,
                                       Tensor output_value, AGGroupGemmAttributes attributes)
    : input_(std::move(input_value)), weight_(std::move(weight_value)), topk_ids_(std::move(topk_ids_value)),
      output_(std::move(output_value)), attributes_(attributes) {
    validateRank(attributes_.getSessionId(), input_, 2, "input");
    validateRank(attributes_.getSessionId(), weight_, 3, "weight");
    validateRank(attributes_.getSessionId(), topk_ids_, 2, "topk_ids");
    validateRank(attributes_.getSessionId(), output_, 2, "output");
    if (attributes_.getRoute() <= FTRAIN_AG_ROUTE_INVALID || attributes_.getRoute() >= FTRAIN_AG_ROUTE_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "AGGroupGemm route is invalid");
    }
    validateRelationships(input_, weight_, topk_ids_, output_);
}

std::vector<std::uint64_t> AGGroupGemmProblem::getProblemKey() const {
    std::vector<std::uint64_t> key{attributes_.getSessionId(), static_cast<std::uint64_t>(OperationKind::kAGGroupGemm),
                                   static_cast<std::uint64_t>(attributes_.getRoute())};
    appendTensorKey(key, input_);
    appendTensorKey(key, weight_);
    appendTensorKey(key, topk_ids_);
    appendTensorKey(key, output_);
    return key;
}

}  // namespace ftrain
