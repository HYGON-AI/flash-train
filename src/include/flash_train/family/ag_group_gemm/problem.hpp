// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_AG_GROUP_GEMM_PROBLEM_HPP_
#define FTRAIN_FAMILY_AG_GROUP_GEMM_PROBLEM_HPP_

#include <flash_train/operation/distributed.hpp>
#include <flash_train/tensor.hpp>

#include <cstdint>
#include <vector>

namespace ftrain {

struct AGGroupGemmProblem final {
    AGGroupGemmProblem(Tensor input, Tensor weight, Tensor topk_ids, Tensor output,
                       AGGroupGemmAttributes attributes);

    std::vector<std::uint64_t> getProblemKey() const;

    const AGGroupGemmAttributes& attributes() const noexcept { return attributes_; }
    const Tensor& getInput() const noexcept { return input_; }
    const Tensor& getWeight() const noexcept { return weight_; }
    const Tensor& getTopkIds() const noexcept { return topk_ids_; }
    const Tensor& getOutput() const noexcept { return output_; }

  private:
    Tensor input_;
    Tensor weight_;
    Tensor topk_ids_;
    Tensor output_;
    AGGroupGemmAttributes attributes_;
};

}  // namespace ftrain

#endif
