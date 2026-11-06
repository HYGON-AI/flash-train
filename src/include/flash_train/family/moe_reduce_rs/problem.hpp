// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_MOE_REDUCE_RS_PROBLEM_HPP_
#define FTRAIN_FAMILY_MOE_REDUCE_RS_PROBLEM_HPP_

#include <flash_train/operation/distributed.hpp>
#include <flash_train/tensor.hpp>

#include <cstdint>
#include <vector>

namespace ftrain {

struct MoEReduceRSProblem final {
    MoEReduceRSProblem(Tensor input, Tensor weight, Tensor chosen_experts, Tensor expert_weight, Tensor output,
                       MoEReduceRSAttributes attributes);

    std::vector<std::uint64_t> getProblemKey() const;

    const MoEReduceRSAttributes& attributes() const noexcept { return attributes_; }
    const Tensor& getInput() const noexcept { return input_; }
    const Tensor& getWeight() const noexcept { return weight_; }
    const Tensor& getChosenExperts() const noexcept { return chosen_experts_; }
    const Tensor& getExpertWeight() const noexcept { return expert_weight_; }
    const Tensor& getOutput() const noexcept { return output_; }

  private:
    Tensor input_;
    Tensor weight_;
    Tensor chosen_experts_;
    Tensor expert_weight_;
    Tensor output_;
    MoEReduceRSAttributes attributes_;
};

}  // namespace ftrain

#endif
