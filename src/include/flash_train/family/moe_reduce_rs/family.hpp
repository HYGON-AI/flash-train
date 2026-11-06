// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_MOE_REDUCE_RS_FAMILY_HPP_
#define FTRAIN_FAMILY_MOE_REDUCE_RS_FAMILY_HPP_

#include <flash_train/family/moe_reduce_rs/finder.hpp>
#include <flash_train/family/moe_reduce_rs/problem.hpp>
#include <flash_train/operation/distributed.hpp>
#include <flash_train/ops_args.hpp>
#include <flash_train/primitive.hpp>

#include <memory>
#include <vector>

namespace ftrain {

struct MoEReduceRSFamily final {
    using Problem = MoEReduceRSProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId input{};
        FTrainTensorId weight{};
        FTrainTensorId chosen_experts{};
        FTrainTensorId expert_weight{};
        FTrainTensorId output{};
        FTrainMoEReduceRSOpId op{};
    };

    static const Roles& makeRoles();
    static Pattern makePattern() { return makeRoles().pattern; }
    static Problem makeProblem(const Args& args);
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};

}  // namespace ftrain

#endif
