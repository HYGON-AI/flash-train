// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_AG_GROUP_GEMM_FAMILY_HPP_
#define FTRAIN_FAMILY_AG_GROUP_GEMM_FAMILY_HPP_

#include <flash_train/family/ag_group_gemm/finder.hpp>
#include <flash_train/family/ag_group_gemm/problem.hpp>
#include <flash_train/operation/distributed.hpp>
#include <flash_train/ops_args.hpp>
#include <flash_train/primitive.hpp>

#include <memory>
#include <vector>

namespace ftrain {

struct AGGroupGemmFamily final {
    using Problem = AGGroupGemmProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId input{};
        FTrainTensorId weight{};
        FTrainTensorId topk_ids{};
        FTrainTensorId output{};
        FTrainAGGroupGemmOpId op{};
    };

    static const Roles& makeRoles();
    static Pattern makePattern() { return makeRoles().pattern; }
    static Problem makeProblem(const Args& args);
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};

}  // namespace ftrain

#endif
