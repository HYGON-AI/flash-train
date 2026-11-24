// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_SPARSE_ATTN_BWD_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_SPARSE_ATTN_BWD_STAGED_FIXTURE_HPP_

#include "../support/staged_api.hpp"

#include <flash_train/common.h>

namespace ftrain_test {

class SparseAttnBwdStagedTest : public StagedApiTest {
  protected:
    // Creates a Plan through the staged C API and registers it for teardown;
    // returns nullptr after recording the failure status.
    FTrainSparseAttnBwdPlan createPlan(const FTrainSparseAttnBwdArgs& args) {
        FTrainSparseAttnBwdPlan plan = nullptr;
        const FTrainStatus rc        = ftrainSparseAttnBwdCreate(&plan, &args);
        if (rc != FTRAIN_STATUS_SUCCESS) {
            ADD_FAILURE() << "ftrainSparseAttnBwdCreate failed with status " << static_cast<unsigned int>(rc) << ": "
                          << ftrainGetLastMessage();
            return nullptr;
        }
        trackPlan(plan);
        return plan;
    }
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_SPARSE_ATTN_BWD_STAGED_FIXTURE_HPP_
