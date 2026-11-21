#ifndef FTRAIN_TESTS_MOE_STAGED_FIXTURE_HPP_
#define FTRAIN_TESTS_MOE_STAGED_FIXTURE_HPP_

#include "flash_train/flash_train.h"

#include "../support/moe_session.hpp"

namespace ftrain_test {

class MoeApiTest : public MoeSessionTest {
  protected:
    void TearDown() override {
        destroySources();
        for (auto plan : plans) { EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS); }
    }

    void destroySources() {
        for (auto args : arguments) { EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS); }
        for (auto ops : operations) { EXPECT_EQ(ftrainOpsDestroy(ops), FTRAIN_STATUS_SUCCESS); }
        for (auto pattern : patterns) { EXPECT_EQ(ftrainPatternDestroy(pattern), FTRAIN_STATUS_SUCCESS); }
        arguments.clear();
        operations.clear();
        patterns.clear();
    }

    FTrainPlan makePlan(FTrainAGGroupGemm ag, FTrainMoEReduceRS rs, const std::vector<FTrainMoeTensor>& tensors) {
        FTrainPattern pattern = nullptr;
        EXPECT_EQ(ftrainPatternCreate(&pattern), FTRAIN_STATUS_SUCCESS);
        patterns.push_back(pattern);
        std::vector<FTrainTensorId> ports(tensors.size());
        for (auto iterator = ports.rbegin(); iterator != ports.rend(); ++iterator) {
            EXPECT_EQ(ftrainPatternAddTensor(pattern, &*iterator), FTRAIN_STATUS_SUCCESS);
        }
        FTrainAGGroupGemmOpId ag_op{};
        FTrainMoEReduceRSOpId rs_op{};
        if (ag != nullptr) {
            EXPECT_EQ(ftrainPatternAddAGGroupGemm(pattern, &ag_op, ports[0], ports[1], ports[2], ports[3]),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            EXPECT_EQ(ftrainPatternAddMoEReduceRS(pattern, &rs_op, ports[0], ports[1], ports[2], ports[3], ports[4]),
                      FTRAIN_STATUS_SUCCESS);
        }
        FTrainOps ops = nullptr;
        EXPECT_EQ(ftrainOpsCreate(&ops, pattern), FTRAIN_STATUS_SUCCESS);
        operations.push_back(ops);
        FTrainArgs args = nullptr;
        EXPECT_EQ(ftrainArgsCreate(&args, ops), FTRAIN_STATUS_SUCCESS);
        arguments.push_back(args);
        for (size_t index = 0; index < tensors.size(); ++index) {
            EXPECT_EQ(ftrainArgsSetTensor(args, ports[index], tensors[index].storage), FTRAIN_STATUS_SUCCESS);
        }
        if (ag != nullptr) {
            EXPECT_EQ(ftrainArgsSetAGGroupGemm(args, ag_op, ag, tensors[1].version, tensors[2].version),
                      FTRAIN_STATUS_SUCCESS);
        } else {
            EXPECT_EQ(ftrainArgsSetMoEReduceRS(args, rs_op, rs, tensors[1].version, tensors[2].version),
                      FTRAIN_STATUS_SUCCESS);
        }
        FTrainPlan plan = nullptr;
        EXPECT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
        if (plan != nullptr) { plans.push_back(plan); }
        return plan;
    }

    std::vector<FTrainPattern> patterns;
    std::vector<FTrainOps> operations;
    std::vector<FTrainArgs> arguments;
    std::vector<FTrainPlan> plans;
};

}

#endif
