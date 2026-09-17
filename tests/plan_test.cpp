// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/constraints.hpp"
#include "flash_train/error.hpp"
#include "flash_train/plan.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {
namespace {

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Plan accepted invalid input";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

class RecordingPrimitive final : public PrimitiveBase {
  public:
    explicit RecordingPrimitive(std::uint64_t required_workspace_bytes) noexcept
        : required_workspace_bytes_(required_workspace_bytes) {}

    const char* getName() const noexcept override { return "RecordingPrimitive"; }

    std::unique_ptr<PrimitiveBase> clone() const override {
        return std::make_unique<RecordingPrimitive>(required_workspace_bytes_);
    }

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return required_workspace_bytes_; }

    int executions                     = 0;
    void* last_workspace               = nullptr;
    std::uint64_t last_workspace_bytes = 0;

  protected:
    void executeImpl(const Resources& resources) override {
        ++executions;
        last_workspace       = resources.getWorkspace();
        last_workspace_bytes = resources.getWorkspaceBytes();
    }

  private:
    std::uint64_t required_workspace_bytes_;
};

std::vector<std::unique_ptr<PrimitiveBase>> makePrimitives(std::initializer_list<std::uint64_t> requirements) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives;
    primitives.reserve(requirements.size());
    for (const std::uint64_t required_workspace_bytes : requirements) {
        primitives.push_back(std::make_unique<RecordingPrimitive>(required_workspace_bytes));
    }
    return primitives;
}

TEST(PlanTest, RejectsEmptyAndNullPrimitiveLists) {
    expectInvalidArgument([&] { static_cast<void>(Plan({}, getCurrentDeviceId())); });

    std::vector<std::unique_ptr<PrimitiveBase>> with_null = makePrimitives({8});
    with_null.push_back(nullptr);
    expectInvalidArgument([&] { static_cast<void>(Plan(std::move(with_null), getCurrentDeviceId())); });
}

TEST(PlanTest, ReportsPerPrimitiveWorkspaceRequirements) {
    Plan plan(makePrimitives({16, 48, 32}), getCurrentDeviceId());

    EXPECT_EQ(plan.getNumPrimitives(), 3);
    EXPECT_EQ(plan.getPrimitiveRequiredWorkspaceBytes(0), 16);
    EXPECT_EQ(plan.getPrimitiveRequiredWorkspaceBytes(1), 48);
    EXPECT_EQ(plan.getPrimitiveRequiredWorkspaceBytes(2), 32);
    expectInvalidArgument([&] { static_cast<void>(plan.getPrimitiveRequiredWorkspaceBytes(3)); });
}

TEST(PlanTest, ExecutesOnlyTheIndexedPrimitiveWithTheSharedWorkspace) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = makePrimitives({16, 0});
    const RecordingPrimitive& first                        = static_cast<RecordingPrimitive&>(*primitives[0]);
    const RecordingPrimitive& second                       = static_cast<RecordingPrimitive&>(*primitives[1]);
    Plan plan(std::move(primitives), getCurrentDeviceId());

    std::uint64_t workspace[6]{};
    plan.execute(1, workspace, sizeof(workspace), nullptr);

    EXPECT_EQ(first.executions, 0);
    EXPECT_EQ(second.executions, 1);
    EXPECT_EQ(second.last_workspace, workspace);
    EXPECT_EQ(second.last_workspace_bytes, sizeof(workspace));

    plan.execute(0, workspace, sizeof(workspace), nullptr);
    EXPECT_EQ(first.executions, 1);
    EXPECT_EQ(first.last_workspace, workspace);
    EXPECT_EQ(first.last_workspace_bytes, sizeof(workspace));
    EXPECT_EQ(second.executions, 1);
}

TEST(PlanTest, RejectsExecutionOnAnotherCurrentDeviceWithoutDispatch) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = makePrimitives({0});
    const RecordingPrimitive& primitive                    = static_cast<RecordingPrimitive&>(*primitives[0]);
    Plan plan(std::move(primitives), static_cast<FTrainDeviceId>(getCurrentDeviceId() + 1));

    std::uint64_t workspace[1]{};
    expectInvalidArgument([&] { plan.execute(0, workspace, sizeof(workspace), nullptr); });
    EXPECT_EQ(primitive.executions, 0);
}

TEST(PlanTest, RejectsOutOfRangePrimitiveIndexWithoutDispatch) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = makePrimitives({0});
    const RecordingPrimitive& primitive                    = static_cast<RecordingPrimitive&>(*primitives[0]);
    Plan plan(std::move(primitives), getCurrentDeviceId());

    std::uint64_t workspace[1]{};
    expectInvalidArgument([&] { plan.execute(1, workspace, sizeof(workspace), nullptr); });
    EXPECT_EQ(primitive.executions, 0);
}

TEST(PlanTest, RejectsInsufficientWorkspaceWithoutDispatch) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = makePrimitives({32});
    const RecordingPrimitive& primitive                    = static_cast<RecordingPrimitive&>(*primitives[0]);
    Plan plan(std::move(primitives), getCurrentDeviceId());

    std::uint64_t workspace[4]{};
    expectInvalidArgument([&] { plan.execute(0, workspace, 31, nullptr); });
    expectInvalidArgument([&] { plan.execute(0, nullptr, sizeof(workspace), nullptr); });
    EXPECT_EQ(primitive.executions, 0);
}

TEST(PlanTest, ZeroRequirementPlanExecutesWithoutWorkspace) {
    std::vector<std::unique_ptr<PrimitiveBase>> primitives = makePrimitives({0});
    const RecordingPrimitive& primitive                    = static_cast<RecordingPrimitive&>(*primitives[0]);
    Plan plan(std::move(primitives), getCurrentDeviceId());

    plan.execute(0, nullptr, 0, nullptr);

    EXPECT_EQ(primitive.executions, 1);
    EXPECT_EQ(primitive.last_workspace, nullptr);
}

}  // namespace
}  // namespace ftrain
