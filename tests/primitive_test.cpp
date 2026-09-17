#include <cstdint>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {
namespace {

class RecordingPrimitive final : public PrimitiveBase {
  public:
    const char* getName() const noexcept override { return "RecordingPrimitive"; }

    std::unique_ptr<PrimitiveBase> clone() const override { return std::make_unique<RecordingPrimitive>(*this); }

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override { return required_workspace_bytes_; }

    void setRequiredWorkspaceBytes(std::uint64_t required_workspace_bytes) noexcept {
        required_workspace_bytes_ = required_workspace_bytes;
    }

    bool wasExecuted() const noexcept { return was_executed_; }

    const Resources& getResources() const noexcept { return received_resources_; }

  private:
    void executeImpl(const Resources& resources) override {
        was_executed_       = true;
        received_resources_ = resources;
    }

    std::uint64_t required_workspace_bytes_{0};
    bool was_executed_{false};
    Resources received_resources_{nullptr, 0, nullptr};
};

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "Primitive accepted invalid execution state";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

static_assert(std::has_virtual_destructor_v<PrimitiveBase>);
static_assert(!std::is_copy_constructible_v<PrimitiveBase>);
static_assert(!std::is_move_constructible_v<PrimitiveBase>);
static_assert(std::is_same_v<decltype(getCurrentDeviceId()), FTrainDeviceId>);
static_assert(noexcept(std::declval<const PrimitiveBase&>().getRequiredWorkspaceBytes()));

TEST(PrimitiveTest, ExposesTheConfiguredWorkspaceRequirement) {
    RecordingPrimitive primitive;
    primitive.setRequiredWorkspaceBytes(128);

    EXPECT_EQ(primitive.getRequiredWorkspaceBytes(), 128U);
}

TEST(PrimitiveTest, ValidatesAndForwardsExecutionArguments) {
    RecordingPrimitive primitive;
    primitive.setRequiredWorkspaceBytes(64);
    std::uint64_t workspace[16]{};
    const FTrainStream stream = nullptr;

    primitive.execute(Resources{workspace, sizeof(workspace), stream});

    EXPECT_TRUE(primitive.wasExecuted());
    EXPECT_EQ(primitive.getResources().getWorkspace(), workspace);
    EXPECT_EQ(primitive.getResources().getWorkspaceBytes(), sizeof(workspace));
    EXPECT_EQ(primitive.getResources().getStream(), stream);
}

TEST(PrimitiveTest, AllowsNullWorkspaceWhenNoWorkspaceIsRequired) {
    RecordingPrimitive primitive;
    primitive.setRequiredWorkspaceBytes(0);

    primitive.execute(Resources{nullptr, 0, nullptr});

    EXPECT_TRUE(primitive.wasExecuted());
    EXPECT_EQ(primitive.getResources().getWorkspace(), nullptr);
}

TEST(PrimitiveTest, RejectsInsufficientOrMissingRequiredWorkspaceWithoutDispatch) {
    RecordingPrimitive primitive;
    primitive.setRequiredWorkspaceBytes(64);
    std::uint64_t workspace[8]{};

    expectInvalidArgument([&] { primitive.execute(Resources{workspace, 63, nullptr}); });
    expectInvalidArgument([&] { primitive.execute(Resources{nullptr, 64, nullptr}); });

    EXPECT_FALSE(primitive.wasExecuted());
}

}  // namespace
}  // namespace ftrain
