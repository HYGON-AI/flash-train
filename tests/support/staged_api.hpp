// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef FTRAIN_TESTS_STAGED_API_HPP_
#define FTRAIN_TESTS_STAGED_API_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <flash_train/common.h>

namespace ftrain_test {

// Base fixture for staged C API tests: stream and handle lifetimes, device
// allocation tracking, view construction, and status assertions. An op
// fixture derives from this, calls StagedApiTest::SetUp() first, then
// assembles its own Pattern/Ops and role bindings (see
// api/gemm_staged_fixture.hpp for the reference shape).
class StagedApiTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_EQ(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream_, nullptr);
    }

    void TearDown() override {
        if (stream_ != nullptr) { EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess); }

        for (FTrainPlan plan : plans_) {
            if (plan != nullptr) { EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS); }
        }
        for (FTrainArgs args : args_) {
            if (args != nullptr) { EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS); }
        }
        if (ops_ != nullptr) { EXPECT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS); }
        if (pattern_ != nullptr) { EXPECT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS); }

        for (void* allocation : allocations_) { EXPECT_EQ(hipFree(allocation), hipSuccess); }
        if (stream_ != nullptr) { EXPECT_EQ(hipStreamDestroy(stream_), hipSuccess); }
    }

    template <typename T>
    T* allocate(std::size_t count) {
        void* allocation    = nullptr;
        const hipError_t rc = hipMalloc(&allocation, count * sizeof(T));
        if (rc != hipSuccess) {
            ADD_FAILURE() << "hipMalloc failed: " << hipGetErrorString(rc);
            return nullptr;
        }
        allocations_.push_back(allocation);
        return static_cast<T*>(allocation);
    }

    FTrainArgs createArgs() {
        FTrainArgs args       = nullptr;
        const FTrainStatus rc = ftrainArgsCreate(&args, ops_);
        if (rc != FTRAIN_STATUS_SUCCESS) {
            ADD_FAILURE() << "ftrainArgsCreate failed with status " << static_cast<unsigned int>(rc);
            return nullptr;
        }
        args_.push_back(args);
        return args;
    }

    void trackPlan(FTrainPlan plan) { plans_.push_back(plan); }

    void forgetArgs(FTrainArgs args) {
        for (FTrainArgs& tracked : args_) {
            if (tracked == args) {
                tracked = nullptr;
                return;
            }
        }
    }

    static FTrainStorageView makeView(void* memory, const std::int64_t* dims, std::uint8_t num_dims,
                                      FTrainNumericType numeric_type = FTRAIN_NUMERIC_TYPE_FP32,
                                      FTrainIndexType index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS,
                                      bool is_host_memory = false, const std::int64_t* strides = nullptr) {
        return FTrainStorageView{memory, dims, strides, num_dims, numeric_type, index_type, is_host_memory};
    }

    void expectPlanStatus(FTrainArgs args, FTrainStatus expected_status) {
        FTrainPlan plan = nullptr;
        EXPECT_EQ(ftrainPlanCreate(&plan, args), expected_status);
        if (plan != nullptr) { trackPlan(plan); }
    }

    FTrainPattern pattern_{nullptr};
    FTrainOps ops_{nullptr};
    hipStream_t stream_{nullptr};
    std::vector<FTrainArgs> args_;
    std::vector<FTrainPlan> plans_;
    std::vector<void*> allocations_;
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_STAGED_API_HPP_
