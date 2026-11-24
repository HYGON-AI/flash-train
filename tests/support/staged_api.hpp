// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_TESTS_STAGED_API_HPP_
#define FTRAIN_TESTS_STAGED_API_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <flash_train/flash_train.h>

namespace ftrain_test {

// Base fixture for staged C API tests: stream and Plan lifetimes, device
// allocation tracking, view construction, and status assertions. An op
// fixture derives from this, calls StagedApiTest::SetUp() first, then wraps
// its own Plan creation (see api/sparse_attn_bwd_staged_fixture.hpp).
class StagedApiTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_EQ(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream_, nullptr);
    }

    void TearDown() override {
        if (stream_ != nullptr) { EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess); }

        for (FTrainSparseAttnBwdPlan plan : plans_) {
            if (plan != nullptr) { EXPECT_EQ(ftrainSparseAttnBwdDestroy(plan), FTRAIN_STATUS_SUCCESS); }
        }

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

    void trackPlan(FTrainSparseAttnBwdPlan plan) { plans_.push_back(plan); }

    static FTrainStorageView makeView(void* memory, const std::int64_t* dims, std::uint8_t num_dims,
                                      FTrainNumericType numeric_type = FTRAIN_NUMERIC_TYPE_FP32,
                                      FTrainIndexType index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS,
                                      bool is_host_memory = false, const std::int64_t* strides = nullptr) {
        return FTrainStorageView{memory, dims, strides, num_dims, numeric_type, index_type, is_host_memory};
    }

    hipStream_t stream_{nullptr};
    std::vector<FTrainSparseAttnBwdPlan> plans_;
    std::vector<void*> allocations_;
};

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_STAGED_API_HPP_
