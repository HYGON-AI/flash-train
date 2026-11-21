// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// Black-box tests of the KDA convenience C API, written against flash_train.h only: plan-time
// behavior (status codes, workspace query) needs no device memory, so fake aligned addresses
// stand in for the tensors.

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "flash_train/flash_train.h"

namespace {

struct Ports {
    std::int64_t thk[3], th[2], th64[3], h[1], hk[2], cu[1];
    FTrainKdaForwardTensors fwd{};

    Ports(std::int64_t T, std::int64_t H, std::int64_t K, bool gate)
        : thk{T, H, K}, th{T, H}, th64{T, H, 64}, h{H}, hk{H, K}, cu{2} {
        auto view = [](std::uintptr_t slot, const std::int64_t* dims, std::uint8_t n, FTrainNumericType t,
                       bool present = true) {
            return FTrainStorageView{present ? reinterpret_cast<void*>((slot + 1) * 0x10000) : nullptr,
                                     dims,
                                     nullptr,
                                     n,
                                     t,
                                     FTRAIN_INDEX_TYPE_CONTINUOUS,
                                     false};
        };
        const FTrainNumericType bf16 = FTRAIN_NUMERIC_TYPE_BF16, fp32 = FTRAIN_NUMERIC_TYPE_FP32;
        fwd = FTrainKdaForwardTensors{view(0, thk, 3, bf16),         view(1, thk, 3, bf16),
                                      view(2, thk, 3, bf16),         view(3, thk, 3, gate ? bf16 : fp32),
                                      view(4, th, 2, fp32),          view(5, h, 1, fp32, gate),
                                      view(6, hk, 2, fp32, gate),    view(7, cu, 1, FTRAIN_NUMERIC_TYPE_INT32, false),
                                      view(8, thk, 3, bf16),         view(9, th, 2, fp32),
                                      view(10, th, 2, fp32),         view(11, th64, 3, bf16),
                                      view(12, th64, 3, bf16),       view(13, thk, 3, bf16, false),
                                      view(14, thk, 3, bf16, false), view(15, cu, 1, bf16, false),
                                      view(16, cu, 1, bf16, false)};
    }
};

bool onGfx936() {
    int device = 0;
    hipDeviceProp_t prop;
    return hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&prop, device) == hipSuccess &&
           std::strncmp(prop.gcnArchName, "gfx936", 6) == 0;
}

constexpr FTrainKdaAttributes kGate{0.0883883F, -5.0F, 1, 1};

TEST(KdaApiTest, NullArgumentsAreInvalid) {
    std::uint64_t bytes = 0;
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(nullptr, kGate, &bytes), FTRAIN_STATUS_INVALID_ARGUMENT);
    const Ports p(128, 2, 128, true);
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(&p.fwd, kGate, nullptr), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(KdaApiTest, InvalidAttributesAreRejected) {
    const Ports p(128, 2, 128, true);
    std::uint64_t bytes = 0;
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(&p.fwd, FTrainKdaAttributes{0.0F, -5.0F, 1, 1}, &bytes),
              FTRAIN_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(&p.fwd, FTrainKdaAttributes{0.1F, 0.5F, 1, 1}, &bytes),
              FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(KdaApiTest, InconsistentShapesAreInvalid) {
    Ports p(128, 2, 128, true);
    p.th[1]             = 3;  // beta / rstd heads no longer match q
    std::uint64_t bytes = 0;
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(&p.fwd, kGate, &bytes), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST(KdaApiTest, UnsupportedHeadDimIsUnsupported) {
    const Ports p(128, 2, 64, true);
    std::uint64_t bytes = 0;
    EXPECT_EQ(ftrainKdaForwardGetWorkspaceBytes(&p.fwd, kGate, &bytes), FTRAIN_STATUS_UNSUPPORTED);
}

TEST(KdaApiTest, SupportedShapeReportsWorkspaceOnGfx936Only) {
    const Ports p(1000, 4, 128, true);
    std::uint64_t bytes       = 0;
    const FTrainStatus status = ftrainKdaForwardGetWorkspaceBytes(&p.fwd, kGate, &bytes);
    if (onGfx936()) {
        EXPECT_EQ(status, FTRAIN_STATUS_SUCCESS);
        EXPECT_GT(bytes, 0U);
    } else {
        EXPECT_EQ(status, FTRAIN_STATUS_UNSUPPORTED);
    }
}

}  // namespace
