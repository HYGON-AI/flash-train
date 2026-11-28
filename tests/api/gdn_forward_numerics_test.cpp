// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <flash_train/flash_train.h>

#include "../support/numerics.hpp"
#include "flash_train/constraints.hpp"

namespace {

template<typename T>
T inputValue(float value) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return value;
    } else {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<T>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    }
}

template<typename T>
double toDouble(T value) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return value;
    } else {
        std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
}

template<typename T>
void runCase(int s, bool with_state, int k = 16, int vdim = 16) {
    if (ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch() != "gfx936") GTEST_SKIP();

    constexpr float scale = 0.25F;
    std::vector<T> q(s * k), key(s * k), value(s * vdim);
    std::vector<float> g(s), beta(s), h0(k * vdim);
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = inputValue<T>((static_cast<int>(i % 7) - 3) * 0.03F);
        key[i] = inputValue<T>((static_cast<int>(i % 11) - 5) * 0.02F);
    }
    for (std::size_t i = 0; i < value.size(); ++i)
        value[i] = inputValue<T>((static_cast<int>(i % 13) - 6) * 0.03F);
    for (int i = 0; i < s; ++i) {
        g[i] = -0.02F;
        beta[i] = 0.2F;
    }
    for (std::size_t i = 0; i < h0.size(); ++i)
        h0[i] = (static_cast<int>(i % 17) - 8) * 0.003F;

    ftrain_test::DeviceBuffer<T> dq(q.size()), dk(key.size()), dv(value.size()), dout(value.size());
    ftrain_test::DeviceBuffer<float> dg(g.size()), dbeta(beta.size()), dh0(h0.size()), dht(h0.size());
    ftrain_test::DeviceBuffer<std::uint8_t> workspace(4 * 1024 * 1024);
    ASSERT_NE(dq.get(), nullptr);
    ASSERT_NE(dk.get(), nullptr);
    ASSERT_NE(dv.get(), nullptr);
    ASSERT_NE(dout.get(), nullptr);
    ASSERT_NE(dg.get(), nullptr);
    ASSERT_NE(dbeta.get(), nullptr);
    ASSERT_NE(dh0.get(), nullptr);
    ASSERT_NE(dht.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    dq.upload(q, nullptr);
    dk.upload(key, nullptr);
    dv.upload(value, nullptr);
    dg.upload(g, nullptr);
    dbeta.upload(beta, nullptr);
    dh0.upload(h0, nullptr);

    const std::array<std::int64_t, 4> qdims{1, s, 1, k}, vdims{1, s, 1, vdim};
    const std::array<std::int64_t, 4> sdims{1, 1, k, vdim}, empty{0, 1, k, vdim};
    const std::array<std::int64_t, 3> gdims{1, s, 1};
    const auto type = sizeof(T) == sizeof(float) ? FTRAIN_NUMERIC_TYPE_FP32 : FTRAIN_NUMERIC_TYPE_BF16;
    auto view = [](void* ptr, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType dtype) {
        return FTrainStorageView{ptr, dims, nullptr, rank, dtype, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    };
    const FTrainStatus status = ftrainGdnForward(
        view(dq.get(), qdims.data(), 4, type), view(dk.get(), qdims.data(), 4, type),
        view(dv.get(), vdims.data(), 4, type), view(dg.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(dbeta.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
        view(with_state ? dh0.get() : nullptr, (with_state ? sdims : empty).data(), 4, FTRAIN_NUMERIC_TYPE_FP32),
        view(dout.get(), vdims.data(), 4, type),
        view(with_state ? dht.get() : nullptr, (with_state ? sdims : empty).data(), 4, FTRAIN_NUMERIC_TYPE_FP32),
        scale, k == 64 ? workspace.get() : nullptr, k == 64 ? workspace.count() : 0, nullptr);
    ASSERT_EQ(status, FTRAIN_STATUS_SUCCESS);
    std::vector<T> actual = dout.download(nullptr);
    std::vector<float> final = with_state ? dht.download(nullptr) : std::vector<float>{};
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<double> state(h0.size(), 0.0), expected(value.size());
    if (with_state) {
        for (std::size_t i = 0; i < state.size(); ++i) state[i] = h0[i];
    }
    for (int t = 0; t < s; ++t) {
        for (double& item : state) item *= std::exp(static_cast<double>(g[t]));
        std::vector<double> prediction(vdim, 0.0);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < vdim; ++j)
                prediction[j] += toDouble(key[t * k + i]) * state[i * vdim + j];
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < vdim; ++j)
                state[i * vdim + j] += beta[t] * toDouble(key[t * k + i]) *
                                       (toDouble(value[t * vdim + j]) - prediction[j]);
        for (int j = 0; j < vdim; ++j) {
            double output = 0;
            for (int i = 0; i < k; ++i) output += toDouble(q[t * k + i]) * state[i * vdim + j];
            expected[t * vdim + j] = scale * output;
        }
    }
    double error = 0, magnitude = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        error = std::max(error, std::abs(toDouble(actual[i]) - expected[i]));
        magnitude = std::max(magnitude, std::abs(expected[i]));
    }
    const double normalized = error / std::max(magnitude, 1e-30);
    EXPECT_LT(normalized, sizeof(T) == sizeof(float) ? 0.015 : 0.08) << "S=" << s;
    if (with_state) {
        double state_error = 0, state_magnitude = 0;
        for (std::size_t i = 0; i < state.size(); ++i) {
            state_error = std::max(state_error, std::abs(static_cast<double>(final[i]) - state[i]));
            state_magnitude = std::max(state_magnitude, std::abs(state[i]));
        }
        EXPECT_LT(state_error / std::max(state_magnitude, 1e-30), 0.015);
    }
}

TEST(GdnForwardNumerics, Fp32WithState) { runCase<float>(32, true); }
TEST(GdnForwardNumerics, Bf16WithState) { runCase<std::uint16_t>(32, true); }
TEST(GdnForwardNumerics, Bf16LongSequenceUsesFusedRoute) { runCase<std::uint16_t>(512, false); }
TEST(GdnForwardNumerics, Bf16FourKernelMatchesFp64Recurrence) {
    runCase<std::uint16_t>(512, false, 64, 64);
}

}  // namespace
