// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

// GDN2 forward numerics on the gfx938 device sources (archive
// 05_gfx938_aicc_port).  Runs only when the runtime device is gfx938.

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

std::uint16_t bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

double fp64(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void runCase(int s, int h, int hv, int kdim, int vdim, const char* label) {
    if (ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch() != "gfx938") GTEST_SKIP();
    std::vector<std::uint16_t> q(s * h * kdim), key(q.size()), value(s * hv * vdim);
    std::vector<float> gate(s * hv * kdim), erase(gate.size()), write(value.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = bf16((static_cast<int>(i % 11) - 5) * 0.01F);
        key[i] = bf16((static_cast<int>(i % 13) - 6) * 0.01F);
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] = bf16((static_cast<int>(i % 17) - 8) * 0.01F);
        write[i] = 0.5F;
    }
    for (std::size_t i = 0; i < gate.size(); ++i) {
        gate[i] = -0.01F;
        erase[i] = 0.06F;
    }
    ftrain_test::DeviceBuffer<std::uint16_t> dq(q.size()), dk(key.size()), dv(value.size()), dout(value.size());
    ftrain_test::DeviceBuffer<float> dg(gate.size()), db(erase.size()), dw(write.size());
    ftrain_test::DeviceBuffer<std::uint8_t> workspace(64 * 1024 * 1024);
    ASSERT_NE(dq.get(), nullptr);
    ASSERT_NE(dk.get(), nullptr);
    ASSERT_NE(dv.get(), nullptr);
    ASSERT_NE(dout.get(), nullptr);
    ASSERT_NE(dg.get(), nullptr);
    ASSERT_NE(db.get(), nullptr);
    ASSERT_NE(dw.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    dq.upload(q, nullptr);
    dk.upload(key, nullptr);
    dv.upload(value, nullptr);
    dg.upload(gate, nullptr);
    db.upload(erase, nullptr);
    dw.upload(write, nullptr);
    const std::array<std::int64_t, 4> qdims{1, s, h, kdim}, vdims{1, s, hv, vdim};
    const std::array<std::int64_t, 4> gdims{1, s, hv, kdim};
    auto view = [](void* ptr, const std::int64_t* dims, FTrainNumericType type) {
        return FTrainStorageView{ptr, dims, nullptr, 4, type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    };
    ASSERT_EQ(ftrainGdn2Forward(
                  view(dq.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dk.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dv.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dg.get(), gdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(db.get(), gdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(dw.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(dout.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  0.125F, workspace.get(), workspace.count(), nullptr),
              FTRAIN_STATUS_SUCCESS) << label;
    const auto output = dout.download(nullptr);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<double> expected(output.size());
    for (int head = 0; head < hv; ++head) {
        std::vector<double> state(kdim * vdim, 0.0);
        for (int t = 0; t < s; ++t) {
            const int qbase = t * kdim, gbase = (t * hv + head) * kdim;
            const int vbase = (t * hv + head) * vdim;
            for (int i = 0; i < kdim; ++i)
                for (int j = 0; j < vdim; ++j)
                    state[i * vdim + j] *= std::exp(static_cast<double>(gate[gbase + i]));
            std::array<double, 256> prediction{};
            for (int i = 0; i < kdim; ++i)
                for (int j = 0; j < vdim; ++j)
                    prediction[j] += fp64(key[qbase + i]) * erase[gbase + i] * state[i * vdim + j];
            for (int i = 0; i < kdim; ++i)
                for (int j = 0; j < vdim; ++j)
                    state[i * vdim + j] += fp64(key[qbase + i]) *
                        (write[vbase + j] * fp64(value[vbase + j]) - prediction[j]);
            for (int j = 0; j < vdim; ++j) {
                double sum = 0;
                for (int i = 0; i < kdim; ++i) sum += fp64(q[qbase + i]) * state[i * vdim + j];
                expected[vbase + j] = 0.125 * sum;
            }
        }
    }
    double error = 0, magnitude = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        error = std::max(error, std::abs(fp64(output[i]) - expected[i]));
        magnitude = std::max(magnitude, std::abs(expected[i]));
    }
    EXPECT_LT(error / std::max(magnitude, 1e-30), 0.08) << label;
}

TEST(Gdn2ForwardNumericsGfx938, GqaMatchesFp64Recurrence) { runCase(512, 1, 2, 64, 64, "native"); }
// The gfx938 route pads any S >= 1 up to a multiple of 64 and K/V to
// {Kp <= 128, Vp <= 256}; zero tokens and channels are exact no-ops.
TEST(Gdn2ForwardNumericsGfx938, PaddedShapeMatchesFp64Recurrence) { runCase(100, 1, 2, 96, 160, "padded"); }

}  // namespace
