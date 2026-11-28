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

constexpr int s = 512, kdim = 64, vdim = 64;

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

template<typename T>
void checkGradient(const std::vector<T>& actual, const std::vector<double>& expected, const char* name) {
    ASSERT_EQ(actual.size(), expected.size());
    double error = 0, magnitude = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double sample = sizeof(T) == sizeof(float) ? static_cast<double>(actual[i]) : fp64(actual[i]);
        error = std::max(error, std::abs(sample - expected[i]));
        magnitude = std::max(magnitude, std::abs(expected[i]));
    }
    EXPECT_LT(error / std::max(magnitude, 1e-30), 0.02) << name;
}

TEST(Gdn2BackwardNumerics, SixGradientsMatchFp64Recurrence) {
    if (ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch() != "gfx936") GTEST_SKIP();
    std::vector<std::uint16_t> q(s * kdim), key(q.size()), value(s * vdim), dout(value.size());
    std::vector<float> gate(q.size(), -0.01F), erase(q.size(), 0.0625F), write(value.size(), 0.5F);
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = bf16((static_cast<int>(i % 11) - 5) * 0.01F);
        key[i] = bf16((static_cast<int>(i % 13) - 6) * 0.01F);
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] = bf16((static_cast<int>(i % 17) - 8) * 0.01F);
        dout[i] = bf16((static_cast<int>(i % 19) - 9) * 0.01F);
    }
    ftrain_test::DeviceBuffer<std::uint16_t> dq_in(q.size()), dk_in(key.size()), dv_in(value.size()),
        ddo(dout.size()), dq(q.size()), dk(key.size()), dv(value.size()), db(erase.size()), dw(write.size());
    ftrain_test::DeviceBuffer<float> dg_in(gate.size()), db_in(erase.size()), dw_in(write.size()), dg(gate.size());
    ftrain_test::DeviceBuffer<std::uint8_t> workspace(16 * 1024 * 1024);
    ASSERT_NE(dq_in.get(), nullptr);
    ASSERT_NE(dk_in.get(), nullptr);
    ASSERT_NE(dv_in.get(), nullptr);
    ASSERT_NE(ddo.get(), nullptr);
    ASSERT_NE(dq.get(), nullptr);
    ASSERT_NE(dk.get(), nullptr);
    ASSERT_NE(dv.get(), nullptr);
    ASSERT_NE(db.get(), nullptr);
    ASSERT_NE(dw.get(), nullptr);
    ASSERT_NE(dg_in.get(), nullptr);
    ASSERT_NE(db_in.get(), nullptr);
    ASSERT_NE(dw_in.get(), nullptr);
    ASSERT_NE(dg.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    dq_in.upload(q, nullptr);
    dk_in.upload(key, nullptr);
    dv_in.upload(value, nullptr);
    ddo.upload(dout, nullptr);
    dg_in.upload(gate, nullptr);
    db_in.upload(erase, nullptr);
    dw_in.upload(write, nullptr);
    const std::array<std::int64_t, 4> qdims{1, s, 1, kdim}, vdims{1, s, 1, vdim};
    auto view = [](void* ptr, const std::int64_t* dims, FTrainNumericType type) {
        return FTrainStorageView{ptr, dims, nullptr, 4, type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    };
    ASSERT_EQ(ftrainGdn2Backward(
                  view(dq_in.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dk_in.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dv_in.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dg_in.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(db_in.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(dw_in.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(ddo.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dq.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dk.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dv.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dg.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_FP32),
                  view(db.get(), qdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  view(dw.get(), vdims.data(), FTRAIN_NUMERIC_TYPE_BF16),
                  0.125F, workspace.get(), workspace.count(), nullptr),
              FTRAIN_STATUS_SUCCESS);
    const auto actual_q = dq.download(nullptr), actual_k = dk.download(nullptr);
    const auto actual_v = dv.download(nullptr);
    const auto actual_g = dg.download(nullptr);
    const auto actual_b = db.download(nullptr), actual_w = dw.download(nullptr);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::size_t kv = kdim * vdim;
    std::vector<double> dec(s * kv), residual(s * vdim), state(kv, 0.0), ds(kv, 0.0);
    for (int t = 0; t < s; ++t) {
        double* d = &dec[t * kv];
        double* r = &residual[t * vdim];
        for (int i = 0; i < kdim; ++i)
            for (int j = 0; j < vdim; ++j)
                d[i * vdim + j] = std::exp(static_cast<double>(gate[t * kdim + i])) * state[i * vdim + j];
        for (int j = 0; j < vdim; ++j) {
            double prediction = 0;
            for (int i = 0; i < kdim; ++i)
                prediction += fp64(key[t * kdim + i]) * erase[t * kdim + i] * d[i * vdim + j];
            r[j] = write[t * vdim + j] * fp64(value[t * vdim + j]) - prediction;
        }
        for (int i = 0; i < kdim; ++i)
            for (int j = 0; j < vdim; ++j)
                state[i * vdim + j] = d[i * vdim + j] + fp64(key[t * kdim + i]) * r[j];
    }
    std::vector<double> ref_q(q.size()), ref_k(key.size()), ref_v(value.size()),
        ref_g(gate.size()), ref_b(erase.size()), ref_w(write.size());
    for (int t = s - 1; t >= 0; --t) {
        const double* d = &dec[t * kv];
        const double* r = &residual[t * vdim];
        std::vector<double> dr(vdim, 0.0);
        for (int i = 0; i < kdim; ++i) {
            double acc = 0;
            for (int j = 0; j < vdim; ++j) {
                const double do_value = fp64(dout[t * vdim + j]);
                ds[i * vdim + j] += 0.125 * fp64(q[t * kdim + i]) * do_value;
                acc += (d[i * vdim + j] + fp64(key[t * kdim + i]) * r[j]) * do_value;
            }
            ref_q[t * kdim + i] = 0.125 * acc;
        }
        for (int i = 0; i < kdim; ++i)
            for (int j = 0; j < vdim; ++j) {
                const double grad = ds[i * vdim + j];
                dr[j] += fp64(key[t * kdim + i]) * grad;
                ref_k[t * kdim + i] += grad * r[j];
            }
        for (int j = 0; j < vdim; ++j) {
            ref_v[t * vdim + j] = write[t * vdim + j] * dr[j];
            ref_w[t * vdim + j] = fp64(value[t * vdim + j]) * dr[j];
        }
        for (int i = 0; i < kdim; ++i) {
            const double decay = std::exp(static_cast<double>(gate[t * kdim + i]));
            const double ki = fp64(key[t * kdim + i]);
            const double kb = ki * erase[t * kdim + i];
            double dkb = 0, dgi = 0;
            for (int j = 0; j < vdim; ++j) {
                const int index = i * vdim + j;
                const double ddec = ds[index] - kb * dr[j];
                dkb -= d[index] * dr[j];
                dgi += ddec * d[index];
                ds[index] = decay * ddec;
            }
            ref_k[t * kdim + i] += erase[t * kdim + i] * dkb;
            ref_b[t * kdim + i] = ki * dkb;
            ref_g[t * kdim + i] = dgi;
        }
    }
    checkGradient(actual_q, ref_q, "dq");
    checkGradient(actual_k, ref_k, "dk");
    checkGradient(actual_v, ref_v, "dv");
    checkGradient(actual_g, ref_g, "dg");
    checkGradient(actual_b, ref_b, "db");
    checkGradient(actual_w, ref_w, "dw");
}

}  // namespace
