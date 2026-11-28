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

void runCase(int hv, bool with_state) {
    if (ftrain::Constraints{ftrain::getCurrentDeviceId()}.getArch() != "gfx936") GTEST_SKIP();
    std::vector<std::uint16_t> q(s * kdim), key(q.size()), value(s * hv * vdim), dout(value.size());
    std::vector<float> gate(s * hv, -0.01F), beta(gate.size(), 0.25F), h0(hv * kdim * vdim),
        dht(h0.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = bf16((static_cast<int>(i % 11) - 5) * 0.01F);
        key[i] = bf16((static_cast<int>(i % 13) - 6) * 0.01F);
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        value[i] = bf16((static_cast<int>(i % 17) - 8) * 0.01F);
        dout[i] = bf16((static_cast<int>(i % 19) - 9) * 0.01F);
    }
    for (std::size_t i = 0; i < h0.size(); ++i) {
        h0[i] = (static_cast<int>(i % 7) - 3) * 0.001F;
        dht[i] = (static_cast<int>(i % 9) - 4) * 0.001F;
    }
    ftrain_test::DeviceBuffer<std::uint16_t> dq_in(q.size()), dk_in(key.size()), dv_in(value.size()),
        ddo(dout.size()), dq(q.size()), dk(key.size()), dv(value.size());
    ftrain_test::DeviceBuffer<float> dg_in(gate.size()), db_in(beta.size()), dh0_in(h0.size()),
        ddht(dht.size()), dg(gate.size()), dbeta(beta.size()), dh0(h0.size());
    ftrain_test::DeviceBuffer<std::uint8_t> workspace(16 * 1024 * 1024);
    ASSERT_NE(dq_in.get(), nullptr);
    ASSERT_NE(dk_in.get(), nullptr);
    ASSERT_NE(dv_in.get(), nullptr);
    ASSERT_NE(ddo.get(), nullptr);
    ASSERT_NE(dq.get(), nullptr);
    ASSERT_NE(dk.get(), nullptr);
    ASSERT_NE(dv.get(), nullptr);
    ASSERT_NE(dg_in.get(), nullptr);
    ASSERT_NE(db_in.get(), nullptr);
    ASSERT_NE(dh0_in.get(), nullptr);
    ASSERT_NE(ddht.get(), nullptr);
    ASSERT_NE(dg.get(), nullptr);
    ASSERT_NE(dbeta.get(), nullptr);
    ASSERT_NE(dh0.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    dq_in.upload(q, nullptr);
    dk_in.upload(key, nullptr);
    dv_in.upload(value, nullptr);
    ddo.upload(dout, nullptr);
    dg_in.upload(gate, nullptr);
    db_in.upload(beta, nullptr);
    dh0_in.upload(h0, nullptr);
    ddht.upload(dht, nullptr);
    const std::array<std::int64_t, 4> qdims{1, s, 1, kdim}, vdims{1, s, hv, vdim};
    const std::array<std::int64_t, 4> state_dims{1, hv, kdim, vdim}, empty{0, hv, kdim, vdim};
    const std::array<std::int64_t, 3> gdims{1, s, hv};
    auto view = [](void* ptr, const std::int64_t* dims, std::uint8_t rank, FTrainNumericType type) {
        return FTrainStorageView{ptr, dims, nullptr, rank, type, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    };
    const auto* sdims = with_state ? state_dims.data() : empty.data();
    ASSERT_EQ(ftrainGdnBackward(
                  view(dq_in.get(), qdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dk_in.get(), qdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dv_in.get(), vdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dg_in.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
                  view(db_in.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
                  view(ddo.get(), vdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(with_state ? dh0_in.get() : nullptr, sdims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                  view(with_state ? ddht.get() : nullptr, sdims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                  view(dq.get(), qdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dk.get(), qdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dv.get(), vdims.data(), 4, FTRAIN_NUMERIC_TYPE_BF16),
                  view(dg.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
                  view(dbeta.get(), gdims.data(), 3, FTRAIN_NUMERIC_TYPE_FP32),
                  view(with_state ? dh0.get() : nullptr, sdims, 4, FTRAIN_NUMERIC_TYPE_FP32),
                  0.125F, workspace.get(), workspace.count(), nullptr),
              FTRAIN_STATUS_SUCCESS);
    const auto actual_q = dq.download(nullptr), actual_k = dk.download(nullptr), actual_v = dv.download(nullptr);
    const auto actual_g = dg.download(nullptr), actual_beta = dbeta.download(nullptr);
    const auto actual_h0 = with_state ? dh0.download(nullptr) : std::vector<float>{};
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<double> ref_q(q.size(), 0.0), ref_k(key.size(), 0.0), ref_v(value.size()),
        ref_g(gate.size()), ref_beta(beta.size()), ref_h0(h0.size());
    const std::size_t kv = kdim * vdim;
    for (int head = 0; head < hv; ++head) {
        std::vector<double> p(s * kv), residual(s * vdim), state(kv, 0.0), ds(kv, 0.0);
        if (with_state) {
            for (std::size_t i = 0; i < kv; ++i) state[i] = h0[head * kv + i];
            for (std::size_t i = 0; i < kv; ++i) ds[i] = dht[head * kv + i];
        }
        for (int t = 0; t < s; ++t) {
            double* pt = &p[t * kv];
            double* e = &residual[t * vdim];
            const double decay = std::exp(static_cast<double>(gate[t * hv + head]));
            for (std::size_t i = 0; i < kv; ++i) pt[i] = state[i] * decay;
            for (int j = 0; j < vdim; ++j) {
                double prediction = 0;
                for (int i = 0; i < kdim; ++i) prediction += fp64(key[t * kdim + i]) * pt[i * vdim + j];
                e[j] = fp64(value[(t * hv + head) * vdim + j]) - prediction;
            }
            for (int i = 0; i < kdim; ++i)
                for (int j = 0; j < vdim; ++j)
                    state[i * vdim + j] = pt[i * vdim + j] +
                        beta[t * hv + head] * fp64(key[t * kdim + i]) * e[j];
        }
        for (int t = s - 1; t >= 0; --t) {
            const double* pt = &p[t * kv];
            const double* e = &residual[t * vdim];
            const int vbase = (t * hv + head) * vdim;
            for (int i = 0; i < kdim; ++i) {
                double acc = 0;
                for (int j = 0; j < vdim; ++j) {
                    const double do_value = fp64(dout[vbase + j]);
                    acc += (pt[i * vdim + j] +
                            beta[t * hv + head] * fp64(key[t * kdim + i]) * e[j]) * do_value;
                    ds[i * vdim + j] += 0.125 * fp64(q[t * kdim + i]) * do_value;
                }
                ref_q[t * kdim + i] += 0.125 * acc;
            }
            std::vector<double> kd(vdim, 0.0), de(vdim);
            for (int i = 0; i < kdim; ++i)
                for (int j = 0; j < vdim; ++j)
                    kd[j] += fp64(key[t * kdim + i]) * ds[i * vdim + j];
            double db = 0;
            for (int j = 0; j < vdim; ++j) {
                de[j] = beta[t * hv + head] * kd[j];
                db += kd[j] * e[j];
                ref_v[vbase + j] = de[j];
            }
            ref_beta[t * hv + head] = db;
            double dgt = 0;
            const double decay = std::exp(static_cast<double>(gate[t * hv + head]));
            for (int i = 0; i < kdim; ++i) {
                const double ki = fp64(key[t * kdim + i]);
                double dki = 0;
                for (int j = 0; j < vdim; ++j) {
                    const int index = i * vdim + j;
                    dki += beta[t * hv + head] * ds[index] * e[j] - pt[index] * de[j];
                    const double dp = ds[index] - ki * de[j];
                    dgt += dp * pt[index];
                    ds[index] = decay * dp;
                }
                ref_k[t * kdim + i] += dki;
            }
            ref_g[t * hv + head] = dgt;
        }
        if (with_state) {
            for (std::size_t i = 0; i < kv; ++i) ref_h0[head * kv + i] = ds[i];
        }
    }
    checkGradient(actual_q, ref_q, "dq");
    checkGradient(actual_k, ref_k, "dk");
    checkGradient(actual_v, ref_v, "dv");
    checkGradient(actual_g, ref_g, "dg");
    checkGradient(actual_beta, ref_beta, "dbeta");
    if (with_state) checkGradient(actual_h0, ref_h0, "dh0 K-first");
}

TEST(GdnBackwardNumerics, StateGradientsKFirst) { runCase(1, true); }
TEST(GdnBackwardNumerics, GqaWithoutState) { runCase(2, false); }

}  // namespace
