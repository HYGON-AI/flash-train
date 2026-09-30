// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef FTRAIN_TESTS_NUMERICS_HPP_
#define FTRAIN_TESTS_NUMERICS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace ftrain_test {

// RAII device buffer with stream-ordered upload/download. Construction failure
// is reported via ADD_FAILURE and leaves get() == nullptr; tests assert on it.
template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        const hipError_t rc = hipMalloc(&memory_, count_ * sizeof(T));
        if (rc != hipSuccess) {
            ADD_FAILURE() << "hipMalloc failed: " << hipGetErrorString(rc);
            memory_ = nullptr;
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (memory_ != nullptr) { EXPECT_EQ(hipFree(memory_), hipSuccess); }
    }

    T* get() const { return static_cast<T*>(memory_); }

    std::size_t count() const { return count_; }

    void upload(const T* host, hipStream_t stream) {
        EXPECT_EQ(hipMemcpyAsync(memory_, host, count_ * sizeof(T), hipMemcpyHostToDevice, stream),
                  hipSuccess);
    }

    void upload(const std::vector<T>& host, hipStream_t stream) {
        EXPECT_EQ(host.size(), count_);
        upload(host.data(), stream);
    }

    void download(T* host, hipStream_t stream) {
        EXPECT_EQ(hipMemcpyAsync(host, memory_, count_ * sizeof(T), hipMemcpyDeviceToHost, stream),
                  hipSuccess);
    }

    std::vector<T> download(hipStream_t stream) {
        std::vector<T> host(count_);
        download(host.data(), stream);
        return host;
    }

  private:
    void* memory_    = nullptr;
    std::size_t count_;
};

// Compares against a host reference with a mixed absolute/relative tolerance:
// an element passes when |actual - reference| <= abs_tol + rel_tol * |reference|.
// Returns the worst observed ratio diff / allowed; a value above 1.0 has already
// produced a failure carrying the worst absolute/relative deviations and position.
// The tolerance policy (per op and precision) must be documented at call sites.
inline double expectNear(const std::vector<float>& actual, const std::vector<float>& reference,
                         double rel_tol, double abs_tol, const char* what) {
    EXPECT_EQ(actual.size(), reference.size());
    if (actual.size() != reference.size()) { return std::numeric_limits<double>::infinity(); }

    bool ok          = true;
    double worst     = 0.0;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double ref  = static_cast<double>(reference[i]);
        const double diff = std::fabs(static_cast<double>(actual[i]) - ref);
        const double allowed = abs_tol + rel_tol * std::fabs(ref);
        const double ratio   = diff / std::max(allowed, 1e-30);
        if (ratio > worst) {
            worst       = ratio;
            worst_abs   = diff;
            worst_rel   = diff / std::max(std::fabs(ref), 1e-30);
            worst_index = i;
        }
        if (ratio > 1.0) { ok = false; }
    }
    if (!ok) {
        ADD_FAILURE() << what << ": element " << worst_index << " deviates abs=" << worst_abs
                      << " rel=" << worst_rel << " (tolerance rel=" << rel_tol << " abs=" << abs_tol
                      << "), worst ratio " << worst;
    }
    return worst;
}

}  // namespace ftrain_test

#endif  // FTRAIN_TESTS_NUMERICS_HPP_
