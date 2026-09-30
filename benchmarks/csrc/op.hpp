// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef FTRAIN_BENCH_OP_HPP_
#define FTRAIN_BENCH_OP_HPP_

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <flash_train/common.h>

namespace ftrain_bench {

// 错误即退出：基准 harness 不做恢复
inline void checkHip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "hip error at %s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

inline void checkFtrain(FTrainStatus status, const char* what) {
    if (status != FTRAIN_STATUS_SUCCESS) {
        std::fprintf(stderr, "ftrain error at %s: status=%u msg=%s\n", what, status,
                     ftrainGetLastMessage());
        std::exit(1);
    }
}

inline void require(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "%s\n", what);
        std::exit(2);
    }
}

// 一次测量的算子会话：持有设备缓冲与端口视图，实现三档口径的 op 侧钩子
struct OpSession {
    virtual ~OpSession() = default;

    // L2 便利层：一次全流程调用（对齐 Python 便利层口径）
    virtual void runConvenience(hipStream_t stream) = 0;

    // L3/L4 分阶段：先装配 Pattern（内部记录角色张量/算子 id），再绑定 Args
    virtual void buildPattern(FTrainPattern pattern) = 0;
    virtual void bindArgs(FTrainArgs args) = 0;
};

// 算子接入点：新增算子实现本接口并注册（参照 ops/gemm.cpp）
struct OpCase {
    virtual ~OpCase() = default;

    virtual const char* name() const = 0;             // 如 "gemm"
    virtual const char* shapeGrammar() const = 0;     // 如 "m,k,n"
    virtual const char* convenienceImpl() const = 0;  // 如 "ftrainGemm"
    // 精度名 → compute type，如 {"fp32", FTRAIN_NUMERIC_TYPE_FP32}
    virtual std::vector<std::pair<std::string, FTrainNumericType>> precisions() const = 0;
    // 由形状与精度构造会话；形状槽位语义见 shapeGrammar，非法形状经 require 报错
    virtual std::unique_ptr<OpSession> create(const std::vector<int64_t>& shape,
                                              FTrainNumericType compute) const = 0;
};

void registerOp(std::unique_ptr<OpCase> op_case);
const OpCase* findOp(const std::string& name);
std::string registeredOpNames();

}  // namespace ftrain_bench

#endif  // FTRAIN_BENCH_OP_HPP_
