// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef FTRAIN_BENCH_OP_HPP_
#define FTRAIN_BENCH_OP_HPP_

#include <cstdint>
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

// 一次测量里建好的算子 Plan：各算子的句柄类型互不相同，这里把它们接到
// 通用骨架需要的查询与执行接口上
struct PlanSession {
    virtual ~PlanSession() = default;

    virtual uint64_t numPrimitives()                                                                    = 0;
    virtual const char* primitiveName(uint64_t index)                                                   = 0;
    virtual uint64_t requiredWorkspaceBytes(uint64_t index)                                             = 0;
    virtual void execute(uint64_t index, void* workspace, uint64_t workspace_bytes, hipStream_t stream) = 0;
};

// 一次测量的算子会话：持有设备缓冲与端口视图，实现三档口径的 op 侧钩子
struct OpSession {
    virtual ~OpSession() = default;

    // L2 便利层：一次全流程调用（对齐 Python 便利层口径）
    virtual void runConvenience(hipStream_t stream) = 0;

    // L3/L4 分阶段：用会话持有的端口建好 Plan；Plan 不得比会话活得久
    virtual std::unique_ptr<PlanSession> createPlan() = 0;
};

// 算子接入点：新增算子实现本接口并注册（参照 ops/sparse_attn_bwd.cpp）
struct OpCase {
    virtual ~OpCase() = default;

    virtual const char* name() const            = 0;  // 如 "sparse_attn_bwd"
    virtual const char* shapeGrammar() const    = 0;  // 如 "tokens,heads,head_dim"
    virtual const char* convenienceImpl() const = 0;  // 如 "ftrainSparseAttnBwd"
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
