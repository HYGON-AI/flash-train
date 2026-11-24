// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "flash_train/flash_train.h"
#include "flash_train/api.hpp"
#include "flash_train/trace.hpp"
#include "flash_train/family/sparse_attn_bwd/family.hpp"

namespace ftrain {
namespace {

void validateComputeType(FTrainNumericType compute_type) {
    if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "compute_type %u is invalid",
                        static_cast<unsigned int>(compute_type));
    }
}

// Selects the Primitives for one call and binds them to the described
// storage. Every port's metadata and the host scale value are copied here;
// the device memory addresses are only recorded.
Plan selectSparseAttnBwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView out, FTrainStorageView dout,
                         FTrainStorageView lse, FTrainStorageView attn_sink, FTrainStorageView topk_indices,
                         FTrainStorageView topk_length, FTrainStorageView dq, FTrainStorageView dkv,
                         FTrainStorageView dsink, FTrainStorageView softmax_scale, FTrainNumericType compute_type) {
    validateComputeType(compute_type);
    const SparseAttnBwdProblem problem{makeTensor(q),
                                       makeTensor(kv),
                                       makeTensor(out),
                                       makeTensor(dout),
                                       makeTensor(lse),
                                       makeTensor(attn_sink),
                                       makeTensor(topk_indices),
                                       makeTensor(topk_length),
                                       makeTensor(dq),
                                       makeTensor(dkv),
                                       makeTensor(dsink),
                                       makeTensor(softmax_scale),
                                       SparseAttnBwdAttributes{compute_type}};
    return sparseAttnBwdEngine().createPlan(problem);
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainSparseAttnBwdCreate(FTrainSparseAttnBwdPlan* plan, FTrainStorageView q,
                                                  FTrainStorageView kv, FTrainStorageView out, FTrainStorageView dout,
                                                  FTrainStorageView lse, FTrainStorageView attn_sink,
                                                  FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                                  FTrainStorageView dq, FTrainStorageView dkv, FTrainStorageView dsink,
                                                  FTrainStorageView softmax_scale, FTrainNumericType compute_type) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnBwdCreate"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdCreate: plan output must not be null");
        }
        auto created_plan = std::make_unique<FTrainSparseAttnBwdPlanStruct>(ftrain::selectSparseAttnBwd(
            q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq, dkv, dsink, softmax_scale, compute_type));
        *plan             = created_plan.release();
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwdDestroy(FTrainSparseAttnBwdPlan plan) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdDestroy: plan handle must not be null");
        }
        delete plan;
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwdGetNumPrimitives(FTrainSparseAttnBwdPlan plan,
                                                            std::uint64_t* num_primitives) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdGetNumPrimitives: plan handle must not be null");
        }
        if (num_primitives == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdGetNumPrimitives: num_primitives output must not be null");
        }
        *num_primitives = plan->plan.getNumPrimitives();
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwdGetPrimitiveName(FTrainSparseAttnBwdPlan plan, std::uint64_t primitive_index,
                                                            const char** name) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdGetPrimitiveName: plan handle must not be null");
        }
        if (name == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdGetPrimitiveName: name output must not be null");
        }
        *name = plan->plan.getPrimitiveName(primitive_index);
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes(FTrainSparseAttnBwdPlan plan,
                                                                              std::uint64_t primitive_index,
                                                                              std::uint64_t* workspace_bytes) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(
                FTRAIN_STATUS_INVALID_ARGUMENT,
                "ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes: plan handle must not be null");
        }
        if (workspace_bytes == nullptr) {
            throw ftrain::Exception(
                FTRAIN_STATUS_INVALID_ARGUMENT,
                "ftrainSparseAttnBwdGetPrimitiveRequiredWorkspaceBytes: workspace_bytes output must not be null");
        }
        *workspace_bytes = plan->plan.getPrimitiveRequiredWorkspaceBytes(primitive_index);
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwdExecute(FTrainSparseAttnBwdPlan plan, std::uint64_t primitive_index,
                                                   void* workspace, std::uint64_t workspace_bytes,
                                                   FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnBwdExecute"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdExecute: plan handle must not be null");
        }
        plan->plan.execute(primitive_index, workspace, workspace_bytes, stream);
    });
}

extern "C" FTrainStatus ftrainSparseAttnBwd(FTrainStorageView q, FTrainStorageView kv, FTrainStorageView out,
                                            FTrainStorageView dout, FTrainStorageView lse, FTrainStorageView attn_sink,
                                            FTrainStorageView topk_indices, FTrainStorageView topk_length,
                                            FTrainStorageView dq, FTrainStorageView dkv, FTrainStorageView dsink,
                                            FTrainStorageView softmax_scale, FTrainNumericType compute_type,
                                            void* workspace, uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnBwd"};
    return ftrain::invokeApi([&] {
        ftrain::Plan plan = ftrain::selectSparseAttnBwd(q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq,
                                                        dkv, dsink, softmax_scale, compute_type);
        plan.execute(0, workspace, workspace_bytes, stream);
    });
}
