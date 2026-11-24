// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
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

// Argument metadata and the scale value are copied here; device memory
// addresses are only recorded.
Plan selectSparseAttnBwd(const FTrainSparseAttnBwdArgs& args) {
    validateComputeType(args.compute_type);
    const SparseAttnBwdProblem problem{
        makeTensor(args.q),
        makeTensor(args.kv),
        makeTensor(args.out),
        makeTensor(args.dout),
        makeTensor(args.lse),
        makeTensor(args.attn_sink),
        makeTensor(args.topk_indices),
        makeTensor(args.topk_length),
        makeTensor(args.dq),
        makeTensor(args.dkv),
        makeTensor(args.dsink),
        SparseAttnBwdAttributes{args.compute_type, args.softmax_scale}
    };
    return sparseAttnBwdEngine().createPlan(problem);
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainSparseAttnBwdCreate(FTrainSparseAttnBwdPlan* plan, const FTrainSparseAttnBwdArgs* args) {
    ftrain::ScopedTraceRange trace_range{"ftrainSparseAttnBwdCreate"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainSparseAttnBwdCreate: plan output must not be null");
        }
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainSparseAttnBwdCreate: args must not be null");
        }
        auto created_plan = std::make_unique<FTrainSparseAttnBwdPlanStruct>(ftrain::selectSparseAttnBwd(*args));
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
