#include <cstdint>
#include <memory>

#include "flash_train/trace.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/api.hpp"

extern "C" FTrainStatus ftrainPlanCreate(FTrainPlan* plan, FTrainArgs args, std::uint64_t max_ws_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainPlanCreate"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanCreate: plan output must not be null");
        }
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanCreate: args handle must not be null");
        }

        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine =
            ftrain::getGlobalHandle().findOpsEngine(args->args.getPatternKey());
        if (ops_engine == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "OpsEngine registered for Args is unavailable");
        }
        auto created_handle = std::make_unique<FTrainPlanStruct>(ops_engine->createPlan(args->args, max_ws_bytes));
        *plan               = created_handle.release();
    });
}

extern "C" FTrainStatus ftrainPlanDestroy(FTrainPlan plan) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanDestroy: plan handle must not be null");
        }
        delete plan;
    });
}

extern "C" FTrainStatus ftrainPlanGetNumPrimitives(FTrainPlan plan, std::uint64_t* num_primitives) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanGetNumPrimitives: plan handle must not be null");
        }
        if (num_primitives == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanGetNumPrimitives: num_primitives output must not be null");
        }
        *num_primitives = plan->plan.getNumPrimitives();
    });
}

extern "C" FTrainStatus ftrainPlanGetPrimitiveRequiredWorkspaceBytes(FTrainPlan plan, std::uint64_t primitive_index,
                                                                     std::uint64_t* workspace_bytes) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanGetPrimitiveRequiredWorkspaceBytes: plan handle must not be null");
        }
        if (workspace_bytes == nullptr) {
            throw ftrain::Exception(
                FTRAIN_STATUS_INVALID_ARGUMENT,
                "ftrainPlanGetPrimitiveRequiredWorkspaceBytes: workspace_bytes output must not be null");
        }
        *workspace_bytes = plan->plan.getPrimitiveRequiredWorkspaceBytes(primitive_index);
    });
}

extern "C" FTrainStatus ftrainPlanExecutePrimitive(FTrainPlan plan, std::uint64_t primitive_index, void* workspace,
                                                   std::uint64_t workspace_bytes, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainPlanExecutePrimitive"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanExecutePrimitive: plan handle must not be null");
        }
        plan->plan.execute(primitive_index, workspace, workspace_bytes, stream);
    });
}
