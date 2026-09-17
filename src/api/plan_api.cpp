#include "flash_train/common.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "flash_train/api.hpp"
#include "flash_train/error.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/primitive/base.hpp"
#include "flash_train/trace.hpp"

extern "C" FTrainStatus ftrainPlanCreate(FTrainPlan* plan, FTrainOps ops, FTrainArgs args, std::uint64_t max_ws_bytes) {
    ftrain::ScopedTraceRange trace_range{"ftrainPlanCreate"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanCreate: plan output must not be null");
        }
        if (ops == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanCreate: ops handle must not be null");
        }
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanCreate: args handle must not be null");
        }
        if (ops->ops.getPatternKey() != args->args.getPatternKey()) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "args was not created for ops");
        }

        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine =
            ftrain::getGlobalHandle().findOpsEngine(ops->ops.getPatternKey());
        if (ops_engine == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INTERNAL_ERROR, "OpsEngine registered for Ops is unavailable");
        }
        const FTrainDeviceId device_id = ftrain::getCurrentDeviceId();
        const ftrain::SelectionContext context(device_id, max_ws_bytes);
        std::unique_ptr<ftrain::PrimitiveBase> created_primitive = ops_engine->createPrimitive(args->args, context);
        std::vector<std::unique_ptr<ftrain::PrimitiveBase>> created_primitives;
        created_primitives.push_back(std::move(created_primitive));
        auto created_handle =
            std::make_unique<FTrainPlanStruct>(ftrain::Plan{device_id, std::move(created_primitives)});
        *plan = created_handle.release();
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

extern "C" FTrainStatus ftrainPlanGetRequiredWs(FTrainPlan plan, std::uint64_t* workspace_bytes) {
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanGetRequiredWs: plan handle must not be null");
        }
        if (workspace_bytes == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPlanGetRequiredWs: workspace_bytes output must not be null");
        }
        *workspace_bytes = plan->plan.getRequiredWorkspaceBytes();
    });
}

extern "C" FTrainStatus ftrainPlanExecute(FTrainPlan plan, void* workspace, std::uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainPlanExecute"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanExecute: plan handle must not be null");
        }
        plan->plan.execute(workspace, workspace_bytes, stream);
    });
}
