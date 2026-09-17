#include "flash_train/common.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "flash_train/api.hpp"
#include "flash_train/error.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/registry.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/trace.hpp"

namespace ftrain {
namespace {

std::shared_ptr<const OpsEngineBase> findOpsEngine(const PatternKey& pattern_key) {
    std::shared_ptr<const OpsEngineBase> ops_engine = getGlobalHandle().findOpsEngine(pattern_key);
    if (ops_engine == nullptr) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "OpsEngine registered for Ops is unavailable");
    }
    return ops_engine;
}

}  // namespace
}  // namespace ftrain

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

        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine = ftrain::findOpsEngine(ops->ops.getPatternKey());
        const FTrainDeviceId device_id                                = ftrain::getCurrentDeviceId();
        const ftrain::SelectionContext context(device_id, max_ws_bytes);
        std::unique_ptr<ftrain::PrimitiveBase> created_primitive = ops_engine->createPrimitive(args->args, context);
        auto created_handle = std::make_unique<FTrainPlanStruct>(device_id, std::move(created_primitive));
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
        std::uint64_t required_workspace_bytes = 0;
        for (const auto& primitive : plan->primitives) {
            required_workspace_bytes = std::max(required_workspace_bytes, primitive->getRequiredWorkspaceBytes());
        }
        *workspace_bytes = required_workspace_bytes;
    });
}

extern "C" FTrainStatus ftrainPlanExecute(FTrainPlan plan, void* workspace, std::uint64_t workspace_bytes,
                                          FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainPlanExecute"};
    return ftrain::invokeApi([&] {
        if (plan == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPlanExecute: plan handle must not be null");
        }
        const FTrainDeviceId current_device_id = ftrain::getCurrentDeviceId();
        if (current_device_id != plan->device_id) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "Plan was created for device %d but the calling thread uses device %d",
                                    static_cast<int>(plan->device_id), static_cast<int>(current_device_id));
        }
        for (const auto& primitive : plan->primitives) { primitive->execute(workspace, workspace_bytes, stream); }
    });
}
