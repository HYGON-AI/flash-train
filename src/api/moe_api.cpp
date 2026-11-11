#include "flash_train/flash_train.h"

#include <array>
#include <memory>
#include <utility>

#include "flash_train/api.hpp"
#include "flash_train/family/moe/session.hpp"
#include "flash_train/primitive/hygon/moe/backend.hpp"
#include "flash_train/trace.hpp"

namespace {

template<typename Value>
void requirePointer(Value* value, const char* name) {
    if (value == nullptr) {
        throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s must not be null", name);
    }
}

void requireSuccess(FTrainStatus status) {
    if (status != FTRAIN_STATUS_SUCCESS) { throw ftrain::Exception(ftrain::getLastResult()); }
}

template<ftrain::OperationKind Kind>
struct MoeEntry final {
    static constexpr size_t num_ports = Kind == ftrain::OperationKind::kAGGroupGemm ? 4 : 5;
    FTrainOps ops = nullptr;
    std::array<FTrainTensorId, num_ports> ports{};
    typename ftrain::OperationTraits<Kind>::Id operation{};
};

template<ftrain::OperationKind Kind>
const MoeEntry<Kind>& moeEntry() {
    static const MoeEntry<Kind> entry = [] {
        MoeEntry<Kind> created;
        FTrainPattern pattern_handle = nullptr;
        requireSuccess(ftrainPatternCreate(&pattern_handle));
        std::unique_ptr<FTrainPatternStruct, decltype(&ftrainPatternDestroy)> pattern(pattern_handle,
                                                                                     ftrainPatternDestroy);
        for (auto& port : created.ports) { requireSuccess(ftrainPatternAddTensor(pattern.get(), &port)); }
        {
            requireSuccess(ftrainPatternAddAGGroupGemm(pattern.get(), &created.operation, created.ports[0],
                                                        created.ports[1], created.ports[2], created.ports[3]));
        }
        requireSuccess(ftrainOpsCreate(&created.ops, pattern.get()));
        return created;
    }();
    return entry;
}

template<ftrain::OperationKind Kind, typename Config>
ftrain::MoeAttributes<Kind> makeMoeAttributes(FTrainCollectiveContext context, const Config* config,
                                             uint64_t weight_version, uint64_t routing_version,
                                             FTrainStream stream = nullptr) {
    requirePointer(context, "context");
    requirePointer(context->context.get(), "collective context");
    requirePointer(config, "config");
    auto settings = ftrain::makeMoeConfig(*context->context, *config);
    ftrain::MoeAttributes<Kind> attributes{context->getSession(std::move(settings), stream)};
    attributes.tensors[1].version = weight_version;
    attributes.tensors[2].version = routing_version;
    return attributes;
}

template<ftrain::OperationKind Kind, typename Config>
FTrainStatus executeMoe(FTrainCollectiveContext context, const Config* config, const std::array<FTrainMoeTensor, MoeEntry<Kind>::num_ports>& tensors,
                        FTrainStream stream) {
    return ftrain::invokeApi([&] {
        auto attributes = makeMoeAttributes<Kind>(context, config, tensors[1].version, tensors[2].version, stream);
        const auto& entry = moeEntry<Kind>();
        FTrainArgs args_handle = nullptr;
        requireSuccess(ftrainArgsCreate(&args_handle, entry.ops));
        std::unique_ptr<FTrainArgsStruct, decltype(&ftrainArgsDestroy)> args(args_handle, ftrainArgsDestroy);
        for (size_t index = 0; index < tensors.size(); ++index) {
            requireSuccess(ftrainArgsSetTensor(args.get(), entry.ports[index], tensors[index].storage));
            attributes.tensors[index] = {tensors[index].identity, tensors[index].version};
        }
        args->args.setOperation<Kind>(ftrain::PatternOperationId{entry.operation.opaque}, std::move(attributes));
        FTrainPlan plan_handle = nullptr;
        requireSuccess(ftrainPlanCreate(&plan_handle, args.get()));
        std::unique_ptr<FTrainPlanStruct, decltype(&ftrainPlanDestroy)> plan(plan_handle, ftrainPlanDestroy);
        requireSuccess(ftrainPlanExecutePrimitive(plan.get(), 0, nullptr, 0, stream));
    });
}

}

extern "C" FTrainStatus ftrainMoeIsAvailable(bool* available) {
    return ftrain::invokeApi([&] {
        requirePointer(available, "available output");
        *available = ftrain::isMoeBackendEnabled();
    });
}

extern "C" FTrainStatus ftrainCollectiveContextCreate(FTrainCollectiveContext* context, int32_t rank,
                                                      int32_t world_size, FTrainDeviceId device,
                                                      FTrainByteAllGather allgather, void* user_data) {
    return ftrain::invokeApi([&] {
        requirePointer(context, "context output");
        auto created = std::make_unique<FTrainCollectiveContextStruct>();
        created->context = std::make_shared<ftrain::CollectiveContext>(rank, world_size, device, allgather, user_data);
        *context = created.release();
    });
}

extern "C" FTrainStatus ftrainCollectiveContextDestroy(FTrainCollectiveContext context) {
    return ftrain::invokeApi([&] {
        requirePointer(context, "context");
        context->synchronize();
        delete context;
    });
}

extern "C" FTrainStatus ftrainCollectiveContextSynchronize(FTrainCollectiveContext context) {
    return ftrain::invokeApi([&] {
        requirePointer(context, "context");
        context->synchronize();
    });
}

extern "C" FTrainStatus ftrainAGGroupGemm(FTrainCollectiveContext context, const FTrainAGGroupGemmConfig* config,
                                         FTrainMoeTensor input, FTrainMoeTensor weight, FTrainMoeTensor routing,
                                         FTrainMoeTensor output, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainAGGroupGemm"};
    return executeMoe<ftrain::OperationKind::kAGGroupGemm>(context, config, {input, weight, routing, output}, stream);
}


extern "C" FTrainStatus ftrainPatternAddAGGroupGemm(FTrainPattern pattern, FTrainAGGroupGemmOpId* op,
                                                   FTrainTensorId input, FTrainTensorId weight,
                                                   FTrainTensorId routing, FTrainTensorId output) {
    return ftrain::invokeApi([&] {
        requirePointer(pattern, "pattern");
        requirePointer(op, "op output");
        *op = pattern->builder.addOperation<ftrain::OperationKind::kAGGroupGemm>(input, weight, routing, output);
    });
}

extern "C" FTrainStatus ftrainArgsSetAGGroupGemm(FTrainArgs args, FTrainAGGroupGemmOpId op,
                                                FTrainCollectiveContext context, const FTrainAGGroupGemmConfig* config,
                                                uint64_t weight_version, uint64_t routing_version,
                                                FTrainStream stream) {
    return ftrain::invokeApi([&] {
        requirePointer(args, "args");
        args->args.validateOperation(ftrain::PatternOperationId{op.opaque}, ftrain::OperationKind::kAGGroupGemm);
        args->args.setOperation<ftrain::OperationKind::kAGGroupGemm>(
            ftrain::PatternOperationId{op.opaque},
            makeMoeAttributes<ftrain::OperationKind::kAGGroupGemm>(context, config, weight_version, routing_version, stream));
    });
}
