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
        if constexpr (Kind == ftrain::OperationKind::kAGGroupGemm) {
            requireSuccess(ftrainPatternAddAGGroupGemm(pattern.get(), &created.operation, created.ports[0],
                                                        created.ports[1], created.ports[2], created.ports[3]));
        } else {
            requireSuccess(ftrainPatternAddMoEReduceRS(pattern.get(), &created.operation, created.ports[0],
                                                        created.ports[1], created.ports[2], created.ports[3],
                                                        created.ports[4]));
        }
        requireSuccess(ftrainOpsCreate(&created.ops, pattern.get()));
        return created;
    }();
    return entry;
}

template<ftrain::OperationKind Kind, typename Handle>
ftrain::MoeAttributes<Kind> makeMoeAttributes(Handle operation, uint64_t weight_version, uint64_t routing_version) {
    requirePointer(operation, "operation");
    requirePointer(operation->session.get(), "operation session");
    ftrain::MoeAttributes<Kind> attributes{operation->session};
    attributes.tensors[1].version = weight_version;
    attributes.tensors[2].version = routing_version;
    return attributes;
}

template<ftrain::OperationKind Kind, typename Handle>
FTrainStatus executeMoe(Handle operation, const std::array<FTrainMoeTensor, MoeEntry<Kind>::num_ports>& tensors,
                        FTrainStream stream) {
    return ftrain::invokeApi([&] {
        auto attributes = makeMoeAttributes<Kind>(operation, tensors[1].version, tensors[2].version);
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

template<typename Handle, typename Config>
void createSession(Handle* operation, FTrainCollectiveContext context, const Config* config, FTrainStream stream) {
    requirePointer(operation, "operation output");
    requirePointer(context, "context");
    requirePointer(config, "config");
    auto settings = ftrain::makeMoeConfig(*context->context, *config);
    auto backend = ftrain::makeMoeBackend(context->context, settings, stream);
    auto session = std::make_shared<ftrain::MoeSession>(context->context, std::move(settings), std::move(backend));
    auto handle = std::make_unique<std::remove_pointer_t<Handle>>();
    handle->session = std::move(session);
    *operation = handle.release();
}

template<typename Handle>
void destroySession(Handle operation) {
    requirePointer(operation, "operation");
    operation->session->synchronize();
    delete operation;
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
        delete context;
    });
}

extern "C" FTrainStatus ftrainAGGroupGemmConfigInit(FTrainAGGroupGemmConfig* config) {
    return ftrain::invokeApi([&] {
        requirePointer(config, "config");
        *config = {};
        config->topk = 1;
        config->route = "ag_nn_ring_chunk";
    });
}

extern "C" FTrainStatus ftrainAGGroupGemmCreate(FTrainAGGroupGemm* operation, FTrainCollectiveContext context,
                                               const FTrainAGGroupGemmConfig* config, FTrainStream stream) {
    return ftrain::invokeApi([&] { createSession(operation, context, config, stream); });
}

extern "C" FTrainStatus ftrainAGGroupGemmExecute(FTrainAGGroupGemm operation, FTrainMoeTensor input,
                                                FTrainMoeTensor weight, FTrainMoeTensor routing,
                                                FTrainMoeTensor output, FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainAGGroupGemmExecute"};
    return executeMoe<ftrain::OperationKind::kAGGroupGemm>(operation, {input, weight, routing, output}, stream);
}

extern "C" FTrainStatus ftrainAGGroupGemmSetRankMSizes(FTrainAGGroupGemm operation, int32_t m_dim,
                                                      const int64_t* sizes, uint32_t count) {
    return ftrain::invokeApi([&] {
        requirePointer(operation, "operation");
        operation->session->setRankMSizes(m_dim, sizes, count);
    });
}

extern "C" FTrainStatus ftrainAGGroupGemmSynchronize(FTrainAGGroupGemm operation) {
    return ftrain::invokeApi([&] {
        requirePointer(operation, "operation");
        operation->session->synchronize();
    });
}

extern "C" FTrainStatus ftrainAGGroupGemmDestroy(FTrainAGGroupGemm operation) {
    return ftrain::invokeApi([&] { destroySession(operation); });
}

extern "C" FTrainStatus ftrainMoEReduceRSConfigInit(FTrainMoEReduceRSConfig* config) {
    return ftrain::invokeApi([&] {
        requirePointer(config, "config");
        *config = {};
        config->topk = 1;
        config->route = "rs_nn";
    });
}

extern "C" FTrainStatus ftrainMoEReduceRSCreate(FTrainMoEReduceRS* operation, FTrainCollectiveContext context,
                                               const FTrainMoEReduceRSConfig* config, FTrainStream stream) {
    return ftrain::invokeApi([&] { createSession(operation, context, config, stream); });
}

extern "C" FTrainStatus ftrainMoEReduceRSExecute(FTrainMoEReduceRS operation, FTrainMoeTensor input,
                                                FTrainMoeTensor weight, FTrainMoeTensor routing,
                                                FTrainMoeTensor expert_weight, FTrainMoeTensor output,
                                                FTrainStream stream) {
    ftrain::ScopedTraceRange trace_range{"ftrainMoEReduceRSExecute"};
    return executeMoe<ftrain::OperationKind::kMoEReduceRS>(operation,
                                                          {input, weight, routing, expert_weight, output}, stream);
}

extern "C" FTrainStatus ftrainMoEReduceRSSetRankMSizes(FTrainMoEReduceRS operation, int32_t m_dim,
                                                      const int64_t* sizes, uint32_t count) {
    return ftrain::invokeApi([&] {
        requirePointer(operation, "operation");
        operation->session->setRankMSizes(m_dim, sizes, count);
    });
}

extern "C" FTrainStatus ftrainMoEReduceRSSynchronize(FTrainMoEReduceRS operation) {
    return ftrain::invokeApi([&] {
        requirePointer(operation, "operation");
        operation->session->synchronize();
    });
}

extern "C" FTrainStatus ftrainMoEReduceRSDestroy(FTrainMoEReduceRS operation) {
    return ftrain::invokeApi([&] { destroySession(operation); });
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
                                                FTrainAGGroupGemm operation, uint64_t weight_version,
                                                uint64_t routing_version) {
    return ftrain::invokeApi([&] {
        requirePointer(args, "args");
        args->args.setOperation<ftrain::OperationKind::kAGGroupGemm>(
            ftrain::PatternOperationId{op.opaque},
            makeMoeAttributes<ftrain::OperationKind::kAGGroupGemm>(operation, weight_version, routing_version));
    });
}

extern "C" FTrainStatus ftrainPatternAddMoEReduceRS(FTrainPattern pattern, FTrainMoEReduceRSOpId* op,
                                                   FTrainTensorId input, FTrainTensorId weight,
                                                   FTrainTensorId routing, FTrainTensorId expert_weight,
                                                   FTrainTensorId output) {
    return ftrain::invokeApi([&] {
        requirePointer(pattern, "pattern");
        requirePointer(op, "op output");
        *op = pattern->builder.addOperation<ftrain::OperationKind::kMoEReduceRS>(input, weight, routing,
                                                                              expert_weight, output);
    });
}

extern "C" FTrainStatus ftrainArgsSetMoEReduceRS(FTrainArgs args, FTrainMoEReduceRSOpId op,
                                                FTrainMoEReduceRS operation, uint64_t weight_version,
                                                uint64_t routing_version) {
    return ftrain::invokeApi([&] {
        requirePointer(args, "args");
        args->args.setOperation<ftrain::OperationKind::kMoEReduceRS>(
            ftrain::PatternOperationId{op.opaque},
            makeMoeAttributes<ftrain::OperationKind::kMoEReduceRS>(operation, weight_version, routing_version));
    });
}
