#include "flash_train/common.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/api.hpp"
#include "flash_train/api_handles.hpp"
#include "flash_train/error.hpp"
#include "flash_train/matcher.hpp"
#include "flash_train/op_schema.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/runtime.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/trace.hpp"

namespace ftrain {
namespace {

template<typename Pointer>
void requirePointer(Pointer* pointer, const char* name) {
    if (pointer == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s must not be null", name); }
}

void requirePattern(FTrainPattern pattern) {
    if (pattern == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "pattern must not be null"); }
}

void requireOps(FTrainOps ops) {
    if (ops == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ops must not be null"); }
}

void requireArgs(FTrainArgs args) {
    if (args == nullptr) { throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "args must not be null"); }
}

template<typename InternalId, typename Id>
InternalId getInternalId(Id id, const char* name) {
    if constexpr (std::numeric_limits<std::size_t>::max() < std::numeric_limits<std::uint64_t>::max()) {
        if (id.opaque > std::numeric_limits<std::size_t>::max()) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s is out of range", name);
        }
    }
    return InternalId{static_cast<std::size_t>(id.opaque)};
}

std::shared_ptr<const OpsEngineBase> findOpsEngine(const PatternKey& pattern_key, FTrainStatus missing_status,
                                                   const char* missing_message) {
    std::shared_ptr<const OpsEngineBase> ops_engine = getGlobalHandle().findOpsEngine(pattern_key);
    if (ops_engine == nullptr) { throw Exception(missing_status, "%s", missing_message); }
    return ops_engine;
}

std::vector<StorageView> copyStorageViews(const FTrainStorageView* storage_views, std::size_t count, const char* name,
                                          bool optional) {
    if (storage_views == nullptr) {
        if (count != 0 && !optional) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s must not be null when its count is %zu", name, count);
        }
        return {};
    }

    std::vector<StorageView> copied;
    if (count > copied.max_size()) {
        throw Exception(FTRAIN_STATUS_OVERFLOW, "%s count %zu exceeds internal capacity", name, count);
    }
    copied.reserve(count);
    for (std::size_t index = 0; index < count; ++index) { copied.emplace_back(storage_views[index]); }
    return copied;
}

GroupedTensorStorage makeGroupedTensorStorage(std::uint32_t num_groups, const FTrainStorageView& data,
                                              const FTrainStorageView* offsets, const FTrainStorageView* dim_sizes,
                                              const FTrainStorageView* strides) {
    StorageView copied_data(data);

    std::optional<StorageView> copied_offsets;
    if (offsets != nullptr) { copied_offsets.emplace(*offsets); }

    const std::size_t rank                    = data.num_dims;
    std::vector<StorageView> copied_dim_sizes = copyStorageViews(dim_sizes, rank, "dim_sizes", true);
    std::vector<StorageView> copied_strides   = copyStorageViews(strides, rank, "strides", true);
    return GroupedTensorStorage(num_groups, std::move(copied_data), std::move(copied_offsets),
                                std::move(copied_dim_sizes), std::move(copied_strides));
}

void validateNumericType(FTrainNumericType numeric_type, const char* name) {
    if (numeric_type == FTRAIN_NUMERIC_TYPE_INVALID || numeric_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "%s %u is invalid", name,
                        static_cast<unsigned int>(numeric_type));
    }
}

template<typename Attributes, typename PublicOpId, typename Setter>
void setGemm(FTrainArgs args, PublicOpId op, FTrainNumericType compute_type, Setter setter) {
    requireArgs(args);
    validateNumericType(compute_type, "compute_type");
    (args->args.*setter)(getInternalId<OperationId>(op, "op"), Attributes(compute_type));
}

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainOpsCreate(FTrainOps* ops, FTrainPattern pattern) {
    ftrain::ScopedTraceRange trace_range{"ftrainOpsCreate"};
    return ftrain::invokeApi([&] {
        ftrain::requirePointer(ops, "ops");
        ftrain::requirePattern(pattern);

        ftrain::PatternCanonicalization user_canonicalization = ftrain::Matcher::canonicalize(pattern->pattern);
        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine =
            ftrain::findOpsEngine(user_canonicalization.getKey(), FTRAIN_STATUS_UNSUPPORTED,
                                  "Pattern does not match any registered OpsEngine");
        ftrain::RoleMapping role_mapping =
            ftrain::RoleMapping::compose(user_canonicalization, ops_engine->getSupportedCanonicalization());
        auto created_ops =
            std::make_unique<FTrainOpsStruct>(ftrain::Ops(user_canonicalization.getKey(), std::move(role_mapping)));
        *ops = created_ops.release();
    });
}

extern "C" FTrainStatus ftrainOpsDestroy(FTrainOps ops) {
    ftrain::ScopedTraceRange trace_range{"ftrainOpsDestroy"};
    return ftrain::invokeApi([&] {
        ftrain::requireOps(ops);
        delete ops;
    });
}

extern "C" FTrainStatus ftrainArgsCreate(FTrainArgs* args, FTrainOps ops) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsCreate"};
    return ftrain::invokeApi([&] {
        ftrain::requirePointer(args, "args");
        ftrain::requireOps(ops);

        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine = ftrain::findOpsEngine(
            ops->ops.getPatternKey(), FTRAIN_STATUS_INTERNAL_ERROR, "OpsEngine registered for Ops is unavailable");
        auto created_args = std::make_unique<FTrainArgsStruct>(ops->ops, ops_engine->getSupportedSchema());
        *args             = created_args.release();
    });
}

extern "C" FTrainStatus ftrainArgsDestroy(FTrainArgs args) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsDestroy"};
    return ftrain::invokeApi([&] {
        ftrain::requireArgs(args);
        delete args;
    });
}

extern "C" FTrainStatus ftrainArgsSetTensor(FTrainArgs args, FTrainTensorId tensor, FTrainStorageView storage_view) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetTensor"};
    return ftrain::invokeApi([&] {
        ftrain::requireArgs(args);
        ftrain::StorageView copied_storage_view(storage_view);
        args->args.setTensor(ftrain::getInternalId<ftrain::OperandId>(tensor, "tensor"),
                             ftrain::TensorStorage(std::move(copied_storage_view)));
    });
}

extern "C" FTrainStatus ftrainArgsSetTensorList(FTrainArgs args, FTrainTensorListId tensor_list,
                                                const FTrainStorageView storage_views[],
                                                std::uint32_t num_storage_views) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetTensorList"};
    return ftrain::invokeApi([&] {
        ftrain::requireArgs(args);
        std::vector<ftrain::StorageView> copied_storage_views =
            ftrain::copyStorageViews(storage_views, num_storage_views, "storage_views", false);
        args->args.setTensorList(ftrain::getInternalId<ftrain::OperandId>(tensor_list, "tensor_list"),
                                 ftrain::TensorListStorage(std::move(copied_storage_views)));
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedTensor(FTrainArgs args, FTrainGroupedTensorId grouped_tensor,
                                                   std::uint32_t num_groups, FTrainStorageView data,
                                                   const FTrainStorageView* offsets,
                                                   const FTrainStorageView dim_sizes[],
                                                   const FTrainStorageView strides[]) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetGroupedTensor"};
    return ftrain::invokeApi([&] {
        ftrain::requireArgs(args);
        ftrain::GroupedTensorStorage storage =
            ftrain::makeGroupedTensorStorage(num_groups, data, offsets, dim_sizes, strides);
        args->args.setGroupedTensor(ftrain::getInternalId<ftrain::OperandId>(grouped_tensor, "grouped_tensor"),
                                    std::move(storage));
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedABCDGemm(FTrainArgs args, FTrainGroupedABCDGemmOpId op,
                                                     FTrainNumericType compute_type) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetGroupedABCDGemm"};
    return ftrain::invokeApi([&] {
        ftrain::setGemm<ftrain::GroupedABCDGemmAttributes>(args, op, compute_type, &ftrain::Args::setGroupedABCDGemm);
    });
}

extern "C" FTrainStatus ftrainArgsSetGemm(FTrainArgs args, FTrainGemmOpId op, FTrainNumericType compute_type) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetGemm"};
    return ftrain::invokeApi(
        [&] { ftrain::setGemm<ftrain::GemmAttributes>(args, op, compute_type, &ftrain::Args::setGemm); });
}

extern "C" FTrainStatus ftrainArgsSetGroupedBCDGemm(FTrainArgs args, FTrainGroupedBCDGemmOpId op,
                                                    FTrainNumericType compute_type) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetGroupedBCDGemm"};
    return ftrain::invokeApi([&] {
        ftrain::setGemm<ftrain::GroupedBCDGemmAttributes>(args, op, compute_type, &ftrain::Args::setGroupedBCDGemm);
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedABGemm(FTrainArgs args, FTrainGroupedABGemmOpId op,
                                                   FTrainNumericType compute_type) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsSetGroupedABGemm"};
    return ftrain::invokeApi([&] {
        ftrain::setGemm<ftrain::GroupedABGemmAttributes>(args, op, compute_type, &ftrain::Args::setGroupedABGemm);
    });
}
