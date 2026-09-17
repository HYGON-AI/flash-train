#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/api.hpp"
#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/ops_engine.hpp"
#include "flash_train/registry.hpp"
#include "flash_train/binding.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/trace.hpp"

namespace ftrain {
namespace {

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

}  // namespace
}  // namespace ftrain

extern "C" FTrainStatus ftrainOpsCreate(FTrainOps* ops, FTrainPattern pattern) {
    ftrain::ScopedTraceRange trace_range{"ftrainOpsCreate"};
    return ftrain::invokeApi([&] {
        if (ops == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainOpsCreate: ops output must not be null");
        }
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainOpsCreate: pattern must not be null");
        }

        const ftrain::Pattern prepared_pattern = pattern->builder.buildPattern();
        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine =
            ftrain::findOpsEngine(prepared_pattern.getKey(), FTRAIN_STATUS_UNSUPPORTED,
                                  "ftrainOpsCreate: PatternBuilder does not match any registered OpsEngine");
        std::optional<ftrain::RoleMapping> role_mapping =
            ftrain::matchRoles(prepared_pattern, ops_engine->getSupportedPattern());
        if (!role_mapping.has_value()) {
            throw ftrain::Exception(FTRAIN_STATUS_UNSUPPORTED,
                                    "ftrainOpsCreate: PatternBuilder does not match its registered OpsEngine");
        }
        auto created_ops =
            std::make_unique<FTrainOpsStruct>(ftrain::Ops(ops_engine->getPatternKey(), std::move(*role_mapping)));
        *ops = created_ops.release();
    });
}

extern "C" FTrainStatus ftrainOpsDestroy(FTrainOps ops) {
    return ftrain::invokeApi([&] {
        if (ops == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainOpsDestroy: ops handle must not be null");
        }
        delete ops;
    });
}

extern "C" FTrainStatus ftrainArgsCreate(FTrainArgs* args, FTrainOps ops) {
    ftrain::ScopedTraceRange trace_range{"ftrainArgsCreate"};
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainArgsCreate: args output must not be null");
        }
        if (ops == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainArgsCreate: ops handle must not be null");
        }

        const std::shared_ptr<const ftrain::OpsEngineBase> ops_engine = ftrain::findOpsEngine(
            ops->ops.getPatternKey(), FTRAIN_STATUS_INTERNAL_ERROR, "OpsEngine registered for Ops is unavailable");
        auto created_args = std::make_unique<FTrainArgsStruct>(ops->ops, ops_engine->getSupportedPattern());
        *args             = created_args.release();
    });
}

extern "C" FTrainStatus ftrainArgsDestroy(FTrainArgs args) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainArgsDestroy: args handle must not be null");
        }
        delete args;
    });
}

extern "C" FTrainStatus ftrainArgsSetTensor(FTrainArgs args, FTrainTensorId tensor, FTrainStorageView storage_view) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetTensor: args handle must not be null");
        }
        ftrain::StorageView copied_storage_view(storage_view);
        args->args.setOperand<ftrain::OperandKind::kTensor>(
            ftrain::getInternalId<ftrain::PatternOperandId>(tensor, "tensor"),
            ftrain::Tensor{ftrain::TensorStorage(std::move(copied_storage_view))});
    });
}

extern "C" FTrainStatus ftrainArgsSetTensorList(FTrainArgs args, FTrainTensorListId tensor_list,
                                                const FTrainStorageView storage_views[],
                                                std::uint32_t num_storage_views) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetTensorList: args handle must not be null");
        }
        std::vector<ftrain::StorageView> copied_storage_views =
            ftrain::copyStorageViews(storage_views, num_storage_views, "storage_views", false);
        args->args.setOperand<ftrain::OperandKind::kTensorList>(
            ftrain::getInternalId<ftrain::PatternOperandId>(tensor_list, "tensor_list"),
            ftrain::TensorList{ftrain::TensorListStorage(std::move(copied_storage_views))});
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedTensor(FTrainArgs args, FTrainGroupedTensorId grouped_tensor,
                                                   std::uint32_t num_groups, FTrainStorageView data,
                                                   const FTrainStorageView* offsets,
                                                   const FTrainStorageView dim_sizes[],
                                                   const FTrainStorageView strides[]) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedTensor: args handle must not be null");
        }
        ftrain::GroupedTensorStorage storage =
            ftrain::makeGroupedTensorStorage(num_groups, data, offsets, dim_sizes, strides);
        args->args.setOperand<ftrain::OperandKind::kGroupedTensor>(
            ftrain::getInternalId<ftrain::PatternOperandId>(grouped_tensor, "grouped_tensor"),
            ftrain::GroupedTensor{std::move(storage)});
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedABCDGemm(FTrainArgs args, FTrainGroupedABCDGemmOpId op,
                                                     FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedABCDGemm: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedABCDGemm: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kGroupedABCDGemm>(
            ftrain::getInternalId<ftrain::PatternOperationId>(op, "op"),
            ftrain::GroupedABCDGemmAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetGemm(FTrainArgs args, FTrainGemmOpId op, FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainArgsSetGemm: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainArgsSetGemm: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kGemm>(
            ftrain::getInternalId<ftrain::PatternOperationId>(op, "op"), ftrain::GemmAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedBCDGemm(FTrainArgs args, FTrainGroupedBCDGemmOpId op,
                                                    FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedBCDGemm: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedBCDGemm: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kGroupedBCDGemm>(
            ftrain::getInternalId<ftrain::PatternOperationId>(op, "op"),
            ftrain::GroupedBCDGemmAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetGroupedABGemm(FTrainArgs args, FTrainGroupedABGemmOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedABGemm: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetGroupedABGemm: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kGroupedABGemm>(
            ftrain::getInternalId<ftrain::PatternOperationId>(op, "op"), ftrain::GroupedABGemmAttributes(compute_type));
    });
}
