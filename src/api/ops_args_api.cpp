#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/common.h"

#include "flash_train/api.hpp"
#include "flash_train/error.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/engine/base.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/tensor.hpp"
#include "flash_train/trace.hpp"

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
            ftrain::getGlobalHandle().findOpsEngine(prepared_pattern.getKey());
        if (ops_engine == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_UNSUPPORTED,
                                    "ftrainOpsCreate: PatternBuilder does not match any registered OpsEngine");
        }
        auto created_ops = std::make_unique<FTrainOpsStruct>(prepared_pattern, ops_engine->getPattern());
        *ops             = created_ops.release();
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

        auto created_args = std::make_unique<FTrainArgsStruct>(ops->ops.makeArgs());
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
        args->args.setOperand<ftrain::OperandKind::kTensor>(ftrain::PatternOperandId{tensor.opaque},
                                                            ftrain::Tensor{std::move(copied_storage_view)});
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
        if (storage_views == nullptr && num_storage_views != 0) {
            throw ftrain::Exception(
                FTRAIN_STATUS_INVALID_ARGUMENT,
                "ftrainArgsSetTensorList: storage_views must not be null when num_storage_views is %u",
                num_storage_views);
        }
        std::vector<ftrain::StorageView> copied_storage_views;
        copied_storage_views.reserve(num_storage_views);
        for (std::uint32_t index = 0; index < num_storage_views; ++index) {
            copied_storage_views.emplace_back(storage_views[index]);
        }
        args->args.setOperand<ftrain::OperandKind::kTensorList>(ftrain::PatternOperandId{tensor_list.opaque},
                                                                ftrain::TensorList{std::move(copied_storage_views)});
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
        ftrain::StorageView copied_data(data);

        std::optional<ftrain::StorageView> copied_offsets;
        if (offsets != nullptr) { copied_offsets.emplace(*offsets); }

        const std::size_t rank = data.num_dims;
        std::vector<ftrain::StorageView> copied_dim_sizes;
        if (dim_sizes != nullptr) {
            copied_dim_sizes.reserve(rank);
            for (std::size_t index = 0; index < rank; ++index) { copied_dim_sizes.emplace_back(dim_sizes[index]); }
        }
        std::vector<ftrain::StorageView> copied_strides;
        if (strides != nullptr) {
            copied_strides.reserve(rank);
            for (std::size_t index = 0; index < rank; ++index) { copied_strides.emplace_back(strides[index]); }
        }
        args->args.setOperand<ftrain::OperandKind::kGroupedTensor>(
            ftrain::PatternOperandId{grouped_tensor.opaque},
            ftrain::GroupedTensor(num_groups, std::move(copied_data), std::move(copied_offsets),
                                  std::move(copied_dim_sizes), std::move(copied_strides)));
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
            ftrain::PatternOperationId{op.opaque}, ftrain::GroupedABCDGemmAttributes(compute_type));
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
        args->args.setOperation<ftrain::OperationKind::kGemm>(ftrain::PatternOperationId{op.opaque},
                                                              ftrain::GemmAttributes(compute_type));
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
        args->args.setOperation<ftrain::OperationKind::kGroupedBCDGemm>(ftrain::PatternOperationId{op.opaque},
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
        args->args.setOperation<ftrain::OperationKind::kGroupedABGemm>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::GroupedABGemmAttributes(compute_type));
    });
}
