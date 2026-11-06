// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/trace.hpp"
#include "flash_train/engine.hpp"
#include "flash_train/handle.hpp"
#include "flash_train/api.hpp"

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
        auto created_ops = std::make_unique<FTrainOpsStruct>(ftrain::Ops{prepared_pattern, ops_engine->getPattern()});
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

// DSA operators: one staged Args setter per family.

extern "C" FTrainStatus ftrainArgsSetIndexerFwd(FTrainArgs args, FTrainIndexerFwdOpId op,
                                                FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerFwd>(ftrain::PatternOperationId{op.opaque},
                                                                    ftrain::IndexerFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerFwdThd(FTrainArgs args, FTrainIndexerFwdThdOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerFwdThd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerFwdThd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerFwdThd>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::IndexerFwdThdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetDenseIndexerScore(FTrainArgs args, FTrainDenseIndexerScoreOpId op,
                                                       FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseIndexerScore: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseIndexerScore: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kDenseIndexerScore>(
            ftrain::PatternOperationId{op.opaque}, ftrain::DenseIndexerScoreAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetDenseIndexerScoreThd(FTrainArgs args, FTrainDenseIndexerScoreThdOpId op,
                                                          FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseIndexerScoreThd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseIndexerScoreThd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kDenseIndexerScoreThd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::DenseIndexerScoreThdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetDenseAttnScore(FTrainArgs args, FTrainDenseAttnScoreOpId op,
                                                    FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseAttnScore: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseAttnScore: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kDenseAttnScore>(ftrain::PatternOperationId{op.opaque},
                                                                        ftrain::DenseAttnScoreAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetDenseAttnScoreThd(FTrainArgs args, FTrainDenseAttnScoreThdOpId op,
                                                       FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseAttnScoreThd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetDenseAttnScoreThd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kDenseAttnScoreThd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::DenseAttnScoreThdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSparseIndexerPredict(FTrainArgs args, FTrainSparseIndexerPredictOpId op,
                                                          FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseIndexerPredict: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseIndexerPredict: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSparseIndexerPredict>(
            ftrain::PatternOperationId{op.opaque}, ftrain::SparseIndexerPredictAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSparseAttnTarget(FTrainArgs args, FTrainSparseAttnTargetOpId op,
                                                      FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnTarget: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnTarget: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSparseAttnTarget>(
            ftrain::PatternOperationId{op.opaque}, ftrain::SparseAttnTargetAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSparseAttnFwd(FTrainArgs args, FTrainSparseAttnFwdOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSparseAttnFwd>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::SparseAttnFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSparseAttnFwdIndexer(FTrainArgs args, FTrainSparseAttnFwdIndexerOpId op,
                                                          FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnFwdIndexer: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnFwdIndexer: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSparseAttnFwdIndexer>(
            ftrain::PatternOperationId{op.opaque}, ftrain::SparseAttnFwdIndexerAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSparseAttnBwd(FTrainArgs args, FTrainSparseAttnBwdOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSparseAttnBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSparseAttnBwd>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::SparseAttnBwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerBwdSparse(FTrainArgs args, FTrainIndexerBwdSparseOpId op,
                                                      FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdSparse: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdSparse: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerBwdSparse>(
            ftrain::PatternOperationId{op.opaque}, ftrain::IndexerBwdSparseAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerBwdDense(FTrainArgs args, FTrainIndexerBwdDenseOpId op,
                                                     FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdDense: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdDense: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerBwdDense>(
            ftrain::PatternOperationId{op.opaque}, ftrain::IndexerBwdDenseAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerBwdDenseThd(FTrainArgs args, FTrainIndexerBwdDenseThdOpId op,
                                                        FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdDenseThd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerBwdDenseThd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerBwdDenseThd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::IndexerBwdDenseThdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerTopK(FTrainArgs args, FTrainIndexerTopKOpId op) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerTopK: args handle must not be null");
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerTopK>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::IndexerTopKAttributes());
    });
}

extern "C" FTrainStatus ftrainArgsSetIndexerTopKThd(FTrainArgs args, FTrainIndexerTopKThdOpId op) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetIndexerTopKThd: args handle must not be null");
        }
        args->args.setOperation<ftrain::OperationKind::kIndexerTopKThd>(ftrain::PatternOperationId{op.opaque},
                                                                        ftrain::IndexerTopKThdAttributes());
    });
}

extern "C" FTrainStatus ftrainArgsSetCompactify(FTrainArgs args, FTrainCompactifyOpId op) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetCompactify: args handle must not be null");
        }
        args->args.setOperation<ftrain::OperationKind::kCompactify>(ftrain::PatternOperationId{op.opaque},
                                                                    ftrain::CompactifyAttributes());
    });
}

extern "C" FTrainStatus ftrainArgsSetCSACompressorFwd(FTrainArgs args, FTrainCSACompressorFwdOpId op,
                                                      FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetCSACompressorFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetCSACompressorFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kCSACompressorFwd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::CSACompressorFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetCSACompressorBwd(FTrainArgs args, FTrainCSACompressorBwdOpId op,
                                                      FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetCSACompressorBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetCSACompressorBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kCSACompressorBwd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::CSACompressorBwdAttributes(compute_type));
    });
}

// mHC operators.

extern "C" FTrainStatus ftrainArgsSetSinkhornFwd(FTrainArgs args, FTrainSinkhornFwdOpId op,
                                                 FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSinkhornFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSinkhornFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSinkhornFwd>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::SinkhornFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetSinkhornBwd(FTrainArgs args, FTrainSinkhornBwdOpId op,
                                                 FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSinkhornBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetSinkhornBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kSinkhornBwd>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::SinkhornBwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetHAggregateFwd(FTrainArgs args, FTrainHAggregateFwdOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHAggregateFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHAggregateFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kHAggregateFwd>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::HAggregateFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetHAggregateBwd(FTrainArgs args, FTrainHAggregateBwdOpId op,
                                                   FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHAggregateBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHAggregateBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kHAggregateBwd>(ftrain::PatternOperationId{op.opaque},
                                                                       ftrain::HAggregateBwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetHPostBdaFwd(FTrainArgs args, FTrainHPostBdaFwdOpId op,
                                                 FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHPostBdaFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHPostBdaFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kHPostBdaFwd>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::HPostBdaFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetHPostBdaBwd(FTrainArgs args, FTrainHPostBdaBwdOpId op,
                                                 FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHPostBdaBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetHPostBdaBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kHPostBdaBwd>(ftrain::PatternOperationId{op.opaque},
                                                                     ftrain::HPostBdaBwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetProjRmsFwd(FTrainArgs args, FTrainProjRmsFwdOpId op,
                                                FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kProjRmsFwd>(ftrain::PatternOperationId{op.opaque},
                                                                    ftrain::ProjRmsFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetProjRmsBwd(FTrainArgs args, FTrainProjRmsBwdOpId op,
                                                FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kProjRmsBwd>(ftrain::PatternOperationId{op.opaque},
                                                                    ftrain::ProjRmsBwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetProjRmsComputeHFwd(FTrainArgs args, FTrainProjRmsComputeHFwdOpId op,
                                                        FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsComputeHFwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsComputeHFwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kProjRmsComputeHFwd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::ProjRmsComputeHFwdAttributes(compute_type));
    });
}

extern "C" FTrainStatus ftrainArgsSetProjRmsComputeHBwd(FTrainArgs args, FTrainProjRmsComputeHBwdOpId op,
                                                        FTrainNumericType compute_type) {
    return ftrain::invokeApi([&] {
        if (args == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsComputeHBwd: args handle must not be null");
        }
        if (compute_type == FTRAIN_NUMERIC_TYPE_INVALID || compute_type >= FTRAIN_NUMERIC_TYPE_COUNT) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainArgsSetProjRmsComputeHBwd: compute_type %u is invalid",
                                    static_cast<unsigned int>(compute_type));
        }
        args->args.setOperation<ftrain::OperationKind::kProjRmsComputeHBwd>(
            ftrain::PatternOperationId{op.opaque}, ftrain::ProjRmsComputeHBwdAttributes(compute_type));
    });
}
