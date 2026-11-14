// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "flash_train/api.hpp"

extern "C" FTrainStatus ftrainPatternCreate(FTrainPattern* pattern) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternCreate: pattern output must not be null");
        }

        auto created_pattern = std::make_unique<FTrainPatternStruct>();
        *pattern             = created_pattern.release();
    });
}

extern "C" FTrainStatus ftrainPatternDestroy(FTrainPattern pattern) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPatternDestroy: pattern must not be null");
        }
        delete pattern;
    });
}

extern "C" FTrainStatus ftrainPatternAddTensor(FTrainPattern pattern, FTrainTensorId* tensor) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPatternAddTensor: pattern must not be null");
        }
        if (tensor == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddTensor: tensor output must not be null");
        }
        *tensor = pattern->builder.addOperand<ftrain::OperandKind::kTensor>();
    });
}

extern "C" FTrainStatus ftrainPatternAddTensorList(FTrainPattern pattern, FTrainTensorListId* tensor_list) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddTensorList: pattern must not be null");
        }
        if (tensor_list == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddTensorList: tensor_list output must not be null");
        }
        *tensor_list = pattern->builder.addOperand<ftrain::OperandKind::kTensorList>();
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedTensor(FTrainPattern pattern, FTrainGroupedTensorId* grouped_tensor) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedTensor: pattern must not be null");
        }
        if (grouped_tensor == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedTensor: grouped_tensor output must not be null");
        }
        *grouped_tensor = pattern->builder.addOperand<ftrain::OperandKind::kGroupedTensor>();
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedABCDGemm(FTrainPattern pattern, FTrainGroupedABCDGemmOpId* op,
                                                        FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                        FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                        FTrainTensorId alpha, FTrainTensorId beta) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedABCDGemm: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedABCDGemm: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGroupedABCDGemm>(a, b, c, d, alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGemm(FTrainPattern pattern, FTrainGemmOpId* op, FTrainTensorId a,
                                             FTrainTensorId b, FTrainTensorId c, FTrainTensorId d, FTrainTensorId alpha,
                                             FTrainTensorId beta) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPatternAddGemm: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "ftrainPatternAddGemm: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGemm>(a, b, c, d, alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedBCDGemm(FTrainPattern pattern, FTrainGroupedBCDGemmOpId* op,
                                                       FTrainTensorListId a, FTrainGroupedTensorId b,
                                                       FTrainGroupedTensorId c, FTrainGroupedTensorId d,
                                                       FTrainTensorId alpha, FTrainTensorId beta) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedBCDGemm: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedBCDGemm: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGroupedBCDGemm>(a, b, c, d, alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddGroupedABGemm(FTrainPattern pattern, FTrainGroupedABGemmOpId* op,
                                                      FTrainGroupedTensorId a, FTrainGroupedTensorId b,
                                                      FTrainTensorListId c, FTrainTensorListId d, FTrainTensorId alpha,
                                                      FTrainTensorId beta) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedABGemm: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddGroupedABGemm: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kGroupedABGemm>(a, b, c, d, alpha, beta);
    });
}

extern "C" FTrainStatus ftrainPatternAddKdaForward(FTrainPattern pattern, FTrainKdaForwardOpId* op, FTrainTensorId q,
                                                   FTrainTensorId k, FTrainTensorId v, FTrainTensorId g,
                                                   FTrainTensorId beta, FTrainTensorId a_log, FTrainTensorId dt_bias,
                                                   FTrainTensorId cu_seqlens, FTrainTensorId o, FTrainTensorId q_rstd,
                                                   FTrainTensorId k_rstd, FTrainTensorId aqk, FTrainTensorId akk,
                                                   FTrainTensorId qn, FTrainTensorId kn, FTrainTensorId h,
                                                   FTrainTensorId state) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddKdaForward: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddKdaForward: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kKdaForward>(
            q, k, v, g, beta, a_log, dt_bias, cu_seqlens, o, q_rstd, k_rstd, aqk, akk, qn, kn, h, state);
    });
}

extern "C" FTrainStatus ftrainPatternAddKdaBackward(
    FTrainPattern pattern, FTrainKdaBackwardOpId* op, FTrainTensorId d_o, FTrainTensorId q, FTrainTensorId k,
    FTrainTensorId v, FTrainTensorId g, FTrainTensorId beta, FTrainTensorId a_log, FTrainTensorId dt_bias,
    FTrainTensorId cu_seqlens, FTrainTensorId q_rstd, FTrainTensorId k_rstd, FTrainTensorId aqk, FTrainTensorId akk,
    FTrainTensorId qn, FTrainTensorId kn, FTrainTensorId dq, FTrainTensorId dk, FTrainTensorId dv, FTrainTensorId dg,
    FTrainTensorId dbeta, FTrainTensorId da_log, FTrainTensorId ddt_bias, FTrainTensorId h, FTrainTensorId state) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddKdaBackward: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddKdaBackward: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kKdaBackward>(
            d_o, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, dq, dk, dv, dg, dbeta,
            da_log, ddt_bias, h, state);
    });
}
