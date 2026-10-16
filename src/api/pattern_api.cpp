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

// DSA operators: one staged Pattern adder per family.

extern "C" FTrainStatus ftrainPatternAddIndexerFwd(FTrainPattern pattern, FTrainIndexerFwdOpId* op, FTrainTensorId q,
                                                   FTrainTensorId k, FTrainTensorId w, FTrainTensorId scores,
                                                   FTrainTensorId ratio, FTrainTensorId sm_scale) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerFwd>(q, k, w, scores, ratio, sm_scale);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerFwdThd(FTrainPattern pattern, FTrainIndexerFwdThdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                      FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                      FTrainTensorId q_causal_offsets, FTrainTensorId scores,
                                                      FTrainTensorId ratio, FTrainTensorId sm_scale,
                                                      FTrainTensorId max_seqlen_kv) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerFwdThd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerFwdThd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerFwdThd>(
            q, k, w, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, scores, ratio, sm_scale, max_seqlen_kv);
    });
}

extern "C" FTrainStatus ftrainPatternAddDenseIndexerScore(FTrainPattern pattern, FTrainDenseIndexerScoreOpId* op,
                                                          FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                          FTrainTensorId out, FTrainTensorId denom,
                                                          FTrainTensorId ratio, FTrainTensorId sm_scale) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseIndexerScore: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseIndexerScore: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kDenseIndexerScore>(q, k, w, out, denom, ratio,
                                                                                       sm_scale);
    });
}

extern "C" FTrainStatus ftrainPatternAddDenseIndexerScoreThd(FTrainPattern pattern, FTrainDenseIndexerScoreThdOpId* op,
                                                             FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                             FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                             FTrainTensorId q_causal_offsets, FTrainTensorId out,
                                                             FTrainTensorId denom, FTrainTensorId ratio,
                                                             FTrainTensorId sm_scale, FTrainTensorId max_seqlen_kv) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseIndexerScoreThd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseIndexerScoreThd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kDenseIndexerScoreThd>(
            q, k, w, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out, denom, ratio, sm_scale, max_seqlen_kv);
    });
}

extern "C" FTrainStatus ftrainPatternAddDenseAttnScore(FTrainPattern pattern, FTrainDenseAttnScoreOpId* op,
                                                       FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                       FTrainTensorId out, FTrainTensorId denom, FTrainTensorId ratio,
                                                       FTrainTensorId softmax_scale) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseAttnScore: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseAttnScore: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kDenseAttnScore>(q, k, lse, out, denom, ratio,
                                                                                    softmax_scale);
    });
}

extern "C" FTrainStatus ftrainPatternAddDenseAttnScoreThd(FTrainPattern pattern, FTrainDenseAttnScoreThdOpId* op,
                                                          FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                          FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_kv,
                                                          FTrainTensorId q_causal_offsets, FTrainTensorId out,
                                                          FTrainTensorId denom, FTrainTensorId ratio,
                                                          FTrainTensorId softmax_scale, FTrainTensorId max_seqlen_kv) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseAttnScoreThd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddDenseAttnScoreThd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kDenseAttnScoreThd>(
            q, k, lse, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out, denom, ratio, softmax_scale, max_seqlen_kv);
    });
}

extern "C" FTrainStatus ftrainPatternAddSparseIndexerPredict(FTrainPattern pattern, FTrainSparseIndexerPredictOpId* op,
                                                             FTrainTensorId q, FTrainTensorId k, FTrainTensorId w,
                                                             FTrainTensorId topk_indices, FTrainTensorId predict,
                                                             FTrainTensorId sm_scale,
                                                             FTrainTensorId topk_indices_global) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseIndexerPredict: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseIndexerPredict: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSparseIndexerPredict>(
            q, k, w, topk_indices, predict, sm_scale, topk_indices_global);
    });
}

extern "C" FTrainStatus ftrainPatternAddSparseAttnTarget(FTrainPattern pattern, FTrainSparseAttnTargetOpId* op,
                                                         FTrainTensorId q, FTrainTensorId k, FTrainTensorId lse,
                                                         FTrainTensorId topk_indices, FTrainTensorId target,
                                                         FTrainTensorId softmax_scale,
                                                         FTrainTensorId topk_indices_global) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnTarget: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnTarget: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSparseAttnTarget>(
            q, k, lse, topk_indices, target, softmax_scale, topk_indices_global);
    });
}

extern "C" FTrainStatus ftrainPatternAddSparseAttnFwd(FTrainPattern pattern, FTrainSparseAttnFwdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId kv, FTrainTensorId attn_sink,
                                                      FTrainTensorId topk_indices, FTrainTensorId topk_length,
                                                      FTrainTensorId out, FTrainTensorId lse,
                                                      FTrainTensorId softmax_scale) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSparseAttnFwd>(
            q, kv, attn_sink, topk_indices, topk_length, out, lse, softmax_scale);
    });
}

extern "C" FTrainStatus ftrainPatternAddSparseAttnFwdIndexer(FTrainPattern pattern, FTrainSparseAttnFwdIndexerOpId* op,
                                                             FTrainTensorId q, FTrainTensorId kv,
                                                             FTrainTensorId attn_sink, FTrainTensorId topk_indices,
                                                             FTrainTensorId topk_length, FTrainTensorId out,
                                                             FTrainTensorId max_logits, FTrainTensorId lse,
                                                             FTrainTensorId lse_indexer, FTrainTensorId softmax_scale,
                                                             FTrainTensorId indexer_topk) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnFwdIndexer: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnFwdIndexer: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSparseAttnFwdIndexer>(
            q, kv, attn_sink, topk_indices, topk_length, out, max_logits, lse, lse_indexer, softmax_scale,
            indexer_topk);
    });
}

extern "C" FTrainStatus ftrainPatternAddSparseAttnBwd(FTrainPattern pattern, FTrainSparseAttnBwdOpId* op,
                                                      FTrainTensorId q, FTrainTensorId kv, FTrainTensorId out,
                                                      FTrainTensorId dout, FTrainTensorId lse, FTrainTensorId attn_sink,
                                                      FTrainTensorId topk_indices, FTrainTensorId topk_length,
                                                      FTrainTensorId dq, FTrainTensorId dkv, FTrainTensorId dsink,
                                                      FTrainTensorId softmax_scale) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSparseAttnBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSparseAttnBwd>(
            q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq, dkv, dsink, softmax_scale);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerBwdSparse(
    FTrainPattern pattern, FTrainIndexerBwdSparseOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId index_score, FTrainTensorId topk_indices,
    FTrainTensorId grad_attn_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk, FTrainTensorId sm_scale,
    FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId topk_indices_global) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdSparse: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdSparse: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerBwdSparse>(
            index_q, weights, index_k, attn_score, index_score, topk_indices, grad_attn_score, dq, dw, dk, sm_scale,
            loss_coeff, grad_loss, topk_indices_global);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerBwdDense(
    FTrainPattern pattern, FTrainIndexerBwdDenseOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId attn_l1norm, FTrainTensorId index_score,
    FTrainTensorId index_lse, FTrainTensorId grad_index_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk,
    FTrainTensorId sm_scale, FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId ratio) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdDense: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdDense: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerBwdDense>(
            index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, grad_index_score, dq, dw, dk,
            sm_scale, loss_coeff, grad_loss, ratio);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerBwdDenseThd(
    FTrainPattern pattern, FTrainIndexerBwdDenseThdOpId* op, FTrainTensorId index_q, FTrainTensorId weights,
    FTrainTensorId index_k, FTrainTensorId attn_score, FTrainTensorId attn_l1norm, FTrainTensorId index_score,
    FTrainTensorId index_lse, FTrainTensorId cu_seqlens_q, FTrainTensorId cu_seqlens_k, FTrainTensorId q_causal_offsets,
    FTrainTensorId grad_index_score, FTrainTensorId dq, FTrainTensorId dw, FTrainTensorId dk, FTrainTensorId sm_scale,
    FTrainTensorId loss_coeff, FTrainTensorId grad_loss, FTrainTensorId ratio) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdDenseThd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerBwdDenseThd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerBwdDenseThd>(
            index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, cu_seqlens_q, cu_seqlens_k,
            q_causal_offsets, grad_index_score, dq, dw, dk, sm_scale, loss_coeff, grad_loss, ratio);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerTopK(FTrainPattern pattern, FTrainIndexerTopKOpId* op,
                                                    FTrainTensorId input_values, FTrainTensorId seq_lens,
                                                    FTrainTensorId out_indices, FTrainTensorId out_values,
                                                    FTrainTensorId top_k, FTrainTensorId next_n,
                                                    FTrainTensorId order_values) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerTopK: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerTopK: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerTopK>(
            input_values, seq_lens, out_indices, out_values, top_k, next_n, order_values);
    });
}

extern "C" FTrainStatus ftrainPatternAddIndexerTopKThd(FTrainPattern pattern, FTrainIndexerTopKThdOpId* op,
                                                       FTrainTensorId input_values, FTrainTensorId cu_seqlens_q,
                                                       FTrainTensorId cu_seqlens_kv, FTrainTensorId q_causal_offsets,
                                                       FTrainTensorId out_indices, FTrainTensorId out_values,
                                                       FTrainTensorId ratio, FTrainTensorId top_k,
                                                       FTrainTensorId order_values) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerTopKThd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddIndexerTopKThd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kIndexerTopKThd>(
            input_values, cu_seqlens_q, cu_seqlens_kv, q_causal_offsets, out_indices, out_values, ratio, top_k,
            order_values);
    });
}

extern "C" FTrainStatus ftrainPatternAddCompactify(FTrainPattern pattern, FTrainCompactifyOpId* op,
                                                   FTrainTensorId indices, FTrainTensorId output,
                                                   FTrainTensorId topk_length) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCompactify: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCompactify: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kCompactify>(indices, output, topk_length);
    });
}

extern "C" FTrainStatus ftrainPatternAddCSACompressorFwd(FTrainPattern pattern, FTrainCSACompressorFwdOpId* op,
                                                         FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                         FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                         FTrainTensorId compressed, FTrainTensorId ratio,
                                                         FTrainTensorId head_dim, FTrainTensorId coff,
                                                         FTrainTensorId total_comp) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCSACompressorFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCSACompressorFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kCSACompressorFwd>(
            kv, score, ape, cu_seqlens, cu_seqlens_comp, compressed, ratio, head_dim, coff, total_comp);
    });
}

extern "C" FTrainStatus ftrainPatternAddCSACompressorBwd(FTrainPattern pattern, FTrainCSACompressorBwdOpId* op,
                                                         FTrainTensorId kv, FTrainTensorId score, FTrainTensorId ape,
                                                         FTrainTensorId cu_seqlens, FTrainTensorId cu_seqlens_comp,
                                                         FTrainTensorId grad_compressed, FTrainTensorId grad_kv,
                                                         FTrainTensorId grad_score, FTrainTensorId grad_ape,
                                                         FTrainTensorId ratio, FTrainTensorId head_dim,
                                                         FTrainTensorId coff) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCSACompressorBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddCSACompressorBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kCSACompressorBwd>(
            kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_compressed, grad_kv, grad_score, grad_ape, ratio,
            head_dim, coff);
    });
}

// mHC operators.

extern "C" FTrainStatus ftrainPatternAddSinkhornFwd(FTrainPattern pattern, FTrainSinkhornFwdOpId* op,
                                                    FTrainTensorId logits, FTrainTensorId output, FTrainTensorId m_init,
                                                    FTrainTensorId num_iterations, FTrainTensorId eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSinkhornFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSinkhornFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSinkhornFwd>(logits, output, m_init, num_iterations,
                                                                                 eps);
    });
}

extern "C" FTrainStatus ftrainPatternAddSinkhornBwd(FTrainPattern pattern, FTrainSinkhornBwdOpId* op,
                                                    FTrainTensorId grad_output, FTrainTensorId m_init,
                                                    FTrainTensorId grad_logits, FTrainTensorId num_iterations,
                                                    FTrainTensorId eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSinkhornBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddSinkhornBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kSinkhornBwd>(grad_output, m_init, grad_logits,
                                                                                 num_iterations, eps);
    });
}

extern "C" FTrainStatus ftrainPatternAddHAggregateFwd(FTrainPattern pattern, FTrainHAggregateFwdOpId* op,
                                                      FTrainTensorId x, FTrainTensorId h_pre,
                                                      FTrainTensorId aggregated) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHAggregateFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHAggregateFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kHAggregateFwd>(x, h_pre, aggregated);
    });
}

extern "C" FTrainStatus ftrainPatternAddHAggregateBwd(FTrainPattern pattern, FTrainHAggregateBwdOpId* op,
                                                      FTrainTensorId grad_aggregated, FTrainTensorId x,
                                                      FTrainTensorId h_pre, FTrainTensorId grad_x,
                                                      FTrainTensorId grad_h_pre) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHAggregateBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHAggregateBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kHAggregateBwd>(grad_aggregated, x, h_pre, grad_x,
                                                                                   grad_h_pre);
    });
}

extern "C" FTrainStatus ftrainPatternAddHPostBdaFwd(FTrainPattern pattern, FTrainHPostBdaFwdOpId* op,
                                                    FTrainTensorId h_res, FTrainTensorId residual,
                                                    FTrainTensorId h_post, FTrainTensorId x, FTrainTensorId bias,
                                                    FTrainTensorId output) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHPostBdaFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHPostBdaFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kHPostBdaFwd>(h_res, residual, h_post, x, bias,
                                                                                 output);
    });
}

extern "C" FTrainStatus ftrainPatternAddHPostBdaBwd(FTrainPattern pattern, FTrainHPostBdaBwdOpId* op,
                                                    FTrainTensorId grad_output, FTrainTensorId residual,
                                                    FTrainTensorId x, FTrainTensorId bias, FTrainTensorId h_res,
                                                    FTrainTensorId h_post, FTrainTensorId grad_residual,
                                                    FTrainTensorId grad_x, FTrainTensorId grad_h_res,
                                                    FTrainTensorId grad_h_post, FTrainTensorId grad_bias) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHPostBdaBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddHPostBdaBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kHPostBdaBwd>(
            grad_output, residual, x, bias, h_res, h_post, grad_residual, grad_x, grad_h_res, grad_h_post, grad_bias);
    });
}

extern "C" FTrainStatus ftrainPatternAddProjRmsFwd(FTrainPattern pattern, FTrainProjRmsFwdOpId* op, FTrainTensorId x,
                                                   FTrainTensorId weight, FTrainTensorId proj, FTrainTensorId r,
                                                   FTrainTensorId eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kProjRmsFwd>(x, weight, proj, r, eps);
    });
}

extern "C" FTrainStatus ftrainPatternAddProjRmsBwd(FTrainPattern pattern, FTrainProjRmsBwdOpId* op,
                                                   FTrainTensorId grad_proj, FTrainTensorId grad_r, FTrainTensorId x,
                                                   FTrainTensorId weight, FTrainTensorId r, FTrainTensorId grad_x,
                                                   FTrainTensorId grad_weight, FTrainTensorId eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kProjRmsBwd>(grad_proj, grad_r, x, weight, r, grad_x,
                                                                                grad_weight, eps);
    });
}

extern "C" FTrainStatus ftrainPatternAddProjRmsComputeHFwd(
    FTrainPattern pattern, FTrainProjRmsComputeHFwdOpId* op, FTrainTensorId x, FTrainTensorId weight,
    FTrainTensorId alpha_pre, FTrainTensorId alpha_post, FTrainTensorId alpha_res, FTrainTensorId bias,
    FTrainTensorId h_pre, FTrainTensorId h_post, FTrainTensorId h_res, FTrainTensorId r, FTrainTensorId proj,
    FTrainTensorId n, FTrainTensorId eps, FTrainTensorId compute_h_eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsComputeHFwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsComputeHFwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kProjRmsComputeHFwd>(
            x, weight, alpha_pre, alpha_post, alpha_res, bias, h_pre, h_post, h_res, r, proj, n, eps, compute_h_eps);
    });
}

extern "C" FTrainStatus ftrainPatternAddProjRmsComputeHBwd(
    FTrainPattern pattern, FTrainProjRmsComputeHBwdOpId* op, FTrainTensorId grad_h_pre, FTrainTensorId grad_h_post,
    FTrainTensorId grad_h_res, FTrainTensorId grad_r, FTrainTensorId h_pre, FTrainTensorId h_post, FTrainTensorId proj,
    FTrainTensorId r_ch, FTrainTensorId x, FTrainTensorId weight, FTrainTensorId alpha_pre, FTrainTensorId alpha_post,
    FTrainTensorId alpha_res, FTrainTensorId grad_x, FTrainTensorId grad_weight, FTrainTensorId grad_alpha,
    FTrainTensorId grad_bias, FTrainTensorId n, FTrainTensorId eps, FTrainTensorId compute_h_eps) {
    return ftrain::invokeApi([&] {
        if (pattern == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsComputeHBwd: pattern must not be null");
        }
        if (op == nullptr) {
            throw ftrain::Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                                    "ftrainPatternAddProjRmsComputeHBwd: op output must not be null");
        }
        *op = pattern->builder.addOperation<ftrain::OperationKind::kProjRmsComputeHBwd>(
            grad_h_pre, grad_h_post, grad_h_res, grad_r, h_pre, h_post, proj, r_ch, x, weight, alpha_pre, alpha_post,
            alpha_res, grad_x, grad_weight, grad_alpha, grad_bias, n, eps, compute_h_eps);
    });
}
