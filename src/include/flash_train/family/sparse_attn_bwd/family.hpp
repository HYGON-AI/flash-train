// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the SparseAttnBwd family contributes to its engine.
struct SparseAttnBwdFamily {
    using Problem = SparseAttnBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId q;
        FTrainTensorId kv;
        FTrainTensorId out;
        FTrainTensorId dout;
        FTrainTensorId lse;
        FTrainTensorId attn_sink;
        FTrainTensorId topk_indices;
        FTrainTensorId topk_length;
        FTrainTensorId dq;
        FTrainTensorId dkv;
        FTrainTensorId dsink;
        FTrainTensorId softmax_scale;
        FTrainSparseAttnBwdOpId sparse_attn_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId q                        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId kv                       = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId out                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dout                     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId lse                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId attn_sink                = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_indices             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId topk_length              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dq                       = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dkv                      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId dsink                    = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId softmax_scale            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainSparseAttnBwdOpId sparse_attn_bwd = pattern.addOperation<OperationKind::kSparseAttnBwd>(
                q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, dq, dkv, dsink, softmax_scale);
            return Roles{pattern.buildPattern(), q,           kv, out, dout,  lse,           attn_sink,
                         topk_indices,           topk_length, dq, dkv, dsink, softmax_scale, sparse_attn_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<SparseAttnBwdProblem>>> makeRecords();

    static SparseAttnBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return SparseAttnBwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.q))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.kv))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.out))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dout))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.lse))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.attn_sink))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_indices))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.topk_length))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dq))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dkv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.dsink))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.softmax_scale))),
            std::get<SparseAttnBwdAttributes>(*args.getOpArgument(
                OperationTraits<OperationKind::kSparseAttnBwd>::createPatternOperationId(roles.sparse_attn_bwd)))};
    }
};

}  // namespace ftrain

#endif
