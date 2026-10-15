// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/csa_compressor_bwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the CSACompressorBwd family contributes to its engine.
struct CSACompressorBwdFamily {
    using Problem = CSACompressorBwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId kv;
        FTrainTensorId score;
        FTrainTensorId ape;
        FTrainTensorId cu_seqlens;
        FTrainTensorId cu_seqlens_comp;
        FTrainTensorId grad_compressed;
        FTrainTensorId grad_kv;
        FTrainTensorId grad_score;
        FTrainTensorId grad_ape;
        FTrainTensorId ratio;
        FTrainTensorId head_dim;
        FTrainTensorId coff;
        FTrainCSACompressorBwdOpId csa_compressor_bwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId kv              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId score           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ape             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_comp = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_compressed = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_kv         = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_score      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId grad_ape        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId head_dim        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId coff            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainCSACompressorBwdOpId csa_compressor_bwd =
                pattern.addOperation<OperationKind::kCSACompressorBwd>(kv, score, ape, cu_seqlens, cu_seqlens_comp,
                                                                       grad_compressed, grad_kv, grad_score, grad_ape,
                                                                       ratio, head_dim, coff);
            return Roles{pattern.buildPattern(),
                         kv,
                         score,
                         ape,
                         cu_seqlens,
                         cu_seqlens_comp,
                         grad_compressed,
                         grad_kv,
                         grad_score,
                         grad_ape,
                         ratio,
                         head_dim,
                         coff,
                         csa_compressor_bwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<CSACompressorBwdProblem>>> makeRecords();

    static CSACompressorBwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return CSACompressorBwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.score))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ape))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_comp))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_compressed))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_score))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.grad_ape))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.head_dim))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.coff))),
            std::get<CSACompressorBwdAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kCSACompressorBwd>::createPatternOperationId(
                    roles.csa_compressor_bwd)))};
    }
};

}  // namespace ftrain

#endif
