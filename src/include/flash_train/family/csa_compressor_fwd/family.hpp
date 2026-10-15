// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FAMILY_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/pattern.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/family/csa_compressor_fwd/problem.hpp"
#include "flash_train/primitive.hpp"

namespace ftrain {

// Everything the CSACompressorFwd family contributes to its engine.
struct CSACompressorFwdFamily {
    using Problem = CSACompressorFwdProblem;

    struct Roles {
        Pattern pattern;
        FTrainTensorId kv;
        FTrainTensorId score;
        FTrainTensorId ape;
        FTrainTensorId cu_seqlens;
        FTrainTensorId cu_seqlens_comp;
        FTrainTensorId compressed;
        FTrainTensorId ratio;
        FTrainTensorId head_dim;
        FTrainTensorId coff;
        FTrainTensorId total_comp;
        FTrainCSACompressorFwdOpId csa_compressor_fwd;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId kv              = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId score           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ape             = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId cu_seqlens_comp = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId compressed      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId ratio           = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId head_dim        = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId coff            = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId total_comp      = pattern.addOperand<OperandKind::kTensor>();
            const FTrainCSACompressorFwdOpId csa_compressor_fwd =
                pattern.addOperation<OperationKind::kCSACompressorFwd>(kv, score, ape, cu_seqlens, cu_seqlens_comp,
                                                                       compressed, ratio, head_dim, coff, total_comp);
            return Roles{pattern.buildPattern(),
                         kv,
                         score,
                         ape,
                         cu_seqlens,
                         cu_seqlens_comp,
                         compressed,
                         ratio,
                         head_dim,
                         coff,
                         total_comp,
                         csa_compressor_fwd};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<CSACompressorFwdProblem>>> makeRecords();

    static CSACompressorFwdProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return CSACompressorFwdProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.kv))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.score))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ape))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.cu_seqlens_comp))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.compressed))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.ratio))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.head_dim))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.coff))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.total_comp))),
            std::get<CSACompressorFwdAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kCSACompressorFwd>::createPatternOperationId(
                    roles.csa_compressor_fwd)))};
    }
};

}  // namespace ftrain

#endif
