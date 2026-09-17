// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_GEMM_FAMILY_HPP_
#define FTRAIN_FAMILY_GEMM_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/family/gemm/problem.hpp"
#include "flash_train/pattern.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/primitive/gemm/fp32_gemm.hpp"
#include "flash_train/ops_args.hpp"
#include "flash_train/engine.hpp"

namespace ftrain {

// Everything the Gemm family contributes to its engine; OpsEngine consumes
// exactly these static functions (the OpsEngine comment carries the
// contract).
struct GemmFamily {
    using Problem = GemmProblem;

    // The supported Pattern together with the typed role IDs its builder
    // produced, built once by makeRoles() and shared by every engine of this
    // family. makeProblem() addresses ports through the saved IDs, so the
    // builder's numbering never has to be mirrored by hand.
    struct Roles {
        Pattern pattern;
        FTrainTensorId a;
        FTrainTensorId b;
        FTrainTensorId c;
        FTrainTensorId d;
        FTrainTensorId alpha;
        FTrainTensorId beta;
        FTrainGemmOpId gemm;
    };

    static const Roles& makeRoles() {
        static const Roles roles = [] {
            PatternBuilder pattern;
            const FTrainTensorId a     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId b     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId c     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId alpha = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId beta  = pattern.addOperand<OperandKind::kTensor>();
            const FTrainTensorId d     = pattern.addOperand<OperandKind::kTensor>();
            const FTrainGemmOpId gemm  = pattern.addOperation<OperationKind::kGemm>(a, b, c, d, alpha, beta);
            return Roles{pattern.buildPattern(), a, b, c, d, alpha, beta, gemm};
        }();
        return roles;
    }

    static Pattern makePattern() { return makeRoles().pattern; }

    static std::vector<std::shared_ptr<const Primitive<GemmProblem>>> makeRecords() {
        return {std::make_shared<Fp32Gemm>()};
    }

    static GemmProblem makeProblem(const Args& args) {
        const Roles& roles = makeRoles();
        return GemmProblem{
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.a))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.b))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.c))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.d))),
            std::get<Tensor>(
                *args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.alpha))),
            std::get<Tensor>(*args.getOperand(OperandTraits<OperandKind::kTensor>::createPatternOperandId(roles.beta))),
            std::get<GemmAttributes>(
                *args.getOpArgument(OperationTraits<OperationKind::kGemm>::createPatternOperationId(roles.gemm)))};
    }
};

// Creates the built-in Gemm engine: six Tensor operands (a, b, c, alpha,
// beta, d) and one Gemm operation. Allocation failure throws std::bad_alloc.
std::shared_ptr<OpsEngineBase> makeGemmOpsEngine();

}  // namespace ftrain

#endif
