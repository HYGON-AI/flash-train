// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "flash_train/primitive/hygon/kda/g2_kda.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kHeadDim  = 128;
constexpr std::int64_t kMaxElems = (std::int64_t{1} << 31) - 1;

// The shared value checklist: architecture, head dims, size limits, layouts, alignment and the
// numeric types every call has. type_of(port) == want for each (port, want) pair.
template<typename Problem>
Result checkCommon(const Problem& problem, const Constraints& constraints,
                   std::initializer_list<std::pair<std::size_t, FTrainNumericType>> types) {
    if (!kda_g2::deviceSupported(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA kernels target gfx936 only");
    }
    const KdaShape& shape = problem.getShape();
    if (shape.key_dim != kHeadDim || shape.value_dim != kHeadDim) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA supports K = V = 128 only");
    }
    if (shape.tokens * shape.heads * kHeadDim > kMaxElems) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA requires T * H * 128 < 2^31");
    }
    const KdaAttributes& attributes = problem.getAttributes();
    if (attributes.isGateInKernel() && !(attributes.getLowerBound() >= -5.0F && attributes.getLowerBound() < 0.0F)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA gate_in_kernel requires lower_bound in [-5, 0)");
    }
    for (std::size_t i = 0; i < problem.size(); ++i) {
        if (!problem.hasMemory(i)) { continue; }
        if (problem.getIndexType(i) != FTRAIN_INDEX_TYPE_CONTINUOUS) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA requires continuously indexed ports");
        }
        if (reinterpret_cast<std::uintptr_t>(problem.getAddress(i)) % 16 != 0) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA requires 16-byte aligned ports");
        }
    }
    for (const auto& [port, want] : types) {
        if (problem.hasMemory(port) && problem.getNumericType(port) != want) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 KDA port has an unsupported numeric type");
        }
    }
    return Result{};
}

FTrainNumericType gateType(const KdaAttributes& attributes) {
    return attributes.isGateInKernel() ? FTRAIN_NUMERIC_TYPE_BF16 : FTRAIN_NUMERIC_TYPE_FP32;
}

// The state port's three regions (bf16 elements per chunk-head): bN(h) 16384 | v_new native
// 8192 | v_new A-natural 8192. Null base -> all null (the state is not exchanged).
template<typename Ptr>
void splitState(Ptr base, const kda_g2::Shape& s, Ptr& hbn, Ptr& vn, Ptr& anv) {
    using Byte    = std::conditional_t<std::is_const_v<std::remove_pointer_t<Ptr>>, const char, char>;
    const auto ch = static_cast<std::uint64_t>(s.chunks()) * static_cast<std::uint64_t>(s.heads);
    Byte* b       = static_cast<Byte*>(base);
    hbn           = base == nullptr ? nullptr : static_cast<Ptr>(b);
    vn            = base == nullptr ? nullptr : static_cast<Ptr>(b + ch * 16384 * 2);
    anv           = base == nullptr ? nullptr : static_cast<Ptr>(b + ch * (16384 + 8192) * 2);
}

kda_g2::Shape makeShape(const KdaShape& shape, const KdaAttributes& attributes, bool has_cu) {
    kda_g2::Shape s;
    s.tokens         = shape.tokens;
    s.heads          = shape.heads;
    s.seqs           = shape.seqs;
    s.has_cu_seqlens = has_cu;
    s.l2norm_qk      = attributes.isL2NormQk();
    s.gate_in_kernel = attributes.isGateInKernel();
    return s;
}

}  // namespace

// ---------------------------------------------------------------- forward

const char* G2KdaForward::getName() const noexcept { return "G2KdaForward"; }

std::unique_ptr<PrimitiveBase> G2KdaForward::clone() const { return std::make_unique<G2KdaForward>(*this); }

Result G2KdaForward::isApplicable(const KdaForwardProblem& problem, const Constraints& constraints) const {
    using P                      = KdaForwardProblem;
    const FTrainNumericType bf16 = FTRAIN_NUMERIC_TYPE_BF16, fp32 = FTRAIN_NUMERIC_TYPE_FP32;
    return checkCommon(problem, constraints,
                       {
                           {P::kQ,         bf16                             },
                           {P::kK,         bf16                             },
                           {P::kV,         bf16                             },
                           {P::kG,         gateType(problem.getAttributes())},
                           {P::kBeta,      fp32                             },
                           {P::kALog,      fp32                             },
                           {P::kDtBias,    fp32                             },
                           {P::kCuSeqlens, FTRAIN_NUMERIC_TYPE_INT32        },
                           {P::kO,         bf16                             },
                           {P::kQRstd,     fp32                             },
                           {P::kKRstd,     fp32                             },
                           {P::kAqk,       bf16                             },
                           {P::kAkk,       bf16                             },
                           {P::kQn,        bf16                             },
                           {P::kKn,        bf16                             },
                           {P::kH,         bf16                             },
                           {P::kState,     bf16                             }
    });
}

std::uint64_t G2KdaForward::getRequiredWorkspaceBytes() const noexcept { return kda_g2::forwardWorkspaceBytes(shape_); }

void G2KdaForward::configure(const KdaForwardProblem& problem) {
    using P                         = KdaForwardProblem;
    const KdaAttributes& attributes = problem.getAttributes();
    shape_                          = makeShape(problem.getShape(), attributes, problem.hasMemory(P::kCuSeqlens));
    args_.q                         = problem.getAddress(P::kQ);
    args_.k                         = problem.getAddress(P::kK);
    args_.v                         = problem.getAddress(P::kV);
    args_.g                         = problem.getAddress(P::kG);
    args_.beta                      = problem.getAddress(P::kBeta);
    args_.a_log                     = problem.getAddress(P::kALog);
    args_.dt_bias                   = problem.getAddress(P::kDtBias);
    args_.cu_seqlens                = static_cast<const int*>(problem.getAddress(P::kCuSeqlens));
    args_.o                         = problem.getAddress(P::kO);
    args_.q_rstd                    = problem.getAddress(P::kQRstd);
    args_.k_rstd                    = problem.getAddress(P::kKRstd);
    args_.aqk                       = problem.getAddress(P::kAqk);
    args_.akk                       = problem.getAddress(P::kAkk);
    args_.qn                        = problem.getAddress(P::kQn);
    args_.kn                        = problem.getAddress(P::kKn);
    args_.saved_h                   = problem.getAddress(P::kH);
    splitState<void*>(problem.getAddress(P::kState), shape_, args_.state_hbn, args_.state_vn, args_.state_anv);
    args_.scale       = attributes.getScale();
    args_.lower_bound = attributes.getLowerBound();
}

void G2KdaForward::executeImpl(const Resources& resources) {
    kda_g2::forward(shape_, args_, resources.getWorkspace(), resources.getStream());
}

// ---------------------------------------------------------------- backward

const char* G2KdaBackward::getName() const noexcept { return "G2KdaBackward"; }

std::unique_ptr<PrimitiveBase> G2KdaBackward::clone() const { return std::make_unique<G2KdaBackward>(*this); }

Result G2KdaBackward::isApplicable(const KdaBackwardProblem& problem, const Constraints& constraints) const {
    using P                      = KdaBackwardProblem;
    const FTrainNumericType bf16 = FTRAIN_NUMERIC_TYPE_BF16, fp32 = FTRAIN_NUMERIC_TYPE_FP32;
    const FTrainNumericType gt = gateType(problem.getAttributes());
    return checkCommon(problem, constraints,
                       {
                           {P::kDO,        bf16                     },
                           {P::kQ,         bf16                     },
                           {P::kK,         bf16                     },
                           {P::kV,         bf16                     },
                           {P::kG,         gt                       },
                           {P::kBeta,      fp32                     },
                           {P::kALog,      fp32                     },
                           {P::kDtBias,    fp32                     },
                           {P::kCuSeqlens, FTRAIN_NUMERIC_TYPE_INT32},
                           {P::kQRstd,     fp32                     },
                           {P::kKRstd,     fp32                     },
                           {P::kAqk,       bf16                     },
                           {P::kAkk,       bf16                     },
                           {P::kQn,        bf16                     },
                           {P::kKn,        bf16                     },
                           {P::kDq,        bf16                     },
                           {P::kDk,        bf16                     },
                           {P::kDv,        bf16                     },
                           {P::kDg,        gt                       },
                           {P::kDBeta,     fp32                     },
                           {P::kDALog,     fp32                     },
                           {P::kDDtBias,   fp32                     },
                           {P::kH,         bf16                     },
                           {P::kState,     bf16                     }
    });
}

std::uint64_t G2KdaBackward::getRequiredWorkspaceBytes() const noexcept {
    return kda_g2::backwardWorkspaceBytes(shape_);
}

void G2KdaBackward::configure(const KdaBackwardProblem& problem) {
    using P                         = KdaBackwardProblem;
    const KdaAttributes& attributes = problem.getAttributes();
    shape_                          = makeShape(problem.getShape(), attributes, problem.hasMemory(P::kCuSeqlens));
    shape_.has_state                = problem.hasMemory(P::kState);
    shape_.has_saved_h              = problem.hasMemory(P::kH);
    // dq | dk as one buffer (dk right after dq) host the gate scratch when qn / kn need no staging.
    const auto row_bytes = static_cast<std::uintptr_t>(problem.getShape().tokens * problem.getShape().heads * 128 * 2);
    shape_.g_in_dqdk     = problem.hasMemory(P::kQn) && problem.hasMemory(P::kDq) && problem.hasMemory(P::kDk) &&
                           reinterpret_cast<std::uintptr_t>(problem.getAddress(P::kDk)) ==
                               reinterpret_cast<std::uintptr_t>(problem.getAddress(P::kDq)) + row_bytes;
    args_.d_o            = problem.getAddress(P::kDO);
    args_.q              = problem.getAddress(P::kQ);
    args_.k              = problem.getAddress(P::kK);
    args_.v              = problem.getAddress(P::kV);
    args_.g              = problem.getAddress(P::kG);
    args_.beta           = problem.getAddress(P::kBeta);
    args_.a_log          = problem.getAddress(P::kALog);
    args_.dt_bias        = problem.getAddress(P::kDtBias);
    args_.cu_seqlens     = static_cast<const int*>(problem.getAddress(P::kCuSeqlens));
    args_.q_rstd         = problem.getAddress(P::kQRstd);
    args_.k_rstd         = problem.getAddress(P::kKRstd);
    args_.aqk            = problem.getAddress(P::kAqk);
    args_.akk            = problem.getAddress(P::kAkk);
    args_.qn             = problem.getAddress(P::kQn);
    args_.kn             = problem.getAddress(P::kKn);
    args_.dq             = problem.getAddress(P::kDq);
    args_.dk             = problem.getAddress(P::kDk);
    args_.dv             = problem.getAddress(P::kDv);
    args_.dg             = problem.getAddress(P::kDg);
    args_.dbeta          = problem.getAddress(P::kDBeta);
    args_.da_log         = problem.getAddress(P::kDALog);
    args_.ddt_bias       = problem.getAddress(P::kDDtBias);
    args_.saved_h        = problem.getAddress(P::kH);
    splitState<const void*>(problem.getAddress(P::kState), shape_, args_.state_hbn, args_.state_vn, args_.state_anv);
    args_.scale       = attributes.getScale();
    args_.lower_bound = attributes.getLowerBound();
}

void G2KdaBackward::executeImpl(const Resources& resources) {
    kda_g2::backward(shape_, args_, resources.getWorkspace(), resources.getStream());
}

}  // namespace ftrain
