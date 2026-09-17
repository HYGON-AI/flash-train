#ifndef FTRAIN_PRIMITIVE_GEMM_FP32_GEMM_HPP_
#define FTRAIN_PRIMITIVE_GEMM_FP32_GEMM_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/ops/gemm/problem.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// Defined in fp32_gemm.hip; internal to the Gemm family. Enqueues
// d = alpha * (a @ b) + beta * c as one row-major contiguous FP32 GEMM.
void launchFp32Gemm(const float* a, const float* b, const float* c, const float* alpha, const float* beta, float* d,
                    std::uint64_t m, std::uint64_t n, std::uint64_t k, FTrainStream stream);

// The FP32 Gemm Primitive: one row-major contiguous FP32 kernel requiring
// no workspace. Accepts only all-FP32, continuously indexed, non-host,
// float-aligned device operands with rank-two a, b, c, and d, scalar alpha
// and beta, an FP32 compute type, and non-overlapping output and inputs.
class Fp32Gemm final : public Primitive<GemmProblem> {
  public:
    const char* getName() const noexcept override;

    std::unique_ptr<PrimitiveBase> clone() const override;

    Result isApplicable(const GemmProblem& problem, const SelectionContext& context) const override;

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;

    void configure(const GemmProblem& problem) override;

  private:
    void executeImpl(void* workspace, std::uint64_t workspace_bytes, FTrainStream stream) override;

    const float* a_     = nullptr;
    const float* b_     = nullptr;
    const float* c_     = nullptr;
    const float* alpha_ = nullptr;
    const float* beta_  = nullptr;
    float* d_           = nullptr;
    std::uint64_t m_    = 0;
    std::uint64_t n_    = 0;
    std::uint64_t k_    = 0;
};

}  // namespace ftrain

#endif
