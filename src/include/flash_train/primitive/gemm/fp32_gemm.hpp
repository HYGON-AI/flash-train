#ifndef FTRAIN_PRIMITIVE_GEMM_FP32_GEMM_HPP_
#define FTRAIN_PRIMITIVE_GEMM_FP32_GEMM_HPP_

#include <cstdint>
#include <memory>

#include "flash_train/common.h"
#include "flash_train/family/gemm/problem.hpp"
#include "flash_train/primitive/base.hpp"

namespace ftrain {

// Defined in fp32_gemm.hip; internal to the Gemm family. Enqueues
// d = alpha * (a @ b) + beta * c as one row-major contiguous FP32 GEMM;
// c may be null, which drops the beta * c term. alpha and beta are host
// scalars passed by value.
void launchFp32Gemm(const float* a, const float* b, const float* c, float alpha, float beta, float* d, std::uint64_t m,
                    std::uint64_t n, std::uint64_t k, FTrainStream stream);

// The FP32 Gemm Primitive: one row-major contiguous FP32 kernel requiring
// no workspace. Accepts only all-FP32, continuously indexed,
// float-aligned operands with rank-two a, b, c, and d in device memory
// (c's address may be null), host alpha and beta scalars, and an FP32
// compute type. In-place c/d aliasing is supported; any other overlap
// between ports is the caller's responsibility.
class Fp32Gemm final : public Primitive<GemmProblem> {
  public:
    const char* getName() const noexcept override;

    std::unique_ptr<PrimitiveBase> clone() const override;

    Result isApplicable(const GemmProblem& problem, const Constraints& constraints) const override;

    std::uint64_t getRequiredWorkspaceBytes() const noexcept override;

    void configure(const GemmProblem& problem) override;

  private:
    void executeImpl(const Resources& resources) override;

    const float* a_  = nullptr;
    const float* b_  = nullptr;
    const float* c_  = nullptr;
    float alpha_     = 0.0F;
    float beta_      = 0.0F;
    float* d_        = nullptr;
    std::uint64_t m_ = 0;
    std::uint64_t n_ = 0;
    std::uint64_t k_ = 0;
};

}  // namespace ftrain

#endif
