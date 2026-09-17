#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "flash_train/common.h"
#include "flash_train/family/gemm/problem.hpp"
#include "flash_train/primitive/gemm/fp32_gemm.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kDims2x2[2]     = {2, 2};
constexpr std::int64_t kDims1x0[2]     = {1, 0};
constexpr std::int64_t kDims0x2[2]     = {0, 2};
constexpr std::int64_t kDims1x2[2]     = {1, 2};
constexpr FTrainNumericType kOtherType = FTRAIN_NUMERIC_TYPE_FP16;

struct ProblemPorts {
    void* a                        = kDefaultAMemory;
    void* b                        = kDefaultBMemory;
    void* c                        = kDefaultCMemory;
    void* d                        = kDefaultDMemory;
    void* alpha                    = &kDefaultScalar;
    void* beta                     = &kDefaultScalar;
    const std::int64_t* a_dims     = kDims2x2;
    const std::int64_t* b_dims     = kDims2x2;
    const std::int64_t* cd_dims    = kDims2x2;
    FTrainNumericType a_type       = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32;
    FTrainIndexType a_index        = FTRAIN_INDEX_TYPE_CONTINUOUS;

    static alignas(float) unsigned char kDefaultAMemory[8];
    static alignas(float) unsigned char kDefaultBMemory[8];
    static alignas(float) unsigned char kDefaultCMemory[16];
    static alignas(float) unsigned char kDefaultDMemory[16];
    static float kDefaultScalar;
};

alignas(float) unsigned char ProblemPorts::kDefaultAMemory[8];
alignas(float) unsigned char ProblemPorts::kDefaultBMemory[8];
alignas(float) unsigned char ProblemPorts::kDefaultCMemory[16];
alignas(float) unsigned char ProblemPorts::kDefaultDMemory[16];
float ProblemPorts::kDefaultScalar = 1.0F;

Tensor makeDeviceTensor(void* memory, const std::int64_t* dims, FTrainNumericType numeric_type,
                        FTrainIndexType index_type) {
    FTrainStorageView view{};
    view.memory         = memory;
    view.dims           = dims;
    view.strides        = nullptr;
    view.num_dims       = 2;
    view.numeric_type   = numeric_type;
    view.index_type     = index_type;
    view.is_host_memory = false;
    return Tensor{StorageView{view}};
}

Tensor makeHostScalar(void* memory) {
    FTrainStorageView view{};
    view.memory         = memory;
    view.dims           = nullptr;
    view.strides        = nullptr;
    view.num_dims       = 0;
    view.numeric_type   = FTRAIN_NUMERIC_TYPE_FP32;
    view.index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS;
    view.is_host_memory = true;
    return Tensor{StorageView{view}};
}

GemmProblem makeProblem(const ProblemPorts& ports) {
    return GemmProblem{makeDeviceTensor(ports.a, ports.a_dims, ports.a_type, ports.a_index),
                       makeDeviceTensor(ports.b, ports.b_dims, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS),
                       makeDeviceTensor(ports.c, ports.cd_dims, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS),
                       makeDeviceTensor(ports.d, ports.cd_dims, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS),
                       makeHostScalar(ports.alpha),
                       makeHostScalar(ports.beta),
                       GemmAttributes{ports.compute_type}};
}

const Constraints kConstraints{0};

}  // namespace

TEST(Fp32GemmTest, AcceptsAnAllFp32ContinuousAlignedProblem) {
    Fp32Gemm primitive;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(ProblemPorts{}), kConstraints).isSuccess());
    EXPECT_STREQ(primitive.getName(), "Fp32Gemm");
    EXPECT_EQ(primitive.getRequiredWorkspaceBytes(), 0U);
}

TEST(Fp32GemmTest, RejectsNonFp32ComputeAndElementTypes) {
    Fp32Gemm primitive;

    ProblemPorts non_fp32_compute;
    non_fp32_compute.compute_type = kOtherType;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(non_fp32_compute), kConstraints).isSuccess());

    ProblemPorts non_fp32_a;
    non_fp32_a.a_type = kOtherType;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(non_fp32_a), kConstraints).isSuccess());
}

TEST(Fp32GemmTest, RejectsNonContinuousIndexTypes) {
    Fp32Gemm primitive;

    ProblemPorts strided_a;
    strided_a.a_index = FTRAIN_INDEX_TYPE_NVIDIA_SF_128X4_SWIZZLED;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(strided_a), kConstraints).isSuccess());
}

TEST(Fp32GemmTest, RejectsMisalignedAddressesButTreatsNullCAsNoAccumulation) {
    Fp32Gemm primitive;
    static alignas(float) unsigned char misaligned[8];

    ProblemPorts misaligned_a;
    misaligned_a.a = misaligned + 1;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(misaligned_a), kConstraints).isSuccess());

    // A null c drops the beta * c term instead of rejecting.
    ProblemPorts null_c;
    null_c.c = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(null_c), kConstraints).isSuccess());
}

TEST(Fp32GemmTest, RequiresOutputAndNonEmptyReductionAddresses) {
    Fp32Gemm primitive;

    ProblemPorts null_d;
    null_d.d = nullptr;
    EXPECT_FALSE(primitive.isApplicable(makeProblem(null_d), kConstraints).isSuccess());

    // k == 0 skips the a and b address requirement.
    ProblemPorts empty_reduction;
    empty_reduction.a_dims  = kDims1x0;
    empty_reduction.b_dims  = kDims0x2;
    empty_reduction.cd_dims = kDims1x2;
    empty_reduction.a       = nullptr;
    empty_reduction.b       = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(empty_reduction), kConstraints).isSuccess());
}

TEST(Fp32GemmTest, ScalarOperandsCanNeverCarryNullMemory) {
    // A scalar StorageView with null memory is rejected at construction, so
    // no GemmProblem -- and no isApplicable check -- ever sees null alpha or
    // beta addresses.
    ProblemPorts null_alpha;
    null_alpha.alpha = nullptr;
    try {
        static_cast<void>(makeProblem(null_alpha));
        FAIL() << "a null scalar memory was accepted";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(Fp32GemmTest, AcceptsEveryEmptyOutputProblemWithoutMatrixAddresses) {
    Fp32Gemm primitive;

    // m == 0: no output elements, so no matrix address is required; the
    // host scalars always carry memory.
    ProblemPorts empty_output;
    empty_output.a_dims  = kDims0x2;
    empty_output.cd_dims = kDims0x2;
    empty_output.a       = nullptr;
    empty_output.b       = nullptr;
    empty_output.c       = nullptr;
    empty_output.d       = nullptr;
    EXPECT_TRUE(primitive.isApplicable(makeProblem(empty_output), kConstraints).isSuccess());
}

TEST(Fp32GemmTest, ClonesIntoIndependentConfigurableCopies) {
    Fp32Gemm prototype;

    const std::unique_ptr<PrimitiveBase> first  = prototype.clone();
    const std::unique_ptr<PrimitiveBase> second = prototype.clone();
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first.get(), second.get());
    EXPECT_STREQ(first->getName(), "Fp32Gemm");
    EXPECT_EQ(first->getRequiredWorkspaceBytes(), 0U);

    // Configuring one clone leaves the others configurable; the numeric
    // effect of configure() is exercised end-to-end by the API tests.
    static_cast<Primitive<GemmProblem>&>(*first).configure(makeProblem(ProblemPorts{}));
    static_cast<Primitive<GemmProblem>&>(*second).configure(makeProblem(ProblemPorts{}));
    const GemmProblem problem = makeProblem(ProblemPorts{});
    EXPECT_TRUE(static_cast<const Primitive<GemmProblem>&>(*first).isApplicable(problem, kConstraints).isSuccess());
    EXPECT_TRUE(static_cast<const Primitive<GemmProblem>&>(*second).isApplicable(problem, kConstraints).isSuccess());
}

}  // namespace ftrain
