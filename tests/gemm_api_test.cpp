#include "flash_train/pattern.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "flash_train/common.h"

namespace {

class GemmApiTest : public testing::Test {
  protected:
    void SetUp() override {
        ASSERT_EQ(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream_, nullptr);

        ASSERT_EQ(ftrainPatternCreate(&pattern_), FTRAIN_STATUS_SUCCESS);

        // Deliberately differs from the built-in PatternBuilder's operand-add order.
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &d_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &beta_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &alpha_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &c_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &b_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddTensor(pattern_, &a_id_), FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainPatternAddGemm(pattern_, &gemm_id_, a_id_, b_id_, c_id_, d_id_, alpha_id_, beta_id_),
                  FTRAIN_STATUS_SUCCESS);
        ASSERT_EQ(ftrainOpsCreate(&ops_, pattern_), FTRAIN_STATUS_SUCCESS);

        ASSERT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS);
        pattern_ = nullptr;
    }

    void TearDown() override {
        if (stream_ != nullptr) { EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess); }

        for (FTrainPlan plan : plans_) {
            if (plan != nullptr) { EXPECT_EQ(ftrainPlanDestroy(plan), FTRAIN_STATUS_SUCCESS); }
        }
        for (FTrainArgs args : args_) {
            if (args != nullptr) { EXPECT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS); }
        }
        if (ops_ != nullptr) { EXPECT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS); }
        if (pattern_ != nullptr) { EXPECT_EQ(ftrainPatternDestroy(pattern_), FTRAIN_STATUS_SUCCESS); }

        for (void* allocation : allocations_) { EXPECT_EQ(hipFree(allocation), hipSuccess); }
        if (stream_ != nullptr) { EXPECT_EQ(hipStreamDestroy(stream_), hipSuccess); }
    }

    template<typename T>
    T* allocate(std::size_t count) {
        void* allocation    = nullptr;
        const hipError_t rc = hipMalloc(&allocation, count * sizeof(T));
        if (rc != hipSuccess) {
            ADD_FAILURE() << "hipMalloc failed: " << hipGetErrorString(rc);
            return nullptr;
        }
        allocations_.push_back(allocation);
        return static_cast<T*>(allocation);
    }

    FTrainArgs createArgs() {
        FTrainArgs args       = nullptr;
        const FTrainStatus rc = ftrainArgsCreate(&args, ops_);
        if (rc != FTRAIN_STATUS_SUCCESS) {
            ADD_FAILURE() << "ftrainArgsCreate failed with status " << static_cast<unsigned int>(rc);
            return nullptr;
        }
        args_.push_back(args);
        return args;
    }

    void trackPlan(FTrainPlan plan) { plans_.push_back(plan); }

    void forgetArgs(FTrainArgs args) {
        for (FTrainArgs& tracked : args_) {
            if (tracked == args) {
                tracked = nullptr;
                return;
            }
        }
    }

    static FTrainStorageView makeView(void* memory, const std::int64_t* dims, std::uint8_t num_dims,
                                      FTrainNumericType numeric_type = FTRAIN_NUMERIC_TYPE_FP32,
                                      FTrainIndexType index_type     = FTRAIN_INDEX_TYPE_CONTINUOUS,
                                      bool is_host_memory = false, const std::int64_t* strides = nullptr) {
        return FTrainStorageView{memory, dims, strides, num_dims, numeric_type, index_type, is_host_memory};
    }

    bool setArgs(FTrainArgs args, FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                 FTrainStorageView alpha, FTrainStorageView beta,
                 FTrainNumericType compute_type = FTRAIN_NUMERIC_TYPE_FP32) {
        const FTrainStatus a_status     = ftrainArgsSetTensor(args, a_id_, a);
        const FTrainStatus b_status     = ftrainArgsSetTensor(args, b_id_, b);
        const FTrainStatus c_status     = ftrainArgsSetTensor(args, c_id_, c);
        const FTrainStatus d_status     = ftrainArgsSetTensor(args, d_id_, d);
        const FTrainStatus alpha_status = ftrainArgsSetTensor(args, alpha_id_, alpha);
        const FTrainStatus beta_status  = ftrainArgsSetTensor(args, beta_id_, beta);
        const FTrainStatus op_status    = ftrainArgsSetGemm(args, gemm_id_, compute_type);
        EXPECT_EQ(a_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(b_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(c_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(d_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(alpha_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(beta_status, FTRAIN_STATUS_SUCCESS);
        EXPECT_EQ(op_status, FTRAIN_STATUS_SUCCESS);
        return a_status == FTRAIN_STATUS_SUCCESS && b_status == FTRAIN_STATUS_SUCCESS &&
               c_status == FTRAIN_STATUS_SUCCESS && d_status == FTRAIN_STATUS_SUCCESS &&
               alpha_status == FTRAIN_STATUS_SUCCESS && beta_status == FTRAIN_STATUS_SUCCESS &&
               op_status == FTRAIN_STATUS_SUCCESS;
    }

    void expectPlanStatus(FTrainArgs args, FTrainStatus expected_status) {
        FTrainPlan plan = nullptr;
        EXPECT_EQ(ftrainPlanCreate(&plan, args), expected_status);
        if (plan != nullptr) { trackPlan(plan); }
    }

    FTrainPattern pattern_{nullptr};
    FTrainOps ops_{nullptr};
    FTrainTensorId a_id_{};
    FTrainTensorId b_id_{};
    FTrainTensorId c_id_{};
    FTrainTensorId d_id_{};
    FTrainTensorId alpha_id_{};
    FTrainTensorId beta_id_{};
    FTrainGemmOpId gemm_id_{};
    hipStream_t stream_{nullptr};
    std::vector<FTrainArgs> args_;
    std::vector<FTrainPlan> plans_;
    std::vector<void*> allocations_;
};

TEST_F(GemmApiTest, RebindsAddressesAndOutlivesOpsAndArgs) {
    constexpr std::int64_t kM = 3;
    constexpr std::int64_t kN = 4;
    constexpr std::int64_t kK = 2;

    const std::array<float, kM * kK> first_a{1.0F, 2.0F, -1.0F, 0.5F, 3.0F, -2.0F};
    const std::array<float, kK * kN> first_b{1.0F, -1.0F, 2.0F, 0.0F, 0.5F, 1.0F, -2.0F, 3.0F};
    const std::array<float, kM * kN> first_c{1.0F, 1.0F, 1.0F,  1.0F,  0.0F,  0.0F,
                                             0.0F, 0.0F, -1.0F, -1.0F, -1.0F, -1.0F};
    constexpr float kFirstAlpha = 2.0F;
    constexpr float kFirstBeta  = -1.0F;

    const std::array<float, kM * kK> second_a{-1.0F, 1.0F, 2.0F, 2.0F, 0.0F, 4.0F};
    const std::array<float, kK * kN> second_b{0.5F, 0.5F, 0.5F, 0.5F, 1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, kM * kN> second_c{2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 2.0F};
    constexpr float kSecondAlpha = 1.0F;
    constexpr float kSecondBeta  = 0.5F;

    auto* first_device_a  = allocate<float>(kM * kK);
    auto* first_device_b  = allocate<float>(kK * kN);
    auto* first_device_c  = allocate<float>(kM * kN);
    auto* first_device_d  = allocate<float>(kM * kN);
    auto* second_device_a = allocate<float>(kM * kK);
    auto* second_device_b = allocate<float>(kK * kN);
    auto* second_device_c = allocate<float>(kM * kN);
    auto* second_device_d = allocate<float>(kM * kN);
    float first_alpha     = kFirstAlpha;
    float first_beta      = kFirstBeta;
    float second_alpha    = kSecondAlpha;
    float second_beta     = kSecondBeta;
    ASSERT_NE(first_device_a, nullptr);
    ASSERT_NE(first_device_b, nullptr);
    ASSERT_NE(first_device_c, nullptr);
    ASSERT_NE(first_device_d, nullptr);
    ASSERT_NE(second_device_a, nullptr);
    ASSERT_NE(second_device_b, nullptr);
    ASSERT_NE(second_device_c, nullptr);
    ASSERT_NE(second_device_d, nullptr);

    ASSERT_EQ(hipMemcpyAsync(first_device_a, first_a.data(), sizeof(first_a), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_b, first_b.data(), sizeof(first_b), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(first_device_c, first_c.data(), sizeof(first_c), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_a, second_a.data(), sizeof(second_a), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_b, second_b.data(), sizeof(second_b), hipMemcpyHostToDevice, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_device_c, second_c.data(), sizeof(second_c), hipMemcpyHostToDevice, stream_),
              hipSuccess);

    const std::int64_t a_dims[]{kM, kK};
    const std::int64_t b_dims[]{kK, kN};
    const std::int64_t output_dims[]{kM, kN};
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(first_device_a, a_dims, 2), makeView(first_device_b, b_dims, 2),
                makeView(first_device_c, output_dims, 2), makeView(first_device_d, output_dims, 2),
                makeView(&first_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&first_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));

    FTrainPlan first_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&first_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(first_plan, nullptr);
    trackPlan(first_plan);

    std::uint64_t workspace_bytes = 1;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(first_plan, 0, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_bytes, 0U);

    ASSERT_TRUE(
        setArgs(args, makeView(second_device_a, a_dims, 2), makeView(second_device_b, b_dims, 2),
                makeView(second_device_c, output_dims, 2), makeView(second_device_d, output_dims, 2),
                makeView(&second_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&second_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));

    FTrainPlan second_plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&second_plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(second_plan, nullptr);
    trackPlan(second_plan);
    workspace_bytes = 1;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(second_plan, 0, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_bytes, 0U);

    ASSERT_EQ(ftrainArgsDestroy(args), FTRAIN_STATUS_SUCCESS);
    forgetArgs(args);
    args = nullptr;
    ASSERT_EQ(ftrainOpsDestroy(ops_), FTRAIN_STATUS_SUCCESS);
    ops_ = nullptr;

    ASSERT_EQ(ftrainPlanExecutePrimitive(first_plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(ftrainPlanExecutePrimitive(second_plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);

    std::array<float, kM * kN> first_d{};
    std::array<float, kM * kN> second_d{};
    ASSERT_EQ(hipMemcpyAsync(first_d.data(), first_device_d, sizeof(first_d), hipMemcpyDeviceToHost, stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(second_d.data(), second_device_d, sizeof(second_d), hipMemcpyDeviceToHost, stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    for (std::int64_t row = 0; row < kM; ++row) {
        for (std::int64_t column = 0; column < kN; ++column) {
            float first_product  = 0.0F;
            float second_product = 0.0F;
            for (std::int64_t depth = 0; depth < kK; ++depth) {
                first_product  += first_a[row * kK + depth] * first_b[depth * kN + column];
                second_product += second_a[row * kK + depth] * second_b[depth * kN + column];
            }
            const std::size_t index = static_cast<std::size_t>(row * kN + column);
            EXPECT_FLOAT_EQ(first_d[index], kFirstAlpha * first_product + kFirstBeta * first_c[index]);
            EXPECT_FLOAT_EQ(second_d[index], kSecondAlpha * second_product + kSecondBeta * second_c[index]);
        }
    }
}

TEST_F(GemmApiTest, SupportsEmptyOutputWithoutMatrixStorageOrKernelLaunch) {
    float host_alpha = 1.0F;
    float host_beta  = 0.0F;

    const std::int64_t a_dims[]{0, 2};
    const std::int64_t b_dims[]{2, 4};
    const std::int64_t output_dims[]{0, 4};
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(nullptr, a_dims, 2), makeView(nullptr, b_dims, 2), makeView(nullptr, output_dims, 2),
                makeView(nullptr, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));

    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);

    std::uint64_t workspace_bytes = 1;
    ASSERT_EQ(ftrainPlanGetPrimitiveRequiredWorkspaceBytes(plan, 0, &workspace_bytes), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(workspace_bytes, 0U);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    EXPECT_EQ(hipStreamSynchronize(stream_), hipSuccess);
}

TEST_F(GemmApiTest, ReportsMatrixSizeOverflowBeforeSelection) {
    float host_alpha = 1.0F;
    float host_beta  = 0.0F;

    constexpr std::int64_t kMaxDimension = std::numeric_limits<std::int64_t>::max();
    const std::int64_t a_dims[]{kMaxDimension, kMaxDimension};
    const std::int64_t b_dims[]{kMaxDimension, kMaxDimension};
    const std::int64_t output_dims[]{kMaxDimension, kMaxDimension};
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(nullptr, a_dims, 2), makeView(nullptr, b_dims, 2), makeView(nullptr, output_dims, 2),
                makeView(nullptr, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));

    expectPlanStatus(args, FTRAIN_STATUS_OVERFLOW);
}

TEST_F(GemmApiTest, RejectsInconsistentRanksAndShapes) {
    auto* device_a   = allocate<float>(12);
    auto* device_b   = allocate<float>(12);
    auto* device_c   = allocate<float>(12);
    auto* device_d   = allocate<float>(12);
    float host_alpha = 1.0F;
    float host_beta  = 0.0F;
    ASSERT_NE(device_a, nullptr);
    ASSERT_NE(device_b, nullptr);
    ASSERT_NE(device_c, nullptr);
    ASSERT_NE(device_d, nullptr);

    const std::int64_t a_dims[]{3, 2};
    const std::int64_t b_dims[]{2, 4};
    const std::int64_t output_dims[]{3, 4};
    const std::int64_t mismatched_b_dims[]{3, 4};
    const std::int64_t wrong_output_dims[]{4, 3};
    const std::int64_t rank_one_dims[]{6};
    const std::int64_t scalar_like_dims[]{1};

    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, rank_one_dims, 1), makeView(device_b, b_dims, 2),
                makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, mismatched_b_dims, 2),
                makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2),
                makeView(device_d, wrong_output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2),
        makeView(device_d, output_dims, 2),
        makeView(&host_alpha, scalar_like_dims, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2),
                makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    EXPECT_EQ(ftrainArgsSetGemm(args, gemm_id_, FTRAIN_NUMERIC_TYPE_COUNT), FTRAIN_STATUS_INVALID_ARGUMENT);
}

TEST_F(GemmApiTest, RejectsUnsupportedStorageProperties) {
    auto* device_a   = allocate<float>(12);
    auto* device_b   = allocate<float>(12);
    auto* device_c   = allocate<float>(12);
    auto* device_d   = allocate<float>(12);
    float host_alpha = 1.0F;
    float host_beta  = 0.0F;
    ASSERT_NE(device_a, nullptr);
    ASSERT_NE(device_b, nullptr);
    ASSERT_NE(device_c, nullptr);
    ASSERT_NE(device_d, nullptr);

    const std::int64_t a_dims[]{3, 2};
    const std::int64_t b_dims[]{2, 4};
    const std::int64_t output_dims[]{3, 4};

    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2, FTRAIN_NUMERIC_TYPE_FP16), makeView(device_b, b_dims, 2),
                makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_UNSUPPORTED);
    EXPECT_NE(strstr(ftrainGetLastMessage(), "Fp32Gemm"), nullptr);
    EXPECT_NE(strstr(ftrainGetLastMessage(), "an operand is not fp32"), nullptr);

    const std::int64_t a_strides[]{2, 1};
    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(
        args, makeView(device_a, a_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID, false, a_strides),
        makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
        makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
        makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_UNSUPPORTED);

    // a in host memory and alpha in device memory are structural violations
    // the GemmProblem constructor rejects.
    std::array<float, 12> host_a{};
    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(host_a.data(), a_dims, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    auto* device_scalar = allocate<float>(1);
    ASSERT_NE(device_scalar, nullptr);
    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2),
                makeView(device_d, output_dims, 2), makeView(device_scalar, nullptr, 0),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_INVALID_ARGUMENT);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(nullptr, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_c, output_dims, 2),
                makeView(device_d, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));
    expectPlanStatus(args, FTRAIN_STATUS_UNSUPPORTED);

    args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2),
                        makeView(device_c, output_dims, 2), makeView(device_d, output_dims, 2),
                        makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                        FTRAIN_NUMERIC_TYPE_FP64));
    expectPlanStatus(args, FTRAIN_STATUS_UNSUPPORTED);
}

TEST_F(GemmApiTest, AccumulatesInPlaceWhenCAndDShareOneAddress) {
    constexpr std::int64_t kM   = 3;
    constexpr std::int64_t kN   = 4;
    constexpr std::int64_t kK   = 2;
    constexpr float kAlpha      = 2.0F;
    constexpr float kBeta       = 3.0F;
    constexpr std::int64_t kMxK = kM * kK;
    constexpr std::int64_t kKxN = kK * kN;
    constexpr std::int64_t kMxN = kM * kN;

    const std::array<float, kMxK> a{1.0F, 2.0F, -1.0F, 0.5F, 3.0F, -2.0F};
    const std::array<float, kKxN> b{1.0F, -1.0F, 2.0F, 0.0F, 0.5F, 1.0F, -2.0F, 3.0F};
    const std::array<float, kMxN> initial_cd{1.0F,  -2.0F, 3.0F, 0.5F,  -1.5F, 2.5F,
                                             -3.0F, 1.0F,  0.0F, -4.0F, 2.0F,  1.5F};

    auto* device_a   = allocate<float>(kMxK);
    auto* device_b   = allocate<float>(kKxN);
    auto* device_cd  = allocate<float>(kMxN);
    float host_alpha = kAlpha;
    float host_beta  = kBeta;
    ASSERT_NE(device_a, nullptr);
    ASSERT_NE(device_b, nullptr);
    ASSERT_NE(device_cd, nullptr);
    ASSERT_EQ(hipMemcpyAsync(device_a, a.data(), sizeof(a), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_b, b.data(), sizeof(b), hipMemcpyHostToDevice, stream_), hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(device_cd, initial_cd.data(), sizeof(initial_cd), hipMemcpyHostToDevice, stream_),
              hipSuccess);

    const std::int64_t a_dims[]{kM, kK};
    const std::int64_t b_dims[]{kK, kN};
    const std::int64_t output_dims[]{kM, kN};
    // c and d name the same buffer: d = alpha * (a @ b) + beta * c in place.
    FTrainArgs args = createArgs();
    ASSERT_NE(args, nullptr);
    ASSERT_TRUE(
        setArgs(args, makeView(device_a, a_dims, 2), makeView(device_b, b_dims, 2), makeView(device_cd, output_dims, 2),
                makeView(device_cd, output_dims, 2),
                makeView(&host_alpha, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true),
                makeView(&host_beta, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS, true)));

    FTrainPlan plan = nullptr;
    ASSERT_EQ(ftrainPlanCreate(&plan, args), FTRAIN_STATUS_SUCCESS);
    ASSERT_NE(plan, nullptr);
    trackPlan(plan);
    EXPECT_EQ(ftrainPlanExecutePrimitive(plan, 0, nullptr, 0, stream_), FTRAIN_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);

    std::array<float, kMxN> result{};
    ASSERT_EQ(hipMemcpyAsync(result.data(), device_cd, sizeof(result), hipMemcpyDeviceToHost, stream_), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
    for (std::int64_t row = 0; row < kM; ++row) {
        for (std::int64_t column = 0; column < kN; ++column) {
            float accumulated = 0.0F;
            for (std::int64_t depth = 0; depth < kK; ++depth) {
                accumulated += a[row * kK + depth] * b[depth * kN + column];
            }
            const float expected = kAlpha * accumulated + kBeta * initial_cd[row * kN + column];
            EXPECT_FLOAT_EQ(result[row * kN + column], expected) << "element " << row << ',' << column;
        }
    }
}

}  // namespace
