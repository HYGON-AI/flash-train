// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_TESTS_FAMILY_GDN_PROBLEM_CASES_HPP_
#define FTRAIN_TESTS_FAMILY_GDN_PROBLEM_CASES_HPP_

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/error.hpp"
#include "flash_train/family/gdn/problem.hpp"

namespace ftrain_test {

using Dims = std::vector<std::int64_t>;

struct Port {
    Dims dims;
    FTrainNumericType type;
    void* memory{reinterpret_cast<void*>(0x1000)};
    bool host{false};
    FTrainIndexType index{FTRAIN_INDEX_TYPE_CONTINUOUS};
    Dims strides{};
};

template<ftrain::OperationKind Kind>
std::vector<Port> ports() {
    const Dims q{2, 64, 2, 8}, v{2, 64, 4, 16}, gate{2, 64, 4}, gate2{2, 64, 4, 8};
    const Dims empty{0, 4, 8, 16};
    if constexpr (Kind == ftrain::OperationKind::kGdnForward) {
        return {{q, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {gate, FTRAIN_NUMERIC_TYPE_FP32},
                {gate, FTRAIN_NUMERIC_TYPE_FP32}, {empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr}};
    } else if constexpr (Kind == ftrain::OperationKind::kGdnBackward) {
        return {{q, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {gate, FTRAIN_NUMERIC_TYPE_FP32},
                {gate, FTRAIN_NUMERIC_TYPE_FP32}, {v, FTRAIN_NUMERIC_TYPE_BF16},
                {empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr}, {empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr},
                {q, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {gate, FTRAIN_NUMERIC_TYPE_FP32},
                {gate, FTRAIN_NUMERIC_TYPE_FP32}, {empty, FTRAIN_NUMERIC_TYPE_FP32, nullptr}};
    } else if constexpr (Kind == ftrain::OperationKind::kGdn2Forward) {
        return {{q, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {gate2, FTRAIN_NUMERIC_TYPE_FP32},
                {gate2, FTRAIN_NUMERIC_TYPE_FP32}, {v, FTRAIN_NUMERIC_TYPE_FP32},
                {v, FTRAIN_NUMERIC_TYPE_BF16}};
    } else {
        return {{q, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {gate2, FTRAIN_NUMERIC_TYPE_FP32},
                {gate2, FTRAIN_NUMERIC_TYPE_FP32}, {v, FTRAIN_NUMERIC_TYPE_FP32},
                {v, FTRAIN_NUMERIC_TYPE_BF16}, {q, FTRAIN_NUMERIC_TYPE_BF16},
                {q, FTRAIN_NUMERIC_TYPE_BF16}, {v, FTRAIN_NUMERIC_TYPE_BF16},
                {gate2, FTRAIN_NUMERIC_TYPE_FP32}, {gate2, FTRAIN_NUMERIC_TYPE_BF16},
                {v, FTRAIN_NUMERIC_TYPE_BF16}};
    }
}

template<ftrain::OperationKind Kind>
ftrain::GdnProblem<Kind> problem(const std::vector<Port>& specs, float scale = 0.5F) {
    std::vector<ftrain::Tensor> tensors;
    for (const Port& port : specs) {
        const FTrainStorageView view{port.memory, port.dims.data(),
                                    port.index == FTRAIN_INDEX_TYPE_INVALID ? port.strides.data() : nullptr,
                                    static_cast<std::uint8_t>(port.dims.size()), port.type, port.index, port.host};
        tensors.emplace_back(ftrain::StorageView{view});
    }
    return ftrain::GdnProblem<Kind>{std::move(tensors), ftrain::GdnAttributes<Kind>{scale}};
}

template<ftrain::OperationKind Kind>
void expectStatus(const std::vector<Port>& specs, FTrainStatus status, float scale = 0.5F) {
    try {
        static_cast<void>(problem<Kind>(specs, scale));
        ADD_FAILURE() << "Expected status " << static_cast<int>(status);
    } catch (const ftrain::Exception& error) {
        EXPECT_EQ(error.getResult().getStatus(), status) << error.what();
    }
}

#define FTRAIN_GDN_PROBLEM_CASES(SUITE, KIND)                                                       \
    TEST(SUITE, ValidGetters) {                                                                       \
        auto p = ftrain_test::problem<KIND>(ftrain_test::ports<KIND>());                              \
        EXPECT_EQ(p.getB(), 2U); EXPECT_EQ(p.getS(), 64U); EXPECT_EQ(p.getH(), 2U);                  \
        EXPECT_EQ(p.getHV(), 4U); EXPECT_EQ(p.getK(), 8U); EXPECT_EQ(p.getV(), 16U);                 \
        EXPECT_TRUE(p.isGqa()); EXPECT_EQ(p.getChunkSize(), 64); EXPECT_FLOAT_EQ(p.getScale(), 0.5F); \
        EXPECT_EQ(p.getQNumericType(), FTRAIN_NUMERIC_TYPE_BF16);                                    \
        EXPECT_EQ(p.getPortAlignment(0), 16U); EXPECT_TRUE(p.isPortContinuous(0));                    \
    }                                                                                                 \
    TEST(SUITE, RejectRank) {                                                                         \
        auto specs = ftrain_test::ports<KIND>(); specs[1].dims.pop_back();                           \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);                       \
    }                                                                                                 \
    TEST(SUITE, RejectMismatchedDimensions) {                                                         \
        auto specs = ftrain_test::ports<KIND>(); specs[1].dims[3] = 9;                               \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);                       \
    }                                                                                                 \
    TEST(SUITE, RejectNonIntegralHeadRatio) {                                                         \
        auto specs = ftrain_test::ports<KIND>(); specs[0].dims[2] = 3; specs[1].dims[2] = 3;         \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);                       \
    }                                                                                                 \
    TEST(SUITE, RejectZeroRequiredDimension) {                                                       \
        auto specs = ftrain_test::ports<KIND>(); specs[0].dims[0] = 0;                               \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);                       \
    }                                                                                                 \
    TEST(SUITE, RejectHostMemory) {                                                                   \
        auto specs = ftrain_test::ports<KIND>(); specs[2].host = true;                               \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_INVALID_ARGUMENT);                       \
    }                                                                                                 \
    TEST(SUITE, RejectInvalidScale) {                                                                 \
        ftrain_test::expectStatus<KIND>(ftrain_test::ports<KIND>(), FTRAIN_STATUS_INVALID_ARGUMENT,   \
                                        std::numeric_limits<float>::infinity());                     \
    }                                                                                                 \
    TEST(SUITE, RejectElementOverflow) {                                                             \
        auto specs = ftrain_test::ports<KIND>();                                                       \
        for (auto& port : specs) for (auto& dim : port.dims) if (dim > 0)                            \
            dim = std::numeric_limits<std::int64_t>::max();                                          \
        ftrain_test::expectStatus<KIND>(specs, FTRAIN_STATUS_OVERFLOW);                               \
    }                                                                                                 \
    TEST(SUITE, KeyTracksSelectionInputs) {                                                          \
        auto specs = ftrain_test::ports<KIND>();                                                       \
        auto original = ftrain_test::problem<KIND>(specs).getProblemKey();                            \
        specs[0].memory = reinterpret_cast<void*>(0x2000);                                           \
        EXPECT_EQ(ftrain_test::problem<KIND>(specs).getProblemKey(), original);                      \
        specs[0].type = FTRAIN_NUMERIC_TYPE_FP32;                                                    \
        if constexpr (KIND == ftrain::OperationKind::kGdnForward ||                                  \
                      KIND == ftrain::OperationKind::kGdn2Forward) specs[6].type = FTRAIN_NUMERIC_TYPE_FP32; \
        EXPECT_NE(ftrain_test::problem<KIND>(specs).getProblemKey(), original);                      \
        specs = ftrain_test::ports<KIND>();                                                           \
        for (auto& port : specs) port.dims[0] = 3;                                                   \
        EXPECT_NE(ftrain_test::problem<KIND>(specs).getProblemKey(), original);                      \
        EXPECT_EQ(ftrain_test::problem<KIND>(ftrain_test::ports<KIND>(), 0.25F).getProblemKey(),      \
                  original);                                                                          \
        specs = ftrain_test::ports<KIND>(); specs[0].memory = reinterpret_cast<void*>(0x1004);       \
        EXPECT_NE(ftrain_test::problem<KIND>(specs).getProblemKey(), original);                      \
    }

}  // namespace ftrain_test

#endif
