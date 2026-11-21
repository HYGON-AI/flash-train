// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "flash_train/flash_train.h"

#include "flash_train/operation/gemm.hpp"
#include "flash_train/operation/operation.hpp"

namespace ftrain {
namespace {

static_assert(std::is_nothrow_constructible_v<GemmAttributes, FTrainNumericType>);

static_assert(noexcept(std::declval<const GemmAttributes&>().getComputeType()));

static_assert(std::is_constructible_v<OperationValue, GemmAttributes>);
static_assert(std::is_constructible_v<OperationValue, AGGroupGemmAttributes>);
static_assert(std::is_constructible_v<OperationValue, MoEReduceRSAttributes>);

TEST(OperationValueTest, PreservesScalarAttributes) {
    const GemmAttributes gemm(FTRAIN_NUMERIC_TYPE_FP32);

    EXPECT_EQ(gemm.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
}

TEST(OperationValueTest, PreservesMoeTensorMetadataByValue) {
    int identity = 0;
    AGGroupGemmAttributes ag{};
    MoEReduceRSAttributes rs{};
    ag.tensors[1] = {&identity, 7};
    rs.tensors[2] = {&identity, 13};
    const OperationValue ag_value(ag);
    const OperationValue rs_value(rs);
    ag.tensors[1].version  = 99;
    rs.tensors[2].identity = nullptr;
    EXPECT_EQ(std::get<AGGroupGemmAttributes>(ag_value).tensors[1].identity, &identity);
    EXPECT_EQ(std::get<AGGroupGemmAttributes>(ag_value).tensors[1].version, 7);
    EXPECT_EQ(std::get<MoEReduceRSAttributes>(rs_value).tensors[2].identity, &identity);
    EXPECT_EQ(std::get<MoEReduceRSAttributes>(rs_value).tensors[2].version, 13);
}

}  // namespace
}  // namespace ftrain
