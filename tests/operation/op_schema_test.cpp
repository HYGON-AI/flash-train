// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <initializer_list>
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

static_assert(std::is_nothrow_constructible_v<SparseAttnBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseAttnBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseAttnBwdAttributes>);

TEST(OperationValueTest, PreservesScalarAttributes) {
    const GemmAttributes gemm(FTRAIN_NUMERIC_TYPE_FP32);

    EXPECT_EQ(gemm.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
}

TEST(OperationValueTest, PreservesAdaptedScalarAttributes) {
    for (const auto type : {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32}) {
        EXPECT_EQ(SparseAttnBwdAttributes(type).getComputeType(), type);
    }
}

}  // namespace
}  // namespace ftrain
