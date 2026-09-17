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

TEST(OperationValueTest, PreservesScalarAttributes) {
    const GemmAttributes gemm(FTRAIN_NUMERIC_TYPE_FP32);

    EXPECT_EQ(gemm.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
}

}  // namespace
}  // namespace ftrain
