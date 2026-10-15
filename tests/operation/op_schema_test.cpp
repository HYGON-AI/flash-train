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

static_assert(std::is_nothrow_constructible_v<CSACompressorBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const CSACompressorBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, CSACompressorBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<CSACompressorFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const CSACompressorFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, CSACompressorFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<DenseAttnScoreAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const DenseAttnScoreAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, DenseAttnScoreAttributes>);
static_assert(std::is_nothrow_constructible_v<DenseAttnScoreThdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const DenseAttnScoreThdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, DenseAttnScoreThdAttributes>);
static_assert(std::is_nothrow_constructible_v<DenseIndexerScoreAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const DenseIndexerScoreAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, DenseIndexerScoreAttributes>);
static_assert(std::is_nothrow_constructible_v<DenseIndexerScoreThdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const DenseIndexerScoreThdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, DenseIndexerScoreThdAttributes>);
static_assert(std::is_nothrow_constructible_v<HAggregateBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const HAggregateBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, HAggregateBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<HAggregateFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const HAggregateFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, HAggregateFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<HPostBdaBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const HPostBdaBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, HPostBdaBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<HPostBdaFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const HPostBdaFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, HPostBdaFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<IndexerBwdDenseAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const IndexerBwdDenseAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, IndexerBwdDenseAttributes>);
static_assert(std::is_nothrow_constructible_v<IndexerBwdDenseThdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const IndexerBwdDenseThdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, IndexerBwdDenseThdAttributes>);
static_assert(std::is_nothrow_constructible_v<IndexerBwdSparseAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const IndexerBwdSparseAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, IndexerBwdSparseAttributes>);
static_assert(std::is_nothrow_constructible_v<IndexerFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const IndexerFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, IndexerFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<IndexerFwdThdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const IndexerFwdThdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, IndexerFwdThdAttributes>);
static_assert(std::is_nothrow_constructible_v<ProjRmsBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const ProjRmsBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, ProjRmsBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<ProjRmsComputeHBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const ProjRmsComputeHBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, ProjRmsComputeHBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<ProjRmsComputeHFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const ProjRmsComputeHFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, ProjRmsComputeHFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<ProjRmsFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const ProjRmsFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, ProjRmsFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<SinkhornBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SinkhornBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SinkhornBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<SinkhornFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SinkhornFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SinkhornFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<SparseAttnBwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseAttnBwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseAttnBwdAttributes>);
static_assert(std::is_nothrow_constructible_v<SparseAttnFwdAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseAttnFwdAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseAttnFwdAttributes>);
static_assert(std::is_nothrow_constructible_v<SparseAttnFwdIndexerAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseAttnFwdIndexerAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseAttnFwdIndexerAttributes>);
static_assert(std::is_nothrow_constructible_v<SparseAttnTargetAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseAttnTargetAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseAttnTargetAttributes>);
static_assert(std::is_nothrow_constructible_v<SparseIndexerPredictAttributes, FTrainNumericType>);
static_assert(noexcept(std::declval<const SparseIndexerPredictAttributes&>().getComputeType()));
static_assert(std::is_constructible_v<OperationValue, SparseIndexerPredictAttributes>);

TEST(OperationValueTest, PreservesScalarAttributes) {
    const GemmAttributes gemm(FTRAIN_NUMERIC_TYPE_FP32);

    EXPECT_EQ(gemm.getComputeType(), FTRAIN_NUMERIC_TYPE_FP32);
}

TEST(OperationValueTest, PreservesAdaptedScalarAttributes) {
    for (const auto type : {FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_NUMERIC_TYPE_FP32}) {
        EXPECT_EQ(CSACompressorBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(CSACompressorFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(DenseAttnScoreAttributes(type).getComputeType(), type);
        EXPECT_EQ(DenseAttnScoreThdAttributes(type).getComputeType(), type);
        EXPECT_EQ(DenseIndexerScoreAttributes(type).getComputeType(), type);
        EXPECT_EQ(DenseIndexerScoreThdAttributes(type).getComputeType(), type);
        EXPECT_EQ(HAggregateBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(HAggregateFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(HPostBdaBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(HPostBdaFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(IndexerBwdDenseAttributes(type).getComputeType(), type);
        EXPECT_EQ(IndexerBwdDenseThdAttributes(type).getComputeType(), type);
        EXPECT_EQ(IndexerBwdSparseAttributes(type).getComputeType(), type);
        EXPECT_EQ(IndexerFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(IndexerFwdThdAttributes(type).getComputeType(), type);
        EXPECT_EQ(ProjRmsBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(ProjRmsComputeHBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(ProjRmsComputeHFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(ProjRmsFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(SinkhornBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(SinkhornFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(SparseAttnBwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(SparseAttnFwdAttributes(type).getComputeType(), type);
        EXPECT_EQ(SparseAttnFwdIndexerAttributes(type).getComputeType(), type);
        EXPECT_EQ(SparseAttnTargetAttributes(type).getComputeType(), type);
        EXPECT_EQ(SparseIndexerPredictAttributes(type).getComputeType(), type);
    }
}

}  // namespace
}  // namespace ftrain
