// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

static_assert(!std::is_default_constructible_v<Tensor>);
static_assert(std::is_nothrow_constructible_v<Tensor, StorageView&&>);
static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(noexcept(std::declval<const Tensor&>().getStorageView()));

static_assert(!std::is_default_constructible_v<TensorList>);
static_assert(std::is_nothrow_constructible_v<TensorList, std::vector<StorageView>&&>);
static_assert(std::is_nothrow_move_constructible_v<TensorList>);
static_assert(noexcept(std::declval<const TensorList&>().getStorageViews()));

static_assert(!std::is_default_constructible_v<GroupedTensor>);
static_assert(std::is_constructible_v<GroupedTensor, std::uint32_t, StorageView&&, std::optional<StorageView>&&,
                                      std::vector<StorageView>&&, std::vector<StorageView>&&>);
static_assert(std::is_nothrow_move_constructible_v<GroupedTensor>);
static_assert(noexcept(std::declval<const GroupedTensor&>().getNumGroups()));
static_assert(noexcept(std::declval<const GroupedTensor&>().getData()));
static_assert(noexcept(std::declval<const GroupedTensor&>().getOffsets()));
static_assert(noexcept(std::declval<const GroupedTensor&>().getDimSizes()));
static_assert(noexcept(std::declval<const GroupedTensor&>().getStrides()));

template<typename Function>
void expectInvalidArgument(Function&& function) {
    try {
        std::forward<Function>(function)();
        FAIL() << "GroupedTensor accepted invalid metadata shapes";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(TensorTest, OwnsStorageView) {
    int memory = 0;
    std::int64_t dims[]{4, 8};
    std::int64_t strides[]{8, 1};
    const FTrainStorageView input{&memory, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_INVALID, true};
    StorageView storage_view(input);

    const Tensor tensor(std::move(storage_view));

    EXPECT_EQ(tensor.getStorageView().getMemory(), &memory);
    EXPECT_EQ(tensor.getStorageView().getDims(), (std::vector<std::int64_t>{4, 8}));
}

TEST(TensorListTest, PreservesOrderedHeterogeneousStorageViews) {
    int first_memory  = 0;
    int second_memory = 0;
    std::int64_t first_dims[]{2, 4};
    std::int64_t second_dims[]{3};
    const FTrainStorageView first_input{
        &first_memory, first_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView second_input{
        &second_memory, second_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    std::vector<StorageView> storage_views;
    storage_views.emplace_back(first_input);
    storage_views.emplace_back(second_input);

    const TensorList tensor_list(std::move(storage_views));

    ASSERT_EQ(tensor_list.getStorageViews().size(), 2);
    EXPECT_EQ(tensor_list.getStorageViews()[0].getMemory(), &first_memory);
    EXPECT_EQ(tensor_list.getStorageViews()[1].getMemory(), &second_memory);
}

TEST(TensorListTest, AcceptsEmptyStorageViews) {
    std::vector<StorageView> storage_views;

    const TensorList tensor_list(std::move(storage_views));

    EXPECT_TRUE(tensor_list.getStorageViews().empty());
}

TEST(GroupedTensorTest, TakesOrderedMetadataAndKeepsNullMemoryOffsetsPresent) {
    int data_memory            = 0;
    int first_dim_size_memory  = 0;
    int second_dim_size_memory = 0;
    int first_stride_memory    = 0;
    int second_stride_memory   = 0;
    std::int64_t data_dims[]{2, 4};
    std::int64_t metadata_dims[]{3};
    const FTrainStorageView data_input{
        &data_memory, data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView offsets_input{
        nullptr, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_UINT64, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView first_dim_size_input{
        &first_dim_size_memory,       metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64,
        FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    const FTrainStorageView second_dim_size_input{
        &second_dim_size_memory,      metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT32,
        FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    const FTrainStorageView first_stride_input{
        &first_stride_memory, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    const FTrainStorageView second_stride_input{
        &second_stride_memory,        metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT32,
        FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets(std::in_place, offsets_input);
    std::vector<StorageView> dim_sizes;
    dim_sizes.emplace_back(first_dim_size_input);
    dim_sizes.emplace_back(second_dim_size_input);
    std::vector<StorageView> strides;
    strides.emplace_back(first_stride_input);
    strides.emplace_back(second_stride_input);

    const GroupedTensor grouped_tensor(3, std::move(data), std::move(offsets), std::move(dim_sizes),
                                       std::move(strides));

    EXPECT_EQ(grouped_tensor.getNumGroups(), 3);
    EXPECT_EQ(grouped_tensor.getData().getMemory(), &data_memory);
    ASSERT_TRUE(grouped_tensor.getOffsets().has_value());
    EXPECT_EQ(grouped_tensor.getOffsets()->getMemory(), nullptr);
    ASSERT_EQ(grouped_tensor.getDimSizes().size(), 2);
    EXPECT_EQ(grouped_tensor.getDimSizes()[0].getMemory(), &first_dim_size_memory);
    EXPECT_EQ(grouped_tensor.getDimSizes()[1].getMemory(), &second_dim_size_memory);
    ASSERT_EQ(grouped_tensor.getStrides().size(), 2);
    EXPECT_EQ(grouped_tensor.getStrides()[0].getMemory(), &first_stride_memory);
    EXPECT_EQ(grouped_tensor.getStrides()[1].getMemory(), &second_stride_memory);
}

TEST(GroupedTensorTest, AcceptsZeroGroupsAndMissingOptionalMetadata) {
    std::int64_t data_dims[]{2, 4};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;

    const GroupedTensor grouped_tensor(0, std::move(data), std::move(offsets), std::move(dim_sizes),
                                       std::move(strides));

    EXPECT_EQ(grouped_tensor.getNumGroups(), 0);
    EXPECT_FALSE(grouped_tensor.getOffsets().has_value());
    EXPECT_TRUE(grouped_tensor.getDimSizes().empty());
    EXPECT_TRUE(grouped_tensor.getStrides().empty());
}

constexpr std::int64_t data_rank_two[]{2, 4};
constexpr std::int64_t data_rank_one[]{4};
constexpr std::int64_t length_two[]{2};
constexpr std::int64_t length_three[]{3};
constexpr std::int64_t metadata_rank_two[]{1, 3};

StorageView makeDataView(const std::int64_t* data_dims, std::size_t num_dims) {
    const FTrainStorageView data_input{nullptr,
                                       data_dims,
                                       nullptr,
                                       static_cast<std::uint8_t>(num_dims),
                                       FTRAIN_NUMERIC_TYPE_FP16,
                                       FTRAIN_INDEX_TYPE_CONTINUOUS,
                                       false};
    return StorageView(data_input);
}

StorageView makeMetadataView(const std::int64_t* metadata_dims, std::size_t num_dims, FTrainNumericType numeric_type) {
    const FTrainStorageView metadata_input{nullptr,      metadata_dims,
                                           nullptr,      static_cast<std::uint8_t>(num_dims),
                                           numeric_type, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                           true};
    return StorageView(metadata_input);
}

TEST(GroupedTensorTest, RejectsDimSizeCountDifferentFromDataRank) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_two, 2), std::optional<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(length_three, 1, FTRAIN_NUMERIC_TYPE_INT64)},
                          std::vector<StorageView>{}));
    });
}

TEST(GroupedTensorTest, RejectsStrideCountDifferentFromDataRank) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_two, 2), std::optional<StorageView>{}, std::vector<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(length_three, 1, FTRAIN_NUMERIC_TYPE_INT64)}));
    });
}

TEST(GroupedTensorTest, RejectsOffsetsRankDifferentFromOne) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_one, 1),
                          std::optional<StorageView>(
                              std::in_place, makeMetadataView(metadata_rank_two, 2, FTRAIN_NUMERIC_TYPE_UINT64)),
                          std::vector<StorageView>{}, std::vector<StorageView>{}));
    });
}

TEST(GroupedTensorTest, RejectsOffsetsLengthDifferentFromNumGroups) {
    expectInvalidArgument([&] {
        static_cast<void>(GroupedTensor(
            3, makeDataView(data_rank_one, 1),
            std::optional<StorageView>(std::in_place, makeMetadataView(length_two, 1, FTRAIN_NUMERIC_TYPE_UINT64)),
            std::vector<StorageView>{}, std::vector<StorageView>{}));
    });
}

TEST(GroupedTensorTest, RejectsDimSizeRankDifferentFromOne) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_one, 1), std::optional<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(metadata_rank_two, 2, FTRAIN_NUMERIC_TYPE_INT64)},
                          std::vector<StorageView>{}));
    });
}

TEST(GroupedTensorTest, RejectsDimSizeLengthDifferentFromNumGroups) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_one, 1), std::optional<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(length_two, 1, FTRAIN_NUMERIC_TYPE_INT64)},
                          std::vector<StorageView>{}));
    });
}

TEST(GroupedTensorTest, RejectsStrideRankDifferentFromOne) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_one, 1), std::optional<StorageView>{}, std::vector<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(metadata_rank_two, 2, FTRAIN_NUMERIC_TYPE_INT64)}));
    });
}

TEST(GroupedTensorTest, RejectsStrideLengthDifferentFromNumGroups) {
    expectInvalidArgument([&] {
        static_cast<void>(
            GroupedTensor(3, makeDataView(data_rank_one, 1), std::optional<StorageView>{}, std::vector<StorageView>{},
                          std::vector<StorageView>{makeMetadataView(length_two, 1, FTRAIN_NUMERIC_TYPE_INT64)}));
    });
}

}  // namespace
}  // namespace ftrain
