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

static_assert(std::is_constructible_v<TensorStorage, StorageView&&>);
static_assert(std::is_nothrow_constructible_v<TensorStorage, StorageView&&>);
static_assert(std::is_nothrow_move_constructible_v<TensorStorage>);
static_assert(noexcept(std::declval<const TensorStorage&>().getStorageView()));

static_assert(std::is_constructible_v<TensorListStorage, std::vector<StorageView>&&>);
static_assert(std::is_nothrow_constructible_v<TensorListStorage, std::vector<StorageView>&&>);
static_assert(std::is_nothrow_move_constructible_v<TensorListStorage>);
static_assert(noexcept(std::declval<const TensorListStorage&>().getStorageViews()));

static_assert(std::is_constructible_v<GroupedTensorStorage, std::uint32_t, StorageView&&, std::optional<StorageView>&&,
                                      std::vector<StorageView>&&, std::vector<StorageView>&&>);
static_assert(std::is_nothrow_move_constructible_v<GroupedTensorStorage>);
static_assert(noexcept(std::declval<const GroupedTensorStorage&>().getNumGroups()));
static_assert(noexcept(std::declval<const GroupedTensorStorage&>().getData()));
static_assert(noexcept(std::declval<const GroupedTensorStorage&>().getOffsets()));
static_assert(noexcept(std::declval<const GroupedTensorStorage&>().getDimSizes()));
static_assert(noexcept(std::declval<const GroupedTensorStorage&>().getStrides()));

static_assert(std::is_nothrow_default_constructible_v<Tensor>);
static_assert(noexcept(std::declval<const Tensor&>().isSet()));
static_assert(noexcept(std::declval<Tensor&>().setStorage(std::declval<TensorStorage&&>())));
static_assert(std::is_same_v<decltype(std::declval<const Tensor&>().getStorage()), const TensorStorage&>);

static_assert(std::is_nothrow_default_constructible_v<TensorList>);
static_assert(noexcept(std::declval<const TensorList&>().isSet()));
static_assert(noexcept(std::declval<TensorList&>().setStorage(std::declval<TensorListStorage&&>())));
static_assert(std::is_same_v<decltype(std::declval<const TensorList&>().getStorage()), const TensorListStorage&>);

static_assert(std::is_nothrow_default_constructible_v<GroupedTensor>);
static_assert(noexcept(std::declval<const GroupedTensor&>().isSet()));
static_assert(noexcept(std::declval<GroupedTensor&>().setStorage(std::declval<GroupedTensorStorage&&>())));
static_assert(std::is_same_v<decltype(std::declval<const GroupedTensor&>().getStorage()), const GroupedTensorStorage&>);

TEST(TensorStorageTest, OwnsStorageView) {
    int memory = 0;
    std::int64_t dims[]{4, 8};
    std::int64_t strides[]{8, 1};
    const FTrainStorageView input{&memory, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_INVALID, true};
    StorageView storage_view(input);

    const TensorStorage storage(std::move(storage_view));

    EXPECT_EQ(storage.getStorageView().getMemory(), &memory);
    EXPECT_EQ(storage.getStorageView().getDims(), (std::vector<std::int64_t>{4, 8}));
}

TEST(TensorTest, StartsUnsetAndRejectsStorageAccess) {
    const Tensor tensor;

    EXPECT_FALSE(tensor.isSet());

    try {
        static_cast<void>(tensor.getStorage());
        FAIL() << "Unset Tensor returned concrete storage";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }

    EXPECT_FALSE(tensor.isSet());
}

TEST(TensorTest, SetsAndReplacesStorage) {
    int memory = 0;
    std::int64_t dims[]{4, 8};
    std::int64_t strides[]{8, 1};
    const FTrainStorageView input{&memory, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_INVALID, true};
    StorageView storage_view(input);
    TensorStorage storage(std::move(storage_view));

    Tensor tensor;
    tensor.setStorage(std::move(storage));
    const TensorStorage& stored_storage = tensor.getStorage();

    EXPECT_TRUE(tensor.isSet());
    EXPECT_EQ(stored_storage.getStorageView().getMemory(), &memory);

    int replacement_memory = 0;
    std::int64_t replacement_dims[]{16};
    const FTrainStorageView replacement_input{&replacement_memory,
                                              replacement_dims,
                                              nullptr,
                                              1,
                                              FTRAIN_NUMERIC_TYPE_BF16,
                                              FTRAIN_INDEX_TYPE_CONTINUOUS,
                                              false};
    StorageView replacement_storage_view(replacement_input);
    TensorStorage replacement_storage(std::move(replacement_storage_view));

    tensor.setStorage(std::move(replacement_storage));

    EXPECT_EQ(tensor.getStorage().getStorageView().getMemory(), &replacement_memory);
    EXPECT_EQ(tensor.getStorage().getStorageView().getDims(), (std::vector<std::int64_t>{16}));
}

TEST(TensorListStorageTest, PreservesOrderedHeterogeneousStorageViews) {
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

    const TensorListStorage storage(std::move(storage_views));

    ASSERT_EQ(storage.getStorageViews().size(), 2);
    EXPECT_EQ(storage.getStorageViews()[0].getMemory(), &first_memory);
    EXPECT_EQ(storage.getStorageViews()[1].getMemory(), &second_memory);
}

TEST(TensorListStorageTest, AcceptsEmptyStorageViews) {
    std::vector<StorageView> storage_views;

    const TensorListStorage storage(std::move(storage_views));

    EXPECT_TRUE(storage.getStorageViews().empty());
}

TEST(TensorListTest, StartsUnsetAndRejectsStorageAccess) {
    const TensorList tensor_list;

    EXPECT_FALSE(tensor_list.isSet());

    try {
        static_cast<void>(tensor_list.getStorage());
        FAIL() << "Unset TensorList returned concrete storage";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }

    EXPECT_FALSE(tensor_list.isSet());
}

TEST(TensorListTest, SetsAndReplacesStorage) {
    int first_memory = 0;
    std::int64_t first_dims[]{2, 4};
    const FTrainStorageView first_input{
        &first_memory, first_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    std::vector<StorageView> storage_views;
    storage_views.emplace_back(first_input);
    TensorListStorage storage(std::move(storage_views));

    TensorList tensor_list;
    tensor_list.setStorage(std::move(storage));
    const TensorListStorage& stored_storage = tensor_list.getStorage();

    EXPECT_TRUE(tensor_list.isSet());
    ASSERT_EQ(stored_storage.getStorageViews().size(), 1);
    EXPECT_EQ(stored_storage.getStorageViews()[0].getMemory(), &first_memory);

    EXPECT_EQ(&tensor_list.getStorage(), &stored_storage);
    EXPECT_EQ(tensor_list.getStorage().getStorageViews()[0].getMemory(), &first_memory);

    std::vector<StorageView> replacement_storage_views;
    TensorListStorage replacement_storage(std::move(replacement_storage_views));
    tensor_list.setStorage(std::move(replacement_storage));

    EXPECT_TRUE(tensor_list.isSet());
    EXPECT_TRUE(tensor_list.getStorage().getStorageViews().empty());
}

TEST(TensorListTest, DistinguishesUnsetFromSetEmptyStorage) {
    std::vector<StorageView> storage_views;
    TensorListStorage storage(std::move(storage_views));
    TensorList tensor_list;

    EXPECT_FALSE(tensor_list.isSet());

    tensor_list.setStorage(std::move(storage));

    EXPECT_TRUE(tensor_list.isSet());
    EXPECT_TRUE(tensor_list.getStorage().getStorageViews().empty());
}

TEST(GroupedTensorStorageTest, TakesOrderedMetadataAndKeepsNullMemoryOffsetsPresent) {
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

    const GroupedTensorStorage storage(3, std::move(data), std::move(offsets), std::move(dim_sizes),
                                       std::move(strides));

    EXPECT_EQ(storage.getNumGroups(), 3);
    EXPECT_EQ(storage.getData().getMemory(), &data_memory);
    ASSERT_TRUE(storage.getOffsets().has_value());
    EXPECT_EQ(storage.getOffsets()->getMemory(), nullptr);
    ASSERT_EQ(storage.getDimSizes().size(), 2);
    EXPECT_EQ(storage.getDimSizes()[0].getMemory(), &first_dim_size_memory);
    EXPECT_EQ(storage.getDimSizes()[1].getMemory(), &second_dim_size_memory);
    ASSERT_EQ(storage.getStrides().size(), 2);
    EXPECT_EQ(storage.getStrides()[0].getMemory(), &first_stride_memory);
    EXPECT_EQ(storage.getStrides()[1].getMemory(), &second_stride_memory);
}

TEST(GroupedTensorStorageTest, AcceptsZeroGroupsAndMissingOptionalMetadata) {
    std::int64_t data_dims[]{2, 4};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;

    const GroupedTensorStorage storage(0, std::move(data), std::move(offsets), std::move(dim_sizes),
                                       std::move(strides));

    EXPECT_EQ(storage.getNumGroups(), 0);
    EXPECT_FALSE(storage.getOffsets().has_value());
    EXPECT_TRUE(storage.getDimSizes().empty());
    EXPECT_TRUE(storage.getStrides().empty());
}

TEST(GroupedTensorTest, DistinguishesUnsetAndZeroGroupsAndReplacesStorage) {
    GroupedTensor grouped_tensor;

    EXPECT_FALSE(grouped_tensor.isSet());

    try {
        static_cast<void>(grouped_tensor.getStorage());
        FAIL() << "Unset GroupedTensor returned concrete storage";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }

    EXPECT_FALSE(grouped_tensor.isSet());

    std::int64_t initial_data_dims[]{2, 4};
    const FTrainStorageView initial_data_input{
        nullptr, initial_data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView initial_data(initial_data_input);
    std::optional<StorageView> initial_offsets;
    std::vector<StorageView> initial_dim_sizes;
    std::vector<StorageView> initial_strides;
    GroupedTensorStorage initial_storage(0, std::move(initial_data), std::move(initial_offsets),
                                         std::move(initial_dim_sizes), std::move(initial_strides));

    grouped_tensor.setStorage(std::move(initial_storage));
    const GroupedTensorStorage& stored_storage = grouped_tensor.getStorage();

    EXPECT_TRUE(grouped_tensor.isSet());
    EXPECT_EQ(stored_storage.getNumGroups(), 0);
    EXPECT_EQ(&grouped_tensor.getStorage(), &stored_storage);

    int replacement_memory = 0;
    std::int64_t replacement_data_dims[]{8};
    const FTrainStorageView replacement_data_input{&replacement_memory,      replacement_data_dims,        nullptr, 1,
                                                   FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    StorageView replacement_data(replacement_data_input);
    std::optional<StorageView> replacement_offsets;
    std::vector<StorageView> replacement_dim_sizes;
    std::vector<StorageView> replacement_strides;
    GroupedTensorStorage replacement_storage(2, std::move(replacement_data), std::move(replacement_offsets),
                                             std::move(replacement_dim_sizes), std::move(replacement_strides));

    grouped_tensor.setStorage(std::move(replacement_storage));

    EXPECT_EQ(grouped_tensor.getStorage().getNumGroups(), 2);
    EXPECT_EQ(grouped_tensor.getStorage().getData().getMemory(), &replacement_memory);
}

TEST(GroupedTensorStorageTest, RejectsDimSizeCountDifferentFromDataRank) {
    std::int64_t data_dims[]{2, 4};
    std::int64_t metadata_dims[]{3};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView dim_size_input{
        nullptr, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    dim_sizes.emplace_back(dim_size_input);
    std::vector<StorageView> strides;

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a dim_size count different from data rank";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsStrideCountDifferentFromDataRank) {
    std::int64_t data_dims[]{2, 4};
    std::int64_t metadata_dims[]{3};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView stride_input{
        nullptr, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;
    strides.emplace_back(stride_input);

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a stride count different from data rank";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsOffsetsRankDifferentFromOne) {
    std::int64_t data_dims[]{4};
    std::int64_t offsets_dims[]{1, 3};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView offsets_input{
        nullptr, offsets_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets(std::in_place, offsets_input);
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted non-rank-one offsets";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsOffsetsLengthDifferentFromNumGroups) {
    std::int64_t data_dims[]{4};
    std::int64_t offsets_dims[]{2};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView offsets_input{
        nullptr, offsets_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets(std::in_place, offsets_input);
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted an offsets length different from num_groups";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsDimSizeRankDifferentFromOne) {
    std::int64_t data_dims[]{4};
    std::int64_t metadata_dims[]{1, 3};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView dim_size_input{
        nullptr, metadata_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    dim_sizes.emplace_back(dim_size_input);
    std::vector<StorageView> strides;

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a non-rank-one dim_size";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsDimSizeLengthDifferentFromNumGroups) {
    std::int64_t data_dims[]{4};
    std::int64_t metadata_dims[]{2};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView dim_size_input{
        nullptr, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    dim_sizes.emplace_back(dim_size_input);
    std::vector<StorageView> strides;

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a dim_size length different from num_groups";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsStrideRankDifferentFromOne) {
    std::int64_t data_dims[]{4};
    std::int64_t metadata_dims[]{1, 3};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView stride_input{
        nullptr, metadata_dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;
    strides.emplace_back(stride_input);

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a non-rank-one stride";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

TEST(GroupedTensorStorageTest, RejectsStrideLengthDifferentFromNumGroups) {
    std::int64_t data_dims[]{4};
    std::int64_t metadata_dims[]{2};
    const FTrainStorageView data_input{
        nullptr, data_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_CONTINUOUS, false};
    const FTrainStorageView stride_input{
        nullptr, metadata_dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INT64, FTRAIN_INDEX_TYPE_CONTINUOUS, true};
    StorageView data(data_input);
    std::optional<StorageView> offsets;
    std::vector<StorageView> dim_sizes;
    std::vector<StorageView> strides;
    strides.emplace_back(stride_input);

    try {
        static_cast<void>(
            GroupedTensorStorage(3, std::move(data), std::move(offsets), std::move(dim_sizes), std::move(strides)));
        FAIL() << "GroupedTensorStorage accepted a stride length different from num_groups";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
    }
}

}  // namespace
}  // namespace ftrain
