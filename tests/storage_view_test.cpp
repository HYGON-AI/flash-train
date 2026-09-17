#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/error.hpp"
#include "flash_train/storage_view.hpp"

namespace ftrain {
namespace {

static_assert(!std::is_default_constructible_v<StorageView>);
static_assert(std::is_constructible_v<StorageView, const FTrainStorageView&>);
static_assert(!std::is_convertible_v<FTrainStorageView, StorageView>);
static_assert(noexcept(std::declval<const StorageView&>().getMemory()));
static_assert(noexcept(std::declval<const StorageView&>().getDims()));
static_assert(noexcept(std::declval<const StorageView&>().getStrides()));
static_assert(noexcept(std::declval<const StorageView&>().getNumericType()));
static_assert(noexcept(std::declval<const StorageView&>().getIndexType()));
static_assert(noexcept(std::declval<const StorageView&>().isHostMemory()));

TEST(StorageViewTest, CopiesMetadataAndPreservesScalarFields) {
    int memory = 0;
    std::int64_t dims[]{4, 8};
    std::int64_t strides[]{8, 1};
    const FTrainStorageView input{&memory, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP16, FTRAIN_INDEX_TYPE_INVALID, true};

    const StorageView storage_view(input);
    dims[0]    = 32;
    strides[0] = 64;

    EXPECT_EQ(storage_view.getMemory(), &memory);
    EXPECT_EQ(storage_view.getDims(), (std::vector<std::int64_t>{4, 8}));
    EXPECT_EQ(storage_view.getStrides(), (std::vector<std::int64_t>{8, 1}));
    EXPECT_EQ(storage_view.getNumericType(), FTRAIN_NUMERIC_TYPE_FP16);
    EXPECT_EQ(storage_view.getIndexType(), FTRAIN_INDEX_TYPE_INVALID);
    EXPECT_TRUE(storage_view.isHostMemory());
}

TEST(StorageViewTest, AcceptsNullStridesWithExplicitIndexType) {
    std::int64_t dims[]{4, 8};
    const FTrainStorageView input{nullptr, dims, nullptr, 2, FTRAIN_NUMERIC_TYPE_BF16, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    const StorageView storage_view(input);

    EXPECT_EQ(storage_view.getMemory(), nullptr);
    EXPECT_EQ(storage_view.getDims(), (std::vector<std::int64_t>{4, 8}));
    EXPECT_TRUE(storage_view.getStrides().empty());
    EXPECT_FALSE(storage_view.isHostMemory());
}

TEST(StorageViewTest, TreatsZeroDimensionalStorageAsScalar) {
    float scalar = 1.0F;
    const FTrainStorageView input{&scalar, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    const StorageView storage_view(input);

    EXPECT_EQ(storage_view.getMemory(), &scalar);
    EXPECT_TRUE(storage_view.getDims().empty());
    EXPECT_TRUE(storage_view.getStrides().empty());
}

TEST(StorageViewTest, RejectsNullMemoryForZeroDimensionalScalar) {
    const FTrainStorageView input{nullptr, nullptr, nullptr, 0, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted null memory for a scalar";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), "StorageView memory must not be null for a scalar");
    }
}

TEST(StorageViewTest, TreatsNonNullZeroLengthMetadataAsEmpty) {
    float scalar                     = 1.0F;
    std::int64_t dims_placeholder    = 17;
    std::int64_t strides_placeholder = 19;
    const FTrainStorageView input{&scalar, &dims_placeholder,        &strides_placeholder,
                                  0,       FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID,
                                  false};

    const StorageView storage_view(input);

    EXPECT_TRUE(storage_view.getDims().empty());
    EXPECT_TRUE(storage_view.getStrides().empty());
}

TEST(StorageViewTest, AcceptsZeroLengthDimension) {
    std::int64_t dims[]{2, 0, 4};
    const FTrainStorageView input{nullptr, dims, nullptr, 3, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    const StorageView storage_view(input);

    EXPECT_EQ(storage_view.getDims(), (std::vector<std::int64_t>{2, 0, 4}));
}

TEST(StorageViewTest, PreservesStrideValuesWithoutSemanticValidation) {
    std::int64_t dims[]{2, 4};
    std::int64_t strides[]{0, -1};
    const FTrainStorageView input{nullptr, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID,
                                  false};

    const StorageView storage_view(input);

    EXPECT_EQ(storage_view.getStrides(), (std::vector<std::int64_t>{0, -1}));
}

TEST(StorageViewTest, AcceptsHighestDefinedNumericAndIndexTypes) {
    std::int64_t dims[]{4};
    constexpr FTrainNumericType last_defined_numeric_type =
        static_cast<FTrainNumericType>(FTRAIN_NUMERIC_TYPE_COUNT - 1);
    constexpr FTrainIndexType last_defined_index_type = static_cast<FTrainIndexType>(FTRAIN_INDEX_TYPE_COUNT - 1);
    const FTrainStorageView input{nullptr, dims, nullptr, 1, last_defined_numeric_type, last_defined_index_type, false};

    const StorageView storage_view(input);

    EXPECT_EQ(storage_view.getNumericType(), last_defined_numeric_type);
    EXPECT_EQ(storage_view.getIndexType(), last_defined_index_type);
}

TEST(StorageViewTest, RejectsNullDimsForNonzeroRank) {
    std::int64_t strides[]{4, 1};
    const FTrainStorageView input{nullptr, nullptr, strides, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted null dims for nonzero rank";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), "StorageView dims must not be null when num_dims is 2");
    }
}

TEST(StorageViewTest, RejectsNegativeDimension) {
    std::int64_t dims[]{4, -1};
    std::int64_t strides[]{4, 1};
    const FTrainStorageView input{nullptr, dims, strides, 2, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted a negative dimension";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), "StorageView dimension 1 must be non-negative, got -1");
    }
}

TEST(StorageViewTest, RejectsInvalidNumericType) {
    std::int64_t dims[]{4};
    const FTrainStorageView input{nullptr, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_INVALID, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted an invalid numeric type";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), "StorageView numeric_type 0 is invalid");
    }
}

TEST(StorageViewTest, RejectsUnknownNumericType) {
    std::int64_t dims[]{4};
    constexpr FTrainNumericType first_unknown_numeric_type = static_cast<FTrainNumericType>(FTRAIN_NUMERIC_TYPE_COUNT);
    const FTrainStorageView input{nullptr, dims, nullptr, 1, first_unknown_numeric_type, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted an unknown numeric type";
    }
    catch (const Exception& exception) {
        const std::string expected_message = "StorageView numeric_type " +
                                             std::to_string(static_cast<unsigned int>(first_unknown_numeric_type)) +
                                             " is invalid";
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), expected_message.c_str());
    }
}

TEST(StorageViewTest, RejectsUnknownIndexType) {
    std::int64_t dims[]{4};
    constexpr FTrainIndexType first_unknown_index_type = static_cast<FTrainIndexType>(FTRAIN_INDEX_TYPE_COUNT);
    const FTrainStorageView input{nullptr, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP32, first_unknown_index_type, false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted an unknown index type";
    }
    catch (const Exception& exception) {
        const std::string expected_message = "StorageView index_type " +
                                             std::to_string(static_cast<unsigned int>(first_unknown_index_type)) +
                                             " is unknown";
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), expected_message.c_str());
    }
}

TEST(StorageViewTest, RejectsMissingStrideAndIndexType) {
    std::int64_t dims[]{4};
    const FTrainStorageView input{nullptr, dims, nullptr, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_INVALID,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted storage without a layout description";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(), "StorageView requires strides or a non-invalid index_type");
    }
}

TEST(StorageViewTest, RejectsStridesWithPredefinedIndexType) {
    std::int64_t dims[]{4};
    std::int64_t strides[]{1};
    const FTrainStorageView input{nullptr, dims, strides, 1, FTRAIN_NUMERIC_TYPE_FP32, FTRAIN_INDEX_TYPE_CONTINUOUS,
                                  false};

    try {
        static_cast<void>(StorageView{input});
        FAIL() << "StorageView accepted strides with a predefined index type";
    }
    catch (const Exception& exception) {
        EXPECT_EQ(exception.getResult().getStatus(), FTRAIN_STATUS_INVALID_ARGUMENT);
        EXPECT_STREQ(exception.what(),
                     "StorageView strides must be null when index_type 1 defines a predefined layout");
    }
}

}  // namespace
}  // namespace ftrain
