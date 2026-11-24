// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flash_train/common.h"

#include "flash_train/storage_view.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {
namespace {

static_assert(!std::is_default_constructible_v<Tensor>);
static_assert(std::is_nothrow_constructible_v<Tensor, StorageView&&>);
static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(noexcept(std::declval<const Tensor&>().getStorageView()));

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

}  // namespace
}  // namespace ftrain
