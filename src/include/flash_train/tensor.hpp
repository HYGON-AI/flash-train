// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_TENSOR_HPP_
#define FTRAIN_TENSOR_HPP_

#include <utility>

#include "flash_train/storage_view.hpp"

namespace ftrain {

// One Tensor argument value: a single StorageView. The memory address is
// not owned.
class Tensor final {
  public:
    explicit Tensor(StorageView&& storage_view) noexcept : storage_view_(std::move(storage_view)) {}

    const StorageView& getStorageView() const noexcept { return storage_view_; }

  private:
    StorageView storage_view_;
};

}  // namespace ftrain

#endif
