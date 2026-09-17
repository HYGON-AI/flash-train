// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_STORAGE_VIEW_HPP_
#define FTRAIN_STORAGE_VIEW_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"

namespace ftrain {

// One tensor's storage description: a memory address plus copied dimension,
// stride, and type metadata. The address is never owned or freed; the caller
// keeps it valid while it is in use. References returned by getters stay
// valid until this object is modified or destroyed. An empty getDims()
// describes a scalar holding one element and always has a non-null address.
class StorageView {
  public:
    // Validates storage_view and copies its metadata. Non-null dims and
    // strides arrays must each hold at least num_dims readable elements during
    // the call. Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT when:
    //   - numeric_type is FTRAIN_NUMERIC_TYPE_INVALID or out of range
    //   - index_type is out of range
    //   - num_dims is zero and memory is null (a scalar needs an address)
    //   - num_dims is nonzero and dims is null
    //   - a dimension is negative
    //   - strides is null while index_type is FTRAIN_INDEX_TYPE_INVALID
    //   - strides is non-null while index_type names a predefined layout
    // Metadata allocation failure throws std::bad_alloc.
    explicit StorageView(const FTrainStorageView& storage_view);

    void* getMemory() const noexcept { return memory_; }

    const std::vector<std::int64_t>& getDims() const noexcept { return dims_; }

    const std::vector<std::int64_t>& getStrides() const noexcept { return strides_; }

    FTrainNumericType getNumericType() const noexcept { return numeric_type_; }

    FTrainIndexType getIndexType() const noexcept { return index_type_; }

    bool isHostMemory() const noexcept { return is_host_memory_; }

  private:
    void* memory_;
    std::vector<std::int64_t> dims_;
    std::vector<std::int64_t> strides_;
    FTrainNumericType numeric_type_;
    FTrainIndexType index_type_;
    bool is_host_memory_;
};

}  // namespace ftrain

#endif
