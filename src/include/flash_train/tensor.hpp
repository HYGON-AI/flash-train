#ifndef FTRAIN_TENSOR_HPP_
#define FTRAIN_TENSOR_HPP_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

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

// One TensorList argument value: an ordered list of StorageViews; an empty
// list is a valid empty TensorList. The memory addresses are not owned.
class TensorList final {
  public:
    explicit TensorList(std::vector<StorageView>&& storage_views) noexcept : storage_views_(std::move(storage_views)) {}

    const std::vector<StorageView>& getStorageViews() const noexcept { return storage_views_; }

  private:
    std::vector<StorageView> storage_views_;
};

// One GroupedTensor argument value: a data view plus optional per-group
// offsets, dimension sizes, and strides. Empty dim_sizes and strides vectors
// mean the descriptions were not supplied. The memory addresses are not
// owned.
class GroupedTensor final {
  public:
    // Moves the supplied descriptions into this object and checks their
    // shapes. dim_sizes and strides may be empty; otherwise each must hold
    // exactly rank(data) entries, and every entry must be a rank-one array of
    // num_groups elements. offsets, when supplied, must also be a rank-one
    // array of num_groups elements. num_groups may be zero. A shape violation
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    GroupedTensor(std::uint32_t num_groups, StorageView&& data, std::optional<StorageView>&& offsets,
                  std::vector<StorageView>&& dim_sizes, std::vector<StorageView>&& strides);

    std::uint32_t getNumGroups() const noexcept { return num_groups_; }

    const StorageView& getData() const noexcept { return data_; }

    const std::optional<StorageView>& getOffsets() const noexcept { return offsets_; }

    const std::vector<StorageView>& getDimSizes() const noexcept { return dim_sizes_; }

    const std::vector<StorageView>& getStrides() const noexcept { return strides_; }

  private:
    std::uint32_t num_groups_;
    StorageView data_;
    std::optional<StorageView> offsets_;
    std::vector<StorageView> dim_sizes_;
    std::vector<StorageView> strides_;
};

}  // namespace ftrain

#endif
