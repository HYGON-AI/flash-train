#ifndef FTRAIN_TENSOR_HPP_
#define FTRAIN_TENSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "flash_train/storage_view.hpp"

namespace ftrain {

// One Tensor's storage: a single StorageView. The memory address is not owned.
class TensorStorage final {
  public:
    explicit TensorStorage(StorageView&& storage_view) noexcept : storage_view_(std::move(storage_view)) {}

    const StorageView& getStorageView() const noexcept { return storage_view_; }

  private:
    StorageView storage_view_;
};

// One TensorList's storage: an ordered list of StorageViews. An empty list is
// a valid empty TensorList. The memory addresses are not owned.
class TensorListStorage final {
  public:
    explicit TensorListStorage(std::vector<StorageView>&& storage_views) noexcept
        : storage_views_(std::move(storage_views)) {}

    const std::vector<StorageView>& getStorageViews() const noexcept { return storage_views_; }

  private:
    std::vector<StorageView> storage_views_;
};

// One GroupedTensor's storage: a data view plus optional per-group offsets,
// dimension sizes, and strides. Empty dim_sizes and strides vectors mean the
// descriptions were not supplied. The memory addresses are not owned.
class GroupedTensorStorage final {
  public:
    // Moves the supplied descriptions into this object and checks their
    // shapes. dim_sizes and strides may be empty; otherwise each must hold
    // exactly rank(data) entries, and every entry must be a rank-one array of
    // num_groups elements. offsets, when supplied, must also be a rank-one
    // array of num_groups elements. num_groups may be zero. A shape violation
    // throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT.
    GroupedTensorStorage(std::uint32_t num_groups, StorageView&& data, std::optional<StorageView>&& offsets,
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

// One Tensor argument slot. Unset until setStorage() is called. The
// getStorage() reference stays valid until the next setStorage() call or
// destruction.
class Tensor final {
  public:
    Tensor() noexcept = default;

    bool isSet() const noexcept { return storage_.has_value(); }

    void setStorage(TensorStorage&& storage) noexcept { storage_.emplace(std::move(storage)); }

    // Returns the concrete storage. An unset Tensor throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const TensorStorage& getStorage() const;

  private:
    std::optional<TensorStorage> storage_;
};

// One TensorList argument slot. Unset until setStorage() is called; an unset
// TensorList is distinct from a set empty TensorListStorage. The getStorage()
// reference stays valid until the next setStorage() call or destruction.
class TensorList final {
  public:
    TensorList() noexcept = default;

    bool isSet() const noexcept { return storage_.has_value(); }

    void setStorage(TensorListStorage&& storage) noexcept { storage_.emplace(std::move(storage)); }

    // Returns the concrete storage. An unset TensorList throws Exception with
    // FTRAIN_STATUS_INVALID_ARGUMENT.
    const TensorListStorage& getStorage() const;

  private:
    std::optional<TensorListStorage> storage_;
};

// One GroupedTensor argument slot. Unset until setStorage() is called; an
// unset GroupedTensor is distinct from set storage with zero groups. The
// getStorage() reference stays valid until the next setStorage() call or
// destruction.
class GroupedTensor final {
  public:
    GroupedTensor() noexcept = default;

    bool isSet() const noexcept { return storage_.has_value(); }

    void setStorage(GroupedTensorStorage&& storage) noexcept { storage_.emplace(std::move(storage)); }

    // Returns the concrete storage. An unset GroupedTensor throws Exception
    // with FTRAIN_STATUS_INVALID_ARGUMENT.
    const GroupedTensorStorage& getStorage() const;

  private:
    std::optional<GroupedTensorStorage> storage_;
};

}  // namespace ftrain

#endif
