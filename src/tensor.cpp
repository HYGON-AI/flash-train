#include "flash_train/tensor.hpp"

#include "flash_train/error.hpp"

namespace ftrain {

GroupedTensorStorage::GroupedTensorStorage(std::uint32_t num_groups, StorageView&& data,
                                           std::optional<StorageView>&& offsets, std::vector<StorageView>&& dim_sizes,
                                           std::vector<StorageView>&& strides)
    : num_groups_(num_groups), data_(std::move(data)), offsets_(std::move(offsets)), dim_sizes_(std::move(dim_sizes)),
      strides_(std::move(strides)) {
    const std::size_t rank = data_.getDims().size();
    if (!dim_sizes_.empty() && dim_sizes_.size() != rank) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "GroupedTensorStorage dim_sizes count %zu must be zero or match data rank %zu",
                        dim_sizes_.size(), rank);
    }
    if (!strides_.empty() && strides_.size() != rank) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                        "GroupedTensorStorage strides count %zu must be zero or match data rank %zu", strides_.size(),
                        rank);
    }

    if (offsets_.has_value()) {
        const std::vector<std::int64_t>& offsets_dims = offsets_->getDims();
        if (offsets_dims.size() != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GroupedTensorStorage offsets rank %zu must be 1",
                            offsets_dims.size());
        }
        if (offsets_dims[0] != static_cast<std::int64_t>(num_groups_)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "GroupedTensorStorage offsets length %lld must match num_groups %u",
                            static_cast<long long>(offsets_dims[0]), static_cast<unsigned int>(num_groups_));
        }
    }

    for (std::size_t dimension = 0; dimension < dim_sizes_.size(); ++dimension) {
        const std::vector<std::int64_t>& metadata_dims = dim_sizes_[dimension].getDims();
        if (metadata_dims.size() != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GroupedTensorStorage dim_sizes[%zu] rank %zu must be 1",
                            dimension, metadata_dims.size());
        }
        if (metadata_dims[0] != static_cast<std::int64_t>(num_groups_)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "GroupedTensorStorage dim_sizes[%zu] length %lld must match num_groups %u", dimension,
                            static_cast<long long>(metadata_dims[0]), static_cast<unsigned int>(num_groups_));
        }
    }

    for (std::size_t dimension = 0; dimension < strides_.size(); ++dimension) {
        const std::vector<std::int64_t>& metadata_dims = strides_[dimension].getDims();
        if (metadata_dims.size() != 1) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GroupedTensorStorage strides[%zu] rank %zu must be 1",
                            dimension, metadata_dims.size());
        }
        if (metadata_dims[0] != static_cast<std::int64_t>(num_groups_)) {
            throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT,
                            "GroupedTensorStorage strides[%zu] length %lld must match num_groups %u", dimension,
                            static_cast<long long>(metadata_dims[0]), static_cast<unsigned int>(num_groups_));
        }
    }
}

const TensorStorage& Tensor::getStorage() const {
    if (!storage_.has_value()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "Tensor concrete storage has not been set");
    }
    return *storage_;
}

const TensorListStorage& TensorList::getStorage() const {
    if (!storage_.has_value()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "TensorList concrete storage has not been set");
    }
    return *storage_;
}

const GroupedTensorStorage& GroupedTensor::getStorage() const {
    if (!storage_.has_value()) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GroupedTensor concrete storage has not been set");
    }
    return *storage_;
}

}  // namespace ftrain
